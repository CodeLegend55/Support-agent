
import json
import os
import sys

from orders_tool import lookup_order, search_orders, extract_order_ids
from retrieval import retrieve

MODEL = "claude-sonnet-4-6"

TOOLS = [
    {
        "name": "lookup_order",
        "description": (
            "Look up a single order by its order ID (e.g. ORD1004). Returns the "
            "exact stored record (status, product, amount, date, payment method, "
            "pincode) or found=False if no such order exists. This is the ONLY "
            "valid source of truth for order-specific facts — never guess them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "e.g. 'ORD1004' or '1004'"}
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_orders",
        "description": (
            "Search orders by optional filters: status, customer_name, category. "
            "Use this for questions about groups of orders rather than one specific ID "
            "(e.g. 'how many orders are Cancelled')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "customer_name": {"type": "string"},
                "category": {"type": "string"},
            },
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Retrieve the most relevant excerpts from the seller's policy documents "
            "(shipping, returns & refunds, payments & pricing, account & support). "
            "Use this for any question about rules, timelines, fees, or eligibility "
            "that isn't about one customer's specific order data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the policy question or topic"}
            },
            "required": ["query"],
        },
    },
]

SYSTEM_PROMPT = """You are a support agent for an Indian e-commerce seller.

You have three tools:
- lookup_order: the ONLY source of truth for a specific order's status/amount/date/etc.
- search_orders: for questions about groups/counts of orders.
- search_policy: the ONLY source of truth for policy rules (shipping, returns, payments, account).

Rules you must follow:
1. Never state an order's status, amount, date, or other field unless it came from a
   lookup_order/search_orders tool result in this conversation. If the tool says
   found=False, tell the user you couldn't find that order — do not invent one.
2. Never state a specific policy rule, number, fee, or timeline unless it came from a
   search_policy tool result. If search_policy returns nothing relevant, say the policy
   docs don't cover it and suggest contacting support.
3. Some questions are hybrid and need both tools — e.g. "can I return ORD1004?" needs
   lookup_order (to see the product category and delivery date) AND search_policy
   (to check the return window and conditions for that category). Call both before answering.
4. Be concise and cite which policy section you're drawing from when relevant.
5. If the question is small talk or unrelated to orders/policies, just answer normally
   without calling tools.
"""


def _execute_tool(name: str, tool_input: dict) -> dict:
    if name == "lookup_order":
        return lookup_order(tool_input["order_id"])
    if name == "search_orders":
        return search_orders(
            status=tool_input.get("status"),
            customer_name=tool_input.get("customer_name"),
            category=tool_input.get("category"),
        )
    if name == "search_policy":
        return {"chunks": retrieve(tool_input["query"])}
    return {"error": f"unknown tool {name}"}


def run_with_claude(client, user_query: str, verbose: bool = False) -> str:
    """Full agentic path: Claude decides which tool(s) to call, we execute them
    deterministically, feed results back, repeat until it gives a final answer."""
    messages = [{"role": "user", "content": user_query}]

    for _ in range(6):  # safety cap on tool-use round trips
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"  [tool call] {block.name}({block.input})", file=sys.stderr)
                result = _execute_tool(block.name, block.input)
                if verbose:
                    print(f"  [tool result] {json.dumps(result)[:300]}", file=sys.stderr)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
        messages.append({"role": "user", "content": tool_results})

    return "Sorry, I couldn't resolve this after several tool calls — please contact support."


# ---------------------------------------------------------------------------
# Rule-based fallback router (no API key needed). Demonstrates the same
# routing/grounding logic without an LLM writing the final sentence.
# ---------------------------------------------------------------------------

POLICY_KEYWORDS = [
    "return", "refund", "exchange", "shipping", "ship", "deliver", "delivery",
    "cod", "coupon", "discount", "payment", "gst", "invoice", "cancel",
    "loyalty", "points", "password", "login", "account", "lost", "damaged",
    "defective", "policy", "warranty", "tracking",
]


import re as _re

STATUS_VALUES = ["delivered", "shipped", "processing", "confirmed", "cancelled"]
AGGREGATE_SIGNALS = ["how many", "count", "list orders", "list all", "which orders"]
_AGGREGATE_RE = _re.compile(r"\b(?:" + "|".join(_re.escape(s) for s in AGGREGATE_SIGNALS) + r")\b")


def rule_based_route(query: str) -> dict:
    order_ids_any = extract_order_ids(query, existing_only=False)
    q_lower = query.lower()
    wants_policy = any(kw in q_lower for kw in POLICY_KEYWORDS)

    is_aggregate = bool(_AGGREGATE_RE.search(q_lower))
    status_filter = next((s for s in STATUS_VALUES if _re.search(rf"\b{s}\b", q_lower)), None)

    return {
        "order_ids": order_ids_any,
        "is_aggregate": is_aggregate,
        "status_filter": status_filter,
        # Only fall back to a policy search when the question doesn't mention
        # an order/aggregate at all AND doesn't hit any policy keywords either.
        "wants_policy": (wants_policy or not order_ids_any) and not is_aggregate,
    }


def _format_chunk(c: dict) -> str:
    # Show the section body (minus the '## Header' line itself), trimmed.
    lines = [ln for ln in c["text"].splitlines() if not ln.startswith("##")]
    body = " ".join(ln.strip("- ").strip() for ln in lines if ln.strip())
    return f"[{c['source']} — {c['section']}]: {body}"


def run_fallback(user_query: str) -> str:
    route = rule_based_route(user_query)
    parts = []

    for oid in route["order_ids"]:
        result = lookup_order(oid)
        if result["found"]:
            o = result["order"]
            parts.append(
                f"Order {o['order_id']} ({o['product']}, {o['category']}): status = {o['status']}, "
                f"amount = ₹{o['amount_inr']}, ordered on {o['order_date']}, "
                f"payment via {o['payment_method']}."
            )
        else:
            parts.append(f"I couldn't find an order matching '{oid}' in our system.")

    if route["is_aggregate"]:
        result = search_orders(status=route["status_filter"])
        label = f" with status '{route['status_filter'].title()}'" if route["status_filter"] else ""
        parts.append(f"Found {result['count']} order(s){label}.")

    if route["wants_policy"]:
        chunks = retrieve(user_query)
        if chunks:
            for c in chunks[:2]:
                parts.append(_format_chunk(c))
        elif not route["order_ids"]:
            parts.append("I couldn't find anything in our policy docs covering that — please contact support.")

    return "\n\n".join(parts) if parts else "Could you clarify your question?"


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = None
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            print("anthropic package not installed; run `pip install anthropic`. Falling back to rule-based mode.\n")

    mode = "Claude tool-use agent" if client else "rule-based fallback (set ANTHROPIC_API_KEY for full agent)"
    print(f"Support Agent ready [{mode}]. Type 'quit' to exit.\n")

    while True:
        try:
            q = input("You: ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"quit", "exit"}:
            break
        if client:
            answer = run_with_claude(client, q, verbose=True)
        else:
            answer = run_fallback(q)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
