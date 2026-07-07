# Support Agent — RAG + Structured Lookup Router

A small CLI support agent for a fictional Indian e-commerce seller. It answers:

1. **Knowledge questions** ("What's your return policy for electronics?") by
   retrieving from the four policy markdown docs.
2. **Data questions** ("What's the status of order ORD1004?") by looking up
   the `orders.csv` dataset through a deterministic function — never guessed.

The interesting part of this assignment is the **routing decision**, so that's
where most of the design effort went.

## Architecture

```
Support-agent/
├── data/
│   ├── orders.csv                  # structured order records
│   └── policies/*.md               # the four policy documents
├── orders_tool.py                  # deterministic order lookup/search
├── retrieval.py                    # lightweight TF-IDF RAG over the policy docs
├── agent.py                        # routing + orchestration + CLI
└── README.md
```

### `orders_tool.py` — the data path
Plain Python over `orders.csv`. `lookup_order(order_id)` normalizes id formats
(`"1004"`, `"ord-1004"`, `"ORD1004"` all resolve) and returns the exact stored
row, or `found=False`. `search_orders(status=..., customer_name=..., category=...)`
handles group/count questions. **No LLM ever touches these values before
they're fetched** — the model only relays what the function returned.

### `retrieval.py` — the knowledge path
Each markdown file is split into chunks on `## ` section headers. Chunks are
scored against the query with a small from-scratch TF-IDF + cosine similarity
implementation (no external embedding API, so it runs fully offline). The top
chunks are returned with their source file and section title, so answers stay
traceable to the specific policy line they came from. A `min_score` cutoff
means genuinely uncovered questions get an honest "not in our docs" instead of
a forced answer.

### `agent.py` — the router
This is the core design decision. Rather than a single upstream classifier
that commits to one path before looking at anything, **routing is delegated
to Claude via tool use**. The model is given three tools:

- `lookup_order` — the only source of truth for one order's fields
- `search_orders` — the only source of truth for order counts/groups
- `search_policy` — the only source of truth for policy rules

and a system prompt that says, in effect: *never state an order fact or a
policy fact unless a tool just told you it — call whichever tool(s) the
question needs, including both, before answering.*

Why tool-use routing instead of a hardcoded if/else:

- **Hybrid questions are common and need both paths.** "I got ORD1004
  delivered, can I still return it?" needs `lookup_order` (to learn it's
  Electronics, delivered on 2026-06-25) *and* `search_policy` (electronics
  get a 3-day window, not the general 7). A single upstream classifier has to
  pick one bucket; tool-use lets the model call one tool, read the result,
  and decide it needs the other too.
- **Grounding is enforced structurally, not just by prompt.** The model can't
  produce a status or a policy number except by relaying JSON that came back
  from a real function call, which keeps hallucinated order data or invented
  policy terms out of the loop.
- **It degrades gracefully.** If a tool returns nothing relevant (e.g.
  `search_policy` finds no matching chunk, or `lookup_order` finds no such
  ID), the system prompt tells Claude to say so rather than fill the gap.

### Fallback mode (no API key needed)
If `ANTHROPIC_API_KEY` isn't set, `agent.py` runs a rule-based router
(`rule_based_route`) that does the same job without an LLM composing the final
sentence — regex order-ID detection, policy keyword matching, and simple
aggregate-query detection ("how many orders are Cancelled?"). This exists so
the retrieval + lookup machinery is testable and demoable without any
credentials, and so you can see the routing logic in its most explicit form.
It's intentionally simpler than the tool-use path and won't handle every
phrasing as gracefully — it's a fallback, not the main design.

## Running it

```bash
cd support_agent

# Full agent (Claude decides routing via tool use)
export ANTHROPIC_API_KEY=sk-ant-...
python3 agent.py

# Fallback mode (no key required)
python3 agent.py
```

Example session:
```
You: What's the status of order ORD1004?
Agent: Order ORD1004 (Smartwatch Pro) was delivered on 2026-06-25 for ₹4999, paid via UPI.

You: Can I return it?
Agent: Since it's Electronics, the return window is 3 days from delivery (shorter than
the general 7-day window), and it needs to be unused with original packaging. If it's
been more than 3 days since 2026-06-25, it may no longer be eligible for a standard
return — but if it arrived damaged/defective, that's a separate 48-hour report window
with free pickup and full refund regardless.

You: How many orders are Cancelled?
Agent: 2 orders currently have status "Cancelled": ORD1008 (Running Shoes) and
ORD1020 (Electric Kettle).
```

## Chunking & retrieval strategy

**Chunking:** each policy markdown file is split on `## ` (H2) headers, so
one chunk = one section (e.g. "General Return Window", "COD Charges"). This
granularity was chosen deliberately over alternatives:

- *Whole-file chunks* would be too coarse — a query about the electronics
  return window would drag in unrelated content about exchanges and damaged
  items from the same file, diluting the similarity score and the eventual
  answer.
- *Fixed-size token chunks* (e.g. 200 tokens with overlap) would risk
  splitting a bullet point's condition from its consequence (e.g. separating
  "Electronics ... 3 days" from the sentence that explains why), since these
  docs are short, hand-written FAQs, not long-form prose.
- Section-level chunking keeps each chunk semantically self-contained (one
  topic per chunk) while staying short enough that a single chunk is a
  complete, quotable answer unit — which is exactly the retrieval grain these
  four short FAQ docs call for.

**Vector store — none, by design.** Retrieval uses a small from-scratch
TF-IDF + cosine similarity implementation (`retrieval.py`), not a vector DB
or an embeddings API. Reasoning:

- The corpus is four short FAQ files (~15 sections total). A vector store
  (Chroma, FAISS, Pinecone, etc.) is built for scaling to thousands–millions
  of chunks with approximate nearest-neighbor search; at this scale it's
  pure overhead with no accuracy benefit.
- These are short, keyword-dense policy statements ("COD fee", "3-day return
  window", "GST invoice") where lexical overlap between query and answer is
  usually strong — TF-IDF's core assumption holds well here. Dense embeddings
  earn their keep on paraphrase-heavy or long-form text; that's not this corpus.
- Zero external dependencies and zero network calls for retrieval — it runs
  in a plain Python process, which matters for a small support-bot repo that
  should be easy to run and audit.
- Trade-off, stated plainly: this **will not** catch pure paraphrases with no
  shared vocabulary (e.g. "can I get my money back" instead of "refund"), and
  it doesn't scale gracefully past a few dozen documents. If the doc set grew
  or got more varied in phrasing, the natural upgrade path is to keep the
  same chunking but swap the scoring function for real sentence embeddings
  (e.g. via the Claude/OpenAI embeddings endpoint) plus a real vector index.

A `min_score` cutoff (0.05) is applied on top of the ranking: if nothing
clears it, `retrieve()` returns an empty list, which the agent is instructed
to treat as "not covered in our docs" rather than force an answer.

## How the routing decision is made

See the "The router" section above for the full explanation — tool-use
routing rather than a single upstream classifier, because hybrid questions
need both a data lookup and a policy lookup chained together, and grounding
is enforced structurally (the model can only state facts a tool just
returned). Concretely, for each incoming question:

1. Claude (or the rule-based fallback) inspects the question.
2. If it references a specific order (an ID, "my order", "when will it
   arrive") → call `lookup_order` / `search_orders`.
3. If it references a rule, fee, timeline, or eligibility condition → call
   `search_policy`.
4. If it needs both (see the chaining example below) → call both, in
   whichever order makes sense, and reason over the combined result.
5. If neither tool returns anything usable → say so rather than guess.

## Chaining walkthrough: "Can I still return order ORD1004?"

This is the case that most clearly needs both steps in sequence, since the
correct return window *depends on* a field only the order lookup can supply
(the product category):

```
Step 1 — lookup_order("ORD1004")
  → {found: True, order: {product: "Smartwatch Pro", category: "Electronics",
     status: "Delivered", order_date: "2026-06-25", ...}}

Step 2 — now that we know it's Electronics, search_policy("return policy electronics")
  → returns_and_refunds.md § General Return Window:
    "Electronics and appliances have a shorter return window of 3 days
     due to manufacturer terms."
  → returns_and_refunds.md § Conditions for Return:
    "Product must be unused ... Products marked 'Final Sale' cannot be returned."

Step 3 — compose answer using BOTH facts: category (from step 1) determines
which policy clause (from step 2) applies, and the order_date (from step 1)
lets the answer compute whether the window has actually passed.
```

Actual output (fallback mode — see table below for the full transcript):
```
Order ORD1004 (Smartwatch Pro, Electronics): status = Delivered, amount = ₹4999,
ordered on 2026-06-25, payment via UPI.

[returns_and_refunds.md — General Return Window]: Most products can be returned
within 7 days of delivery... Electronics and appliances have a shorter return
window of 3 days due to manufacturer terms...

[returns_and_refunds.md — Conditions for Return]: Product must be unused,
unwashed, and with original tags/packaging intact...
```
The full Claude tool-use agent would go one step further and explicitly state
the conclusion ("3-day window from 2026-06-25 has passed, so it's likely no
longer eligible for a standard return") since it reasons over both tool
results in prose; the fallback mode surfaces the same two facts but leaves
the arithmetic to the reader.

## Test questions: expected vs. actual

All "actual" outputs below are from the rule-based fallback mode (no API key
required), run directly against the current code. The "expected" column
describes the correct *routing* decision and the facts the answer must
contain — not exact wording, since the full Claude agent will phrase things
more fluently than the fallback's raw fact-dump.

| # | Question | Expected routing / facts | Actual (fallback mode) | Result |
|---|----------|---------------------------|--------------------------|--------|
| 1 | What is the status of order ORD1004? | **Data only.** Status=Delivered, ₹4999, 2026-06-25, UPI | `Order ORD1004 (Smartwatch Pro, Electronics): status = Delivered, amount = ₹4999, ordered on 2026-06-25, payment via UPI.` | ✅ |
| 2 | What is your return policy for electronics? | **Knowledge only.** 3-day window (shorter than general 7-day), from Conditions for Return | Returns both correct chunks: 3-day electronics window + return conditions | ✅ |
| 3 | Can I still return order ORD1004? | **Hybrid.** Needs order's category+date, then category-specific policy | Returns order record (category=Electronics, date) + both return-policy chunks | ✅ (see chaining walkthrough above) |
| 4 | Is cash on delivery available and what fee applies? | **Knowledge only.** ₹40 COD fee, available under ₹5000 | Returns COD Charges chunk (₹40 fee) + Accepted Payment Methods chunk | ✅ |
| 5 | How many orders are Cancelled? | **Data only (aggregate).** Count = 2 | `Found 2 order(s) with status 'Cancelled'.` | ✅ |
| 6 | What is the status of order ORD9999? *(edge case: invalid ID)* | **Data only**, but ID doesn't exist → clear "not found", no invented status | `I couldn't find an order matching 'ORD9999' in our system.` | ✅ |
| 7 | Do you offer international shipping to Dubai? *(edge case: not covered)* | Docs only say "we do not ship internationally" — no Dubai-specific answer exists | Retrieves Domestic Shipping chunk, which does state international shipping isn't offered — correctly grounded, doesn't invent Dubai-specific details | ✅ |
| 8 | What is the capital of France? *(edge case: no doc answer)* | Nothing relevant in docs → should say so, not hallucinate policy content | `I couldn't find anything in our policy docs covering that — please contact support.` | ✅ |
| 9 | Can I combine a coupon with a festive sale discount? | **Knowledge only.** No, unless explicitly stated | Returns Discounts and Coupons chunk (correct answer: "cannot be combined ... unless explicitly stated") | ✅ |
| 10 | Is order ORD1008 eligible for cancellation? | **Hybrid.** Order status is Cancelled already (so N/A) + cancellation policy (only before shipped) | Returns order record (status=Cancelled) + Order Cancellation policy chunk | ✅ |

## Edge cases handled explicitly

- **Invalid order ID** (`ORD9999`, test #6): `orders_tool.extract_order_ids(...,
  existing_only=False)` still recognizes the token as order-ID-*shaped*, so
  `lookup_order` is called (not skipped), and its `found=False` result is
  turned into an honest "I couldn't find that order" — never a fabricated
  status.
- **No answer in the docs** (test #8, and #7 as a softer version): the
  TF-IDF `min_score` cutoff means `retrieve()` returns nothing for queries
  with no real lexical overlap with the corpus, and the agent is instructed
  (fallback: hardcoded message; Claude path: system prompt rule) to say the
  docs don't cover it rather than force a plausible-sounding but invented
  answer.
- **Chaining both steps** (tests #3 and #10): see the walkthrough above —
  order metadata resolves *which* policy clause applies before the policy
  lookup even happens.

*A bug worth noting from testing:* an earlier version of the aggregate-query
detector matched the substring `"count"` inside `"discount"`, so "coupon...
discount" questions were misrouted to an order-count lookup instead of policy
retrieval. Fixed with word-boundary regex matching — included here because it's
a good example of why routing logic needs adversarial test questions, not just
happy-path ones.

## Design notes / trade-offs

- **TF-IDF instead of embeddings**: the policy corpus is four short files —
  a from-scratch lexical retriever is transparent, dependency-free, and
  plenty accurate at this scale. It would need to be swapped for real
  embeddings if the doc set grew much larger or became less keyword-aligned
  with likely questions.
- **Tool-use routing instead of intent classification**: a classifier is
  simpler but forces a single-path decision upfront; tool-use lets the model
  chain lookups and handles hybrid questions without special-casing them.
- **Hard separation between "fetching facts" and "writing the answer"**: the
  model composes prose, but every number, date, and status in that prose
  traces back to a tool result in the same turn.
