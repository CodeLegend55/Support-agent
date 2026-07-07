
import csv
import os
import re
from typing import Optional

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "orders.csv")

ORDER_ID_RE = re.compile(r"\bORD\s?-?\s?(\d{3,7})\b", re.IGNORECASE)


def _load_orders() -> list[dict]:
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_ORDERS = _load_orders()
_ORDERS_BY_ID = {o["order_id"].upper(): o for o in _ORDERS}


def normalize_order_id(raw: str) -> Optional[str]:
    """Turn '1004', 'ord1004', 'ORD-1004' etc into 'ORD1004' if it matches the pattern."""
    raw = raw.strip().upper()
    if raw in _ORDERS_BY_ID:
        return raw
    m = ORDER_ID_RE.search(raw)
    if m:
        candidate = f"ORD{m.group(1)}"
        if candidate in _ORDERS_BY_ID:
            return candidate
    # bare numeric id, e.g. "1004"
    if raw.isdigit():
        candidate = f"ORD{raw}"
        if candidate in _ORDERS_BY_ID:
            return candidate
    return None


def extract_order_ids(text: str, existing_only: bool = True) -> list[str]:
    """Find all order-id-shaped tokens mentioned in free text and normalize them.
    By default only returns ones that actually exist in the dataset; pass
    existing_only=False to also surface ids that look valid but aren't found,
    so the caller can report a clean 'not found' instead of silently ignoring them.
    """
    found = []
    for m in ORDER_ID_RE.finditer(text):
        candidate = f"ORD{m.group(1)}"
        if candidate in found:
            continue
        if existing_only and candidate not in _ORDERS_BY_ID:
            continue
        found.append(candidate)
    return found


def lookup_order(order_id: str) -> dict:
    """
    Deterministic single-order lookup.
    Returns {"found": True, "order": {...}} or {"found": False, "order_id": ...}
    """
    norm = normalize_order_id(order_id)
    if norm is None:
        return {"found": False, "order_id": order_id}
    return {"found": True, "order": dict(_ORDERS_BY_ID[norm])}


def search_orders(status: Optional[str] = None, customer_name: Optional[str] = None,
                   category: Optional[str] = None) -> dict:
    """
    Deterministic filtered search across all orders. All filters are optional
    and case-insensitive substring matches.
    """
    results = _ORDERS
    if status:
        results = [o for o in results if o["status"].lower() == status.lower()]
    if customer_name:
        results = [o for o in results if customer_name.lower() in o["customer_name"].lower()]
    if category:
        results = [o for o in results if category.lower() in o["category"].lower()]
    return {"count": len(results), "orders": results}


if __name__ == "__main__":
    print(lookup_order("ORD1004"))
    print(lookup_order("1099"))  # not a real id
    print(search_orders(status="Delivered")["count"])
