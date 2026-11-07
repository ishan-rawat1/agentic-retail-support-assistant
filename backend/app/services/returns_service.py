"""Return-eligibility rules.

The constants here mirror the published Returns & Refunds Policy
document (``app/data/docs/returns-policy.md``). Eligibility is computed
from the actual order record — status, delivery date, item categories —
never from canned responses.
"""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.db.models import Order
from app.services.order_service import get_order_for_customer

# Policy: days from delivery, by category. Mirrors returns-policy.md.
DEFAULT_RETURN_WINDOW_DAYS = 30
CATEGORY_RETURN_WINDOWS = {"electronics": 15}
NON_RETURNABLE_CATEGORIES = {"gift-cards"}


def _days_since(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).days


def check_return_eligibility(db: Session, customer_id: int, order_id: str) -> dict:
    """Evaluate return eligibility for every item in an order."""
    order: Order = get_order_for_customer(db, customer_id, order_id)

    if order.status == "cancelled":
        return {
            "order_id": order.external_id,
            "order_status": order.status,
            "eligible": False,
            "reason": "This order was cancelled, so there is nothing to return.",
            "items": [],
        }

    if order.status in ("processing", "shipped") or order.delivered_at is None:
        reason = (
            "Orders can only be returned after delivery. "
            + (
                "Since this order hasn't shipped yet, you can still cancel it free of charge."
                if order.status == "processing"
                else "This order is still in transit; once it is delivered, the return window begins."
            )
        )
        return {
            "order_id": order.external_id,
            "order_status": order.status,
            "eligible": False,
            "reason": reason,
            "items": [],
        }

    days_since_delivery = _days_since(order.delivered_at)
    items = []
    any_eligible = False
    for item in order.items:
        product = item.product
        window = CATEGORY_RETURN_WINDOWS.get(product.category, DEFAULT_RETURN_WINDOW_DAYS)

        if product.category in NON_RETURNABLE_CATEGORIES:
            eligible, reason = False, "Gift cards are non-returnable and non-refundable per policy."
        elif product.final_sale:
            eligible, reason = False, "Final sale / clearance items are not eligible for return."
        elif days_since_delivery <= window:
            eligible = True
            reason = (
                f"Within the {window}-day return window for {product.category} "
                f"(delivered {days_since_delivery} days ago; "
                f"{window - days_since_delivery} days remaining)."
            )
            any_eligible = True
        else:
            eligible = False
            reason = (
                f"The {window}-day return window for {product.category} has expired "
                f"(delivered {days_since_delivery} days ago)."
            )

        items.append(
            {
                "product_id": product.external_id,
                "name": product.name,
                "category": product.category,
                "return_window_days": window,
                "eligible": eligible,
                "reason": reason,
            }
        )

    return {
        "order_id": order.external_id,
        "order_status": order.status,
        "delivered_at": order.delivered_at.strftime("%Y-%m-%d"),
        "days_since_delivery": days_since_delivery,
        "eligible": any_eligible,
        "reason": (
            "At least one item in this order is eligible for return."
            if any_eligible
            else "No items in this order are currently eligible for return."
        ),
        "refund_info": "Refunds go to the original payment method within 5-7 business days after the warehouse receives the item.",
        "items": items,
    }
