"""Order lookup — always scoped to the authenticated customer."""

import re

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.db.models import Order
from app.services.exceptions import NotFoundError, ValidationError

ORDER_ID_RE = re.compile(r"^ORD-\d{3,8}$")


def _format_date(dt) -> str | None:
    return dt.strftime("%Y-%m-%d") if dt else None


def get_order_for_customer(db: Session, customer_id: int, order_external_id: str) -> Order:
    """Fetch an order by external ID, enforcing customer ownership.

    Raises the same NotFoundError whether the order doesn't exist or
    belongs to someone else, so order IDs cannot be enumerated.
    """
    order_external_id = (order_external_id or "").strip().upper()
    if not ORDER_ID_RE.match(order_external_id):
        raise ValidationError(
            f"'{order_external_id}' is not a valid order id. Expected format: ORD-1234."
        )
    stmt = (
        select(Order)
        .options(joinedload(Order.items))
        .where(Order.external_id == order_external_id, Order.customer_id == customer_id)
    )
    order = db.execute(stmt).unique().scalar_one_or_none()
    if order is None:
        raise NotFoundError(f"No order {order_external_id} found for your account.")
    return order


def order_to_dict(db: Session, order: Order) -> dict:
    """Serialize an order (with items) for tool results and the API."""
    return {
        "order_id": order.external_id,
        "status": order.status,
        "placed_at": _format_date(order.placed_at),
        "shipped_at": _format_date(order.shipped_at),
        "delivered_at": _format_date(order.delivered_at),
        "estimated_delivery": _format_date(order.estimated_delivery),
        "tracking_number": order.tracking_number,
        "carrier": order.carrier,
        "total_amount": order.total_amount,
        "items": [
            {
                "product_id": item.product.external_id,
                "name": item.product.name,
                "category": item.product.category,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
            }
            for item in order.items
        ],
    }


def lookup_order(db: Session, customer_id: int, order_id: str) -> dict:
    order = get_order_for_customer(db, customer_id, order_id)
    return order_to_dict(db, order)


def list_customer_order_ids(db: Session, customer_id: int) -> list[str]:
    stmt = select(Order.external_id).where(Order.customer_id == customer_id).order_by(Order.placed_at.desc())
    return list(db.execute(stmt).scalars())
