"""Product catalog queries."""

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Product
from app.services.exceptions import NotFoundError, ValidationError

PRODUCT_ID_RE = re.compile(r"^PROD-\d{3,8}$")


def get_product_details(db: Session, product_id: str) -> dict:
    product_id = (product_id or "").strip().upper()
    if not PRODUCT_ID_RE.match(product_id):
        raise ValidationError(
            f"'{product_id}' is not a valid product id. Expected format: PROD-101."
        )
    product = db.execute(
        select(Product).where(Product.external_id == product_id)
    ).scalar_one_or_none()
    if product is None:
        raise NotFoundError(f"No product found with id {product_id}.")
    return {
        "product_id": product.external_id,
        "name": product.name,
        "category": product.category,
        "price": product.price,
        "description": product.description,
        "specs": product.specs,
        "in_stock": product.stock_quantity > 0,
        "stock_quantity": product.stock_quantity,
        "final_sale": product.final_sale,
    }


def list_products(db: Session) -> list[dict]:
    products = db.execute(select(Product).order_by(Product.external_id)).scalars()
    return [
        {
            "product_id": p.external_id,
            "name": p.name,
            "category": p.category,
            "price": p.price,
            "in_stock": p.stock_quantity > 0,
        }
        for p in products
    ]
