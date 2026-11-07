"""Seed the database with realistic, documentation-consistent demo data.

Order dates are computed relative to "now" so that return-eligibility
demos stay meaningful regardless of when the project is run:

* ORD-1001 (Alice)  shipped, in transit            -> not returnable yet
* ORD-1002 (Alice)  delivered 10 days ago, shoes   -> returnable (30-day window)
* ORD-1003 (Alice)  delivered 40 days ago          -> window expired
* ORD-1004 (Bob)    delivered 5 days ago, incl.
                    electronics + gift card        -> mixed eligibility
* ORD-1005 (Bob)    processing                     -> cancellable, not returnable
* ORD-1006 (Carol)  delivered 20 days ago, incl.
                    electronics (15-day window)    -> electronics expired, final-sale blocked
* ORD-1007 (Carol)  cancelled
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.db.base import Base, SessionLocal, engine
from app.db.models import Customer, Order, OrderItem, Product

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


PRODUCTS = [
    dict(external_id="PROD-101", name="Aurora Wireless Headphones", category="electronics",
         price=129.99, stock_quantity=42, final_sale=False,
         description="Over-ear Bluetooth 5.3 headphones with active noise cancellation and 40-hour battery life.",
         specs={"battery_life": "40 hours", "bluetooth": "5.3", "weight": "254 g",
                "noise_cancellation": "Active (ANC)", "warranty": "1 year limited"}),
    dict(external_id="PROD-102", name="TrailBlaze Running Shoes", category="footwear",
         price=89.50, stock_quantity=120, final_sale=False,
         description="Lightweight trail running shoes with breathable mesh upper and grippy outsole.",
         specs={"weight": "280 g", "drop": "6 mm", "sizes": "US 6-13", "warranty": "90 days"}),
    dict(external_id="PROD-103", name="NimbusDesk Standing Desk", category="furniture",
         price=449.00, stock_quantity=15, final_sale=False,
         description="Electric sit-stand desk with dual motors, 48x30 inch bamboo top, memory presets.",
         specs={"height_range": "28-48 in", "max_load": "120 kg", "surface": "bamboo",
                "warranty": "5 years frame, 2 years electronics"}),
    dict(external_id="PROD-104", name="EverBrew Coffee Maker", category="appliances",
         price=74.25, stock_quantity=63, final_sale=False,
         description="10-cup programmable drip coffee maker with thermal carafe and auto-brew scheduling.",
         specs={"capacity": "10 cups", "carafe": "thermal stainless", "warranty": "2 years"}),
    dict(external_id="PROD-105", name="SummitPack 45L Hiking Backpack", category="outdoor",
         price=119.00, stock_quantity=34, final_sale=False,
         description="45-liter internal-frame hiking backpack with rain cover and hydration sleeve.",
         specs={"volume": "45 L", "weight": "1.4 kg", "frame": "internal aluminum", "warranty": "Lifetime"}),
    dict(external_id="PROD-106", name="LumenGlow Smart Lamp", category="electronics",
         price=45.99, stock_quantity=88, final_sale=False,
         description="Wi-Fi smart table lamp with 16M colors, schedules, and voice-assistant support.",
         specs={"connectivity": "Wi-Fi 2.4 GHz", "brightness": "800 lm", "warranty": "1 year limited"}),
    dict(external_id="PROD-107", name="CozyCloud Weighted Blanket", category="home",
         price=59.99, stock_quantity=9, final_sale=True,
         description="15 lb weighted blanket, clearance item. Final sale.",
         specs={"weight": "15 lb", "material": "cotton/glass beads"}),
    dict(external_id="PROD-108", name="Retail Gift Card $50", category="gift-cards",
         price=50.00, stock_quantity=9999, final_sale=False,
         description="Digital gift card redeemable on any purchase. Non-returnable per policy.",
         specs={"delivery": "email", "expiry": "none"}),
]

CUSTOMERS = [
    dict(external_id="CUST-001", name="Alice Johnson", email="alice@example.com", api_token="demo-alice"),
    dict(external_id="CUST-002", name="Bob Martinez", email="bob@example.com", api_token="demo-bob"),
    dict(external_id="CUST-003", name="Carol Chen", email="carol@example.com", api_token="demo-carol"),
]


def seed(db: Session, *, reset: bool = True) -> None:
    if reset:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

    products = {p["external_id"]: Product(**p) for p in PRODUCTS}
    customers = {c["external_id"]: Customer(**c) for c in CUSTOMERS}
    db.add_all(products.values())
    db.add_all(customers.values())
    db.flush()

    now = _now()

    def order(ext_id, cust, status, placed_days, items, *, shipped_days=None,
              delivered_days=None, eta_days=None, tracking=None, carrier=None):
        o = Order(
            external_id=ext_id,
            customer_id=customers[cust].id,
            status=status,
            placed_at=now - timedelta(days=placed_days),
            shipped_at=(now - timedelta(days=shipped_days)) if shipped_days is not None else None,
            delivered_at=(now - timedelta(days=delivered_days)) if delivered_days is not None else None,
            estimated_delivery=(now + timedelta(days=eta_days)) if eta_days is not None else None,
            tracking_number=tracking,
            carrier=carrier,
        )
        db.add(o)
        db.flush()
        total = 0.0
        for prod_ext, qty in items:
            p = products[prod_ext]
            db.add(OrderItem(order_id=o.id, product_id=p.id, quantity=qty, unit_price=p.price))
            total += p.price * qty
        o.total_amount = round(total, 2)
        return o

    order("ORD-1001", "CUST-001", "shipped", 4, [("PROD-101", 1), ("PROD-106", 1)],
          shipped_days=2, eta_days=3, tracking="TRK-84512093", carrier="UPS")
    order("ORD-1002", "CUST-001", "delivered", 15, [("PROD-102", 1)],
          shipped_days=13, delivered_days=10, tracking="TRK-77120455", carrier="FedEx")
    order("ORD-1003", "CUST-001", "delivered", 45, [("PROD-104", 1)],
          shipped_days=43, delivered_days=40, tracking="TRK-66234871", carrier="USPS")
    order("ORD-1004", "CUST-002", "delivered", 9, [("PROD-101", 1), ("PROD-108", 2)],
          shipped_days=7, delivered_days=5, tracking="TRK-90331268", carrier="UPS")
    order("ORD-1005", "CUST-002", "processing", 0, [("PROD-103", 1)])
    order("ORD-1006", "CUST-003", "delivered", 25, [("PROD-106", 2), ("PROD-107", 1)],
          shipped_days=23, delivered_days=20, tracking="TRK-55120934", carrier="FedEx")
    order("ORD-1007", "CUST-003", "cancelled", 12, [("PROD-105", 1)])

    db.commit()
    logger.info("Seeded %d products, %d customers, 7 orders", len(products), len(customers))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    db = SessionLocal()
    try:
        seed(db)
        print("Database seeded successfully.")
        print("Demo customers / tokens:")
        for c in CUSTOMERS:
            print(f"  {c['external_id']}  {c['name']:<16} token={c['api_token']}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
