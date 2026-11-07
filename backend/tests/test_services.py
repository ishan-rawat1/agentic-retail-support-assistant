"""Unit tests: order lookup, return eligibility, authorization scoping."""

import pytest

from app.services import order_service, returns_service
from app.services.exceptions import NotFoundError, ValidationError


class TestOrderLookup:
    def test_lookup_own_order(self, db_session, alice_id):
        order = order_service.lookup_order(db_session, alice_id, "ORD-1001")
        assert order["order_id"] == "ORD-1001"
        assert order["status"] == "shipped"
        assert order["tracking_number"] == "TRK-84512093"
        names = {item["name"] for item in order["items"]}
        assert names == {"Aurora Wireless Headphones", "LumenGlow Smart Lamp"}

    def test_lookup_is_case_insensitive_and_trims(self, db_session, alice_id):
        order = order_service.lookup_order(db_session, alice_id, "  ord-1001 ")
        assert order["order_id"] == "ORD-1001"

    def test_invalid_format_rejected(self, db_session, alice_id):
        with pytest.raises(ValidationError):
            order_service.lookup_order(db_session, alice_id, "1001")
        with pytest.raises(ValidationError):
            order_service.lookup_order(db_session, alice_id, "ORD-1001; DROP TABLE orders")

    def test_unknown_order_not_found(self, db_session, alice_id):
        with pytest.raises(NotFoundError):
            order_service.lookup_order(db_session, alice_id, "ORD-9999")

    def test_other_customers_order_indistinguishable_from_missing(
        self, db_session, alice_id, bob_id
    ):
        # ORD-1004 belongs to Bob. Alice must get the same error as for a
        # nonexistent order — no enumeration of other customers' ids.
        with pytest.raises(NotFoundError) as exc_other:
            order_service.lookup_order(db_session, alice_id, "ORD-1004")
        with pytest.raises(NotFoundError) as exc_missing:
            order_service.lookup_order(db_session, alice_id, "ORD-9999")
        assert type(exc_other.value) is type(exc_missing.value)
        # Bob himself can see it.
        assert order_service.lookup_order(db_session, bob_id, "ORD-1004")["order_id"] == "ORD-1004"


class TestReturnEligibility:
    def test_delivered_within_window_is_eligible(self, db_session, alice_id):
        result = returns_service.check_return_eligibility(db_session, alice_id, "ORD-1002")
        assert result["eligible"] is True
        shoes = result["items"][0]
        assert shoes["category"] == "footwear"
        assert shoes["eligible"] is True
        assert shoes["return_window_days"] == 30

    def test_expired_window_is_ineligible(self, db_session, alice_id):
        result = returns_service.check_return_eligibility(db_session, alice_id, "ORD-1003")
        assert result["eligible"] is False
        assert "expired" in result["items"][0]["reason"]

    def test_in_transit_order_not_returnable_yet(self, db_session, alice_id):
        result = returns_service.check_return_eligibility(db_session, alice_id, "ORD-1001")
        assert result["eligible"] is False
        assert result["order_status"] == "shipped"
        assert "delivery" in result["reason"].lower()

    def test_processing_order_suggests_cancellation(self, db_session, bob_id):
        result = returns_service.check_return_eligibility(db_session, bob_id, "ORD-1005")
        assert result["eligible"] is False
        assert "cancel" in result["reason"].lower()

    def test_electronics_window_and_gift_card_rules(self, db_session, bob_id):
        # ORD-1004: headphones (electronics, delivered 5 days ago -> within
        # 15-day window) + gift cards (never returnable).
        result = returns_service.check_return_eligibility(db_session, bob_id, "ORD-1004")
        by_category = {item["category"]: item for item in result["items"]}
        assert by_category["electronics"]["eligible"] is True
        assert by_category["electronics"]["return_window_days"] == 15
        assert by_category["gift-cards"]["eligible"] is False
        assert result["eligible"] is True  # at least one item eligible

    def test_expired_electronics_and_final_sale(self, db_session):
        from sqlalchemy import select

        from app.db.models import Customer

        carol_id = db_session.execute(
            select(Customer.id).where(Customer.api_token == "demo-carol")
        ).scalar_one()
        # ORD-1006 delivered 20 days ago: electronics window (15d) expired,
        # final-sale blanket always blocked.
        result = returns_service.check_return_eligibility(db_session, carol_id, "ORD-1006")
        assert result["eligible"] is False
        reasons = {item["product_id"]: item["reason"] for item in result["items"]}
        assert "expired" in reasons["PROD-106"]
        assert "final sale" in reasons["PROD-107"].lower()

    def test_cancelled_order_has_nothing_to_return(self, db_session):
        from sqlalchemy import select

        from app.db.models import Customer

        carol_id = db_session.execute(
            select(Customer.id).where(Customer.api_token == "demo-carol")
        ).scalar_one()
        result = returns_service.check_return_eligibility(db_session, carol_id, "ORD-1007")
        assert result["eligible"] is False
        assert "cancelled" in result["reason"]

    def test_cannot_check_other_customers_order(self, db_session, alice_id):
        with pytest.raises(NotFoundError):
            returns_service.check_return_eligibility(db_session, alice_id, "ORD-1004")
