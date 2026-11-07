"""Allowlisted tool registry.

Single source of truth for the tools the agent (and the
``POST /api/v1/tools/invoke`` endpoint) may execute. Each tool has a
Pydantic argument schema for validation, runs real service logic
against the database, is always scoped to the authenticated customer,
and is audit-logged to ``tool_execution_logs``.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError
from sqlalchemy.orm import Session

from app.db.models import ToolExecutionLog
from app.services import order_service, product_service, returns_service
from app.services.exceptions import NotFoundError, ValidationError


class LookupOrderArgs(BaseModel):
    order_id: str = Field(description="The order id, e.g. ORD-1001")


class CheckReturnEligibilityArgs(BaseModel):
    order_id: str = Field(description="The order id to evaluate for return eligibility, e.g. ORD-1002")


class GetProductDetailsArgs(BaseModel):
    product_id: str = Field(description="The product id, e.g. PROD-101")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., dict]  # (db, customer_id, **validated_args) -> dict


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "lookup_order": ToolSpec(
        name="lookup_order",
        description=(
            "Look up one of the customer's orders by order id. Returns status, "
            "tracking, carrier, estimated delivery, and purchased items."
        ),
        args_model=LookupOrderArgs,
        handler=lambda db, customer_id, order_id: order_service.lookup_order(db, customer_id, order_id),
    ),
    "check_return_eligibility": ToolSpec(
        name="check_return_eligibility",
        description=(
            "Check whether the items in one of the customer's orders can be returned, "
            "based on store policy, delivery date, and product category."
        ),
        args_model=CheckReturnEligibilityArgs,
        handler=lambda db, customer_id, order_id: returns_service.check_return_eligibility(db, customer_id, order_id),
    ),
    "get_product_details": ToolSpec(
        name="get_product_details",
        description="Get structured catalog details (price, specs, stock) for a product id.",
        args_model=GetProductDetailsArgs,
        handler=lambda db, customer_id, product_id: product_service.get_product_details(db, product_id),
    ),
}


def execute_tool(
    db: Session,
    *,
    tool_name: str,
    arguments: dict,
    customer_id: int,
    session_id: str | None = None,
) -> dict:
    """Validate and execute an allowlisted tool, with audit logging.

    Returns ``{"success": bool, "result": ... | None, "error": str | None}``
    so the agent can relay failures (bad ID, not found) conversationally
    instead of crashing the graph.
    """
    spec = TOOL_REGISTRY.get(tool_name)
    if spec is None:
        raise ValidationError(f"Unknown tool '{tool_name}'. Allowed: {sorted(TOOL_REGISTRY)}")

    try:
        args = spec.args_model(**(arguments or {}))
    except PydanticValidationError as exc:
        raise ValidationError(f"Invalid arguments for {tool_name}: {exc.errors()}") from exc

    start = time.perf_counter()
    success, result, error = True, None, None
    try:
        result = spec.handler(db, customer_id, **args.model_dump())
    except (NotFoundError, ValidationError) as exc:
        success, error = False, str(exc)
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        db.add(
            ToolExecutionLog(
                session_id=session_id,
                customer_id=customer_id,
                tool_name=tool_name,
                arguments=args.model_dump(),
                result=result if success else None,
                success=success,
                error=error,
                duration_ms=round(duration_ms, 2),
            )
        )
        db.commit()

    return {"success": success, "result": result, "error": error}
