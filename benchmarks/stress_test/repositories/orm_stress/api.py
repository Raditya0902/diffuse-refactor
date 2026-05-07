"""
api.py
HTTP-like API handler layer — the top-level orchestrator.
Depends on: config.py, models.py, validators.py, services.py
"""
from __future__ import annotations

from typing import Optional

from config import DEFAULT_PAGE_SIZE, STATUS_ACTIVE, ENABLE_USER_RATE_LIMIT, RATE_LIMIT_PER_MINUTE
from models import User, Product, Order, OrderItem
from validators import UserValidator, ProductValidator, OrderValidator, ValidationError
from services import UserService, ProductService, OrderService, ServiceError
from repository import UserRepository, ProductRepository, OrderRepository


class RateLimiter:
    """Simple in-memory rate limiter per user."""
    def __init__(self, limit: int = RATE_LIMIT_PER_MINUTE):
        self._counters: dict[str, int] = {}
        self._limit = limit

    def is_allowed(self, user_id: str) -> bool:
        count = self._counters.get(user_id, 0)
        if count >= self._limit:
            return False
        self._counters[user_id] = count + 1
        return True

    def reset(self, user_id: str) -> None:
        self._counters.pop(user_id, None)


class ApiResponse:
    """Standardized API response envelope."""
    def __init__(self, success: bool, data=None, error: Optional[str] = None, code: int = 200):
        self.success = success
        self.data = data
        self.error = error
        self.code = code

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "code": self.code,
        }

    @classmethod
    def ok(cls, data) -> "ApiResponse":
        return cls(success=True, data=data, code=200)

    @classmethod
    def created(cls, data) -> "ApiResponse":
        return cls(success=True, data=data, code=201)

    @classmethod
    def not_found(cls, msg: str) -> "ApiResponse":
        return cls(success=False, error=msg, code=404)

    @classmethod
    def bad_request(cls, msg: str) -> "ApiResponse":
        return cls(success=False, error=msg, code=400)

    @classmethod
    def rate_limited(cls) -> "ApiResponse":
        return cls(success=False, error="Rate limit exceeded", code=429)

    @classmethod
    def server_error(cls, msg: str) -> "ApiResponse":
        return cls(success=False, error=msg, code=500)


class UserApi:
    """Handles all /users/* endpoints."""

    def __init__(self, user_service: UserService, rate_limiter: RateLimiter):
        self.user_service = user_service
        self.rate_limiter = rate_limiter

    def create(self, payload: dict) -> ApiResponse:
        try:
            user = self.user_service.create_user(
                username=payload.get("username", ""),
                email=payload.get("email", ""),
            )
            return ApiResponse.created(user.to_dict())
        except ServiceError as e:
            return ApiResponse.bad_request(e.message)

    def get(self, user_id: str) -> ApiResponse:
        try:
            user = self.user_service.get_user(user_id)
            return ApiResponse.ok(user.to_dict())
        except ServiceError as e:
            return ApiResponse.not_found(e.message)

    def list_active(self, page: int = 1) -> ApiResponse:
        users = self.user_service.list_active_users(page=page)
        return ApiResponse.ok([u.to_dict() for u in users])

    def deactivate(self, user_id: str) -> ApiResponse:
        try:
            user = self.user_service.deactivate_user(user_id)
            return ApiResponse.ok(user.to_dict())
        except ServiceError as e:
            return ApiResponse.bad_request(e.message)

    def delete(self, user_id: str) -> ApiResponse:
        try:
            self.user_service.delete_user(user_id)
            return ApiResponse.ok({"deleted": user_id})
        except ServiceError as e:
            return ApiResponse.not_found(e.message)


class ProductApi:
    """Handles all /products/* endpoints."""

    def __init__(self, product_service: ProductService):
        self.product_service = product_service

    def create(self, payload: dict) -> ApiResponse:
        try:
            product = self.product_service.create_product(
                name=payload.get("name", ""),
                price=float(payload.get("price", 0)),
                stock=int(payload.get("stock", 0)),
                category=payload.get("category", "general"),
            )
            return ApiResponse.created(product.to_dict())
        except ServiceError as e:
            return ApiResponse.bad_request(e.message)

    def get(self, product_id: str) -> ApiResponse:
        try:
            product = self.product_service.get_product(product_id)
            return ApiResponse.ok(product.to_dict())
        except ServiceError as e:
            return ApiResponse.not_found(e.message)

    def list(self, category: Optional[str] = None, page: int = 1) -> ApiResponse:
        products = self.product_service.list_products(category=category, page=page)
        return ApiResponse.ok([p.to_dict() for p in products])

    def apply_discount(self, category: str, discount_pct: float) -> ApiResponse:
        try:
            products = self.product_service.apply_category_discount(category, discount_pct)
            return ApiResponse.ok({"updated": len(products)})
        except (ServiceError, ValueError) as e:
            return ApiResponse.bad_request(str(e))


class OrderApi:
    """Handles all /orders/* endpoints."""

    def __init__(self, order_service: OrderService, rate_limiter: RateLimiter):
        self.order_service = order_service
        self.rate_limiter = rate_limiter

    def place(self, user_id: str, payload: dict) -> ApiResponse:
        if ENABLE_USER_RATE_LIMIT and not self.rate_limiter.is_allowed(user_id):
            return ApiResponse.rate_limited()
        try:
            order = self.order_service.place_order(
                user_id=user_id,
                items_data=payload.get("items", []),
            )
            return ApiResponse.created(order.to_dict())
        except ServiceError as e:
            return ApiResponse.bad_request(e.message)

    def get(self, order_id: str) -> ApiResponse:
        try:
            summary = self.order_service.get_order_summary(order_id)
            return ApiResponse.ok(summary)
        except ServiceError as e:
            return ApiResponse.not_found(e.message)

    def cancel(self, order_id: str) -> ApiResponse:
        try:
            order = self.order_service.cancel_order(order_id)
            return ApiResponse.ok(order.to_dict())
        except ServiceError as e:
            return ApiResponse.bad_request(e.message)

    def complete(self, order_id: str) -> ApiResponse:
        try:
            order = self.order_service.complete_order(order_id)
            return ApiResponse.ok(order.to_dict())
        except ServiceError as e:
            return ApiResponse.bad_request(e.message)

    def list_for_user(self, user_id: str, page: int = 1) -> ApiResponse:
        try:
            orders = self.order_service.list_user_orders(user_id, page=page)
            return ApiResponse.ok([o.to_dict() for o in orders])
        except ServiceError as e:
            return ApiResponse.not_found(e.message)


def build_app() -> dict:
    """
    Dependency-injection wiring: construct the full object graph.
    Returns dict of {api_name: api_handler}.
    """
    user_repo = UserRepository()
    product_repo = ProductRepository()
    order_repo = OrderRepository()

    user_validator = UserValidator()
    product_validator = ProductValidator()
    order_validator = OrderValidator(product_validator=product_validator)

    rate_limiter = RateLimiter()

    user_svc = UserService(user_repo=user_repo, user_validator=user_validator)
    product_svc = ProductService(product_repo=product_repo, product_validator=product_validator)
    order_svc = OrderService(
        order_repo=order_repo,
        product_repo=product_repo,
        order_validator=order_validator,
        product_service=product_svc,
        user_service=user_svc,
    )

    return {
        "users": UserApi(user_service=user_svc, rate_limiter=rate_limiter),
        "products": ProductApi(product_service=product_svc),
        "orders": OrderApi(order_service=order_svc, rate_limiter=rate_limiter),
    }
