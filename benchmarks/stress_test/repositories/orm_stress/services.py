"""
services.py
Business logic layer orchestrating repositories and validators.
Depends on: config.py, models.py, repository.py, validators.py
Used by: api.py
"""
from __future__ import annotations

from typing import Optional

from config import (
    STATUS_ACTIVE, STATUS_INACTIVE, STATUS_PENDING,
    STATUS_CANCELLED, STATUS_COMPLETED,
    ENABLE_PRICE_CACHE, DEFAULT_PAGE_SIZE,
)
from models import User, Product, Order, OrderItem, generate_id, utcnow
from repository import UserRepository, ProductRepository, OrderRepository
from validators import UserValidator, ProductValidator, OrderValidator, ValidationError


class ServiceError(Exception):
    """Raised by service layer on logical failures."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class UserService:
    """Business logic for user management."""

    def __init__(
        self,
        user_repo: UserRepository,
        user_validator: UserValidator,
    ):
        self.user_repo = user_repo
        self.user_validator = user_validator

    def create_user(self, username: str, email: str) -> User:
        user = User(username=username, email=email, status=STATUS_ACTIVE)
        errors = self.user_validator.validate(user)
        if errors:
            raise ServiceError("VALIDATION_FAILED", "; ".join(e.message for e in errors))
        existing = self.user_repo.get_by_email(email)
        if existing:
            raise ServiceError("EMAIL_EXISTS", f"Email {email!r} is already registered")
        return self.user_repo.save(user)

    def get_user(self, user_id: str) -> User:
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise ServiceError("NOT_FOUND", f"User {user_id!r} not found")
        return user

    def deactivate_user(self, user_id: str) -> User:
        user = self.get_user(user_id)
        if not user.is_active():
            raise ServiceError("ALREADY_INACTIVE", f"User {user_id!r} is already inactive")
        user.deactivate()
        return self.user_repo.save(user)

    def reactivate_user(self, user_id: str) -> User:
        user = self.get_user(user_id)
        if user.is_active():
            raise ServiceError("ALREADY_ACTIVE", f"User {user_id!r} is already active")
        user.reactivate()
        return self.user_repo.save(user)

    def list_active_users(self, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> list[User]:
        return self.user_repo.list_active(page=page, page_size=page_size)

    def delete_user(self, user_id: str) -> bool:
        self.get_user(user_id)  # raises NOT_FOUND if missing
        return self.user_repo.delete(user_id)


class ProductService:
    """Business logic for product management."""

    def __init__(
        self,
        product_repo: ProductRepository,
        product_validator: ProductValidator,
        price_cache: Optional[dict] = None,
    ):
        self.product_repo = product_repo
        self.product_validator = product_validator
        self._price_cache: dict[str, float] = price_cache if price_cache is not None else {}
        self._cache_enabled = ENABLE_PRICE_CACHE

    def create_product(self, name: str, price: float, stock: int, category: str = "general") -> Product:
        product = Product(name=name, price=price, stock=stock, category=category, status=STATUS_ACTIVE)
        errors = self.product_validator.validate(product)
        if errors:
            raise ServiceError("VALIDATION_FAILED", "; ".join(e.message for e in errors))
        saved = self.product_repo.save(product)
        if self._cache_enabled:
            self._price_cache[saved.id] = saved.price
        return saved

    def get_product(self, product_id: str) -> Product:
        product = self.product_repo.get_by_id(product_id)
        if not product:
            raise ServiceError("NOT_FOUND", f"Product {product_id!r} not found")
        return product

    def get_price(self, product_id: str) -> float:
        if self._cache_enabled and product_id in self._price_cache:
            return self._price_cache[product_id]
        product = self.get_product(product_id)
        if self._cache_enabled:
            self._price_cache[product_id] = product.price
        return product.price

    def update_stock(self, product_id: str, delta: int) -> Product:
        product = self.get_product(product_id)
        if delta < 0:
            product.reduce_stock(abs(delta))
        else:
            product.restore_stock(delta)
        return self.product_repo.save(product)

    def apply_category_discount(self, category: str, discount_pct: float) -> list[Product]:
        products = self.product_repo.list_by_category(category)
        updated = []
        for product in products:
            product.price = product.apply_discount(discount_pct)
            if self._cache_enabled:
                self._price_cache[product.id] = product.price
            self.product_repo.save(product)
            updated.append(product)
        return updated

    def list_products(self, category: Optional[str] = None, page: int = 1) -> list[Product]:
        if category:
            return self.product_repo.list_by_category(category, page=page)
        return self.product_repo.list_all(page=page)


class OrderService:
    """Business logic for order management."""

    def __init__(
        self,
        order_repo: OrderRepository,
        product_repo: ProductRepository,
        order_validator: OrderValidator,
        product_service: ProductService,
        user_service: UserService,
    ):
        self.order_repo = order_repo
        self.product_repo = product_repo
        self.order_validator = order_validator
        self.product_service = product_service
        self.user_service = user_service

    def place_order(self, user_id: str, items_data: list[dict]) -> Order:
        # Verify user exists and is active
        user = self.user_service.get_user(user_id)
        if not user.is_active():
            raise ServiceError("USER_INACTIVE", f"User {user_id!r} is not active")

        # Build OrderItems with current prices
        items = []
        product_ids = [d["product_id"] for d in items_data]
        products = self.product_repo.get_by_ids(product_ids)

        for d in items_data:
            pid = d["product_id"]
            qty = d.get("quantity", 1)
            price = self.product_service.get_price(pid)
            items.append(OrderItem(product_id=pid, quantity=qty, unit_price=price))

        order = Order(user_id=user_id, items=items, status=STATUS_PENDING)
        errors = self.order_validator.validate(order, products)
        if errors:
            raise ServiceError("VALIDATION_FAILED", "; ".join(e.message for e in errors))

        # Deduct stock for each item
        for item in items:
            self.product_service.update_stock(item.product_id, -item.quantity)

        return self.order_repo.save(order)

    def cancel_order(self, order_id: str) -> Order:
        order = self.order_repo.get_by_id(order_id)
        if not order:
            raise ServiceError("NOT_FOUND", f"Order {order_id!r} not found")
        order.cancel()
        # Restore stock
        for item in order.items:
            self.product_service.update_stock(item.product_id, item.quantity)
        return self.order_repo.save(order)

    def complete_order(self, order_id: str) -> Order:
        order = self.order_repo.get_by_id(order_id)
        if not order:
            raise ServiceError("NOT_FOUND", f"Order {order_id!r} not found")
        order.complete()
        return self.order_repo.save(order)

    def get_order(self, order_id: str) -> Order:
        order = self.order_repo.get_by_id(order_id)
        if not order:
            raise ServiceError("NOT_FOUND", f"Order {order_id!r} not found")
        return order

    def list_user_orders(self, user_id: str, page: int = 1) -> list[Order]:
        self.user_service.get_user(user_id)  # ensure user exists
        return self.order_repo.list_by_user(user_id, page=page)

    def get_order_summary(self, order_id: str) -> dict:
        order = self.get_order(order_id)
        return {
            **order.to_dict(),
            "is_cancellable": order.status == STATUS_PENDING,
            "is_completed": order.status == STATUS_COMPLETED,
        }
