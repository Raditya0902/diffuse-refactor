"""
validators.py
Validation logic for all entity types.
Depends on: config.py, models.py
Used by: services.py, api.py
"""
from __future__ import annotations

import re
from typing import Optional

from config import (
    MAX_USERNAME_LEN, MAX_EMAIL_LEN, MAX_PRODUCT_NAME_LEN,
    MIN_PRICE, MAX_PRICE, MAX_ORDER_ITEMS, VALID_STATUSES,
    STATUS_ACTIVE,
)
from models import User, Product, Order, OrderItem


# ── Regex patterns ─────────────────────────────────────────────────────────────
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')
USERNAME_REGEX = re.compile(r'^[a-zA-Z0-9_\-]{3,}$')


@dataclass_like = None  # placeholder — real validation is class-based here


class ValidationError(Exception):
    """Raised when entity validation fails."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class UserValidator:
    """Validates User entities before persistence."""

    def validate(self, user: User) -> list[ValidationError]:
        errors: list[ValidationError] = []
        errors.extend(self.validate_username(user.username))
        errors.extend(self.validate_email(user.email))
        errors.extend(self.validate_status(user.status))
        return errors

    def validate_username(self, username: str) -> list[ValidationError]:
        errors = []
        if not username:
            errors.append(ValidationError("username", "Username is required"))
        elif len(username) > MAX_USERNAME_LEN:
            errors.append(ValidationError("username", f"Username exceeds {MAX_USERNAME_LEN} characters"))
        elif not USERNAME_REGEX.match(username):
            errors.append(ValidationError("username", "Username contains invalid characters"))
        return errors

    def validate_email(self, email: str) -> list[ValidationError]:
        errors = []
        if not email:
            errors.append(ValidationError("email", "Email is required"))
        elif len(email) > MAX_EMAIL_LEN:
            errors.append(ValidationError("email", f"Email exceeds {MAX_EMAIL_LEN} characters"))
        elif not EMAIL_REGEX.match(email):
            errors.append(ValidationError("email", "Email format is invalid"))
        return errors

    def validate_status(self, status: str) -> list[ValidationError]:
        if status not in VALID_STATUSES:
            return [ValidationError("status", f"Invalid status: {status!r}")]
        return []

    def is_valid(self, user: User) -> bool:
        return len(self.validate(user)) == 0


class ProductValidator:
    """Validates Product entities before persistence."""

    def validate(self, product: Product) -> list[ValidationError]:
        errors: list[ValidationError] = []
        errors.extend(self.validate_name(product.name))
        errors.extend(self.validate_price(product.price))
        errors.extend(self.validate_stock(product.stock))
        errors.extend(self.validate_status(product.status))
        return errors

    def validate_name(self, name: str) -> list[ValidationError]:
        errors = []
        if not name:
            errors.append(ValidationError("name", "Product name is required"))
        elif len(name) > MAX_PRODUCT_NAME_LEN:
            errors.append(ValidationError("name", f"Name exceeds {MAX_PRODUCT_NAME_LEN} characters"))
        return errors

    def validate_price(self, price: float) -> list[ValidationError]:
        errors = []
        if price < MIN_PRICE:
            errors.append(ValidationError("price", f"Price must be >= {MIN_PRICE}"))
        elif price > MAX_PRICE:
            errors.append(ValidationError("price", f"Price must be <= {MAX_PRICE}"))
        return errors

    def validate_stock(self, stock: int) -> list[ValidationError]:
        if stock < 0:
            return [ValidationError("stock", "Stock cannot be negative")]
        return []

    def validate_status(self, status: str) -> list[ValidationError]:
        if status not in VALID_STATUSES:
            return [ValidationError("status", f"Invalid status: {status!r}")]
        return []

    def is_valid(self, product: Product) -> bool:
        return len(self.validate(product)) == 0


class OrderValidator:
    """Validates Order entities before persistence."""

    def __init__(self, product_validator: ProductValidator):
        self.product_validator = product_validator

    def validate(self, order: Order, products: dict[str, Product]) -> list[ValidationError]:
        errors: list[ValidationError] = []
        errors.extend(self.validate_items(order.items, products))
        errors.extend(self.validate_status(order.status))
        if not order.user_id:
            errors.append(ValidationError("user_id", "Order must have a valid user_id"))
        return errors

    def validate_items(
        self,
        items: list[OrderItem],
        products: dict[str, Product],
    ) -> list[ValidationError]:
        errors = []
        if not items:
            errors.append(ValidationError("items", "Order must have at least one item"))
        if len(items) > MAX_ORDER_ITEMS:
            errors.append(ValidationError("items", f"Order cannot exceed {MAX_ORDER_ITEMS} items"))
        for item in items:
            errors.extend(self.validate_item(item, products))
        return errors

    def validate_item(
        self, item: OrderItem, products: dict[str, Product]
    ) -> list[ValidationError]:
        errors = []
        if item.quantity < 1:
            errors.append(ValidationError("quantity", "Quantity must be >= 1"))
        product = products.get(item.product_id)
        if product is None:
            errors.append(ValidationError("product_id", f"Product {item.product_id!r} not found"))
        elif not product.is_available():
            errors.append(ValidationError("product_id", f"Product {item.product_id!r} is unavailable"))
        elif item.quantity > product.stock:
            errors.append(ValidationError("quantity", f"Requested {item.quantity} but only {product.stock} in stock"))
        return errors

    def validate_status(self, status: str) -> list[ValidationError]:
        if status not in VALID_STATUSES:
            return [ValidationError("status", f"Invalid order status: {status!r}")]
        return []

    def is_valid(self, order: Order, products: dict[str, Product]) -> bool:
        return len(self.validate(order, products)) == 0
