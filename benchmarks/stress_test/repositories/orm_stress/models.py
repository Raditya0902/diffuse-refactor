"""
models.py
ORM entity models for the stress-test project.
Depends on: config.py
Used by: repository.py, services.py, validators.py, api.py
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import (
    STATUS_ACTIVE, STATUS_INACTIVE, STATUS_PENDING,
    STATUS_COMPLETED, STATUS_CANCELLED, VALID_STATUSES,
    MAX_USERNAME_LEN, MAX_EMAIL_LEN, MAX_PRODUCT_NAME_LEN,
    MAX_ORDER_ITEMS,
)


def generate_id() -> str:
    """Generate a unique string ID."""
    return str(uuid.uuid4())


def utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.utcnow()


# ── User model ────────────────────────────────────────────────────────────────

@dataclass
class User:
    """Represents a registered user."""
    id: str = field(default_factory=generate_id)
    username: str = ""
    email: str = ""
    status: str = STATUS_ACTIVE
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    metadata: dict = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.status == STATUS_ACTIVE

    def deactivate(self) -> None:
        self.status = STATUS_INACTIVE
        self.updated_at = utcnow()

    def reactivate(self) -> None:
        self.status = STATUS_ACTIVE
        self.updated_at = utcnow()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} status={self.status}>"


# ── Product model ─────────────────────────────────────────────────────────────

@dataclass
class Product:
    """Represents a purchasable product."""
    id: str = field(default_factory=generate_id)
    name: str = ""
    price: float = 0.0
    stock: int = 0
    status: str = STATUS_ACTIVE
    category: str = "general"
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def is_available(self) -> bool:
        return self.status == STATUS_ACTIVE and self.stock > 0

    def reduce_stock(self, quantity: int) -> None:
        if quantity > self.stock:
            raise ValueError(f"Insufficient stock: {self.stock} < {quantity}")
        self.stock -= quantity
        self.updated_at = utcnow()

    def restore_stock(self, quantity: int) -> None:
        self.stock += quantity
        self.updated_at = utcnow()

    def apply_discount(self, pct: float) -> float:
        """Return discounted price without mutating."""
        if not 0.0 <= pct <= 1.0:
            raise ValueError("Discount must be in [0.0, 1.0]")
        return round(self.price * (1.0 - pct), 2)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "stock": self.stock,
            "status": self.status,
            "category": self.category,
        }


# ── OrderItem model ───────────────────────────────────────────────────────────

@dataclass
class OrderItem:
    """A single line-item inside an order."""
    product_id: str = ""
    quantity: int = 1
    unit_price: float = 0.0

    def subtotal(self) -> float:
        return round(self.unit_price * self.quantity, 2)

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "subtotal": self.subtotal(),
        }


# ── Order model ───────────────────────────────────────────────────────────────

@dataclass
class Order:
    """Represents a customer order."""
    id: str = field(default_factory=generate_id)
    user_id: str = ""
    items: list[OrderItem] = field(default_factory=list)
    status: str = STATUS_PENDING
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    notes: str = ""

    def total(self) -> float:
        return round(sum(i.subtotal() for i in self.items), 2)

    def item_count(self) -> int:
        return sum(i.quantity for i in self.items)

    def add_item(self, item: OrderItem) -> None:
        if len(self.items) >= MAX_ORDER_ITEMS:
            raise ValueError(f"Order cannot exceed {MAX_ORDER_ITEMS} items")
        self.items.append(item)
        self.updated_at = utcnow()

    def cancel(self) -> None:
        if self.status == STATUS_COMPLETED:
            raise ValueError("Cannot cancel a completed order")
        self.status = STATUS_CANCELLED
        self.updated_at = utcnow()

    def complete(self) -> None:
        if self.status != STATUS_PENDING:
            raise ValueError("Only pending orders can be completed")
        self.status = STATUS_COMPLETED
        self.updated_at = utcnow()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "status": self.status,
            "total": self.total(),
            "item_count": self.item_count(),
            "items": [i.to_dict() for i in self.items],
            "notes": self.notes,
        }
