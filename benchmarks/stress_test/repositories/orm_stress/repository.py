"""
repository.py
In-memory data access layer (simulates ORM persistence).
Depends on: config.py, models.py
Used by: services.py, api.py
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from config import (
    DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE,
    STATUS_ACTIVE, STATUS_INACTIVE, ENABLE_AUDIT_LOG,
)
from models import User, Product, Order, OrderItem, generate_id, utcnow


class AuditEntry:
    """Single audit log entry."""
    def __init__(self, entity_type: str, entity_id: str, action: str, diff: dict):
        self.id = generate_id()
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.action = action
        self.diff = diff
        self.timestamp = utcnow()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "action": self.action,
            "diff": self.diff,
            "timestamp": self.timestamp.isoformat(),
        }


class UserRepository:
    """CRUD operations for User entities."""

    def __init__(self, audit_log: bool = ENABLE_AUDIT_LOG):
        self._store: dict[str, User] = {}
        self._email_index: dict[str, str] = {}   # email -> user_id
        self._audit: list[AuditEntry] = []
        self._audit_enabled = audit_log

    def save(self, user: User) -> User:
        is_new = user.id not in self._store
        self._store[user.id] = user
        self._email_index[user.email] = user.id
        if self._audit_enabled:
            self._audit.append(AuditEntry("user", user.id, "create" if is_new else "update", user.to_dict()))
        return user

    def get_by_id(self, user_id: str) -> Optional[User]:
        return self._store.get(user_id)

    def get_by_email(self, email: str) -> Optional[User]:
        uid = self._email_index.get(email)
        return self._store.get(uid) if uid else None

    def list_active(self, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> list[User]:
        page_size = min(page_size, MAX_PAGE_SIZE)
        active = [u for u in self._store.values() if u.is_active()]
        start = (page - 1) * page_size
        return active[start: start + page_size]

    def delete(self, user_id: str) -> bool:
        user = self._store.pop(user_id, None)
        if user:
            self._email_index.pop(user.email, None)
            if self._audit_enabled:
                self._audit.append(AuditEntry("user", user_id, "delete", {}))
            return True
        return False

    def count(self) -> int:
        return len(self._store)

    def get_audit_log(self) -> list[AuditEntry]:
        return list(self._audit)


class ProductRepository:
    """CRUD operations for Product entities."""

    def __init__(self, audit_log: bool = ENABLE_AUDIT_LOG):
        self._store: dict[str, Product] = {}
        self._category_index: dict[str, list[str]] = defaultdict(list)
        self._audit: list[AuditEntry] = []
        self._audit_enabled = audit_log

    def save(self, product: Product) -> Product:
        is_new = product.id not in self._store
        self._store[product.id] = product
        if product.id not in self._category_index[product.category]:
            self._category_index[product.category].append(product.id)
        if self._audit_enabled:
            self._audit.append(AuditEntry("product", product.id, "create" if is_new else "update", product.to_dict()))
        return product

    def get_by_id(self, product_id: str) -> Optional[Product]:
        return self._store.get(product_id)

    def get_by_ids(self, product_ids: list[str]) -> dict[str, Product]:
        return {pid: self._store[pid] for pid in product_ids if pid in self._store}

    def list_by_category(self, category: str, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> list[Product]:
        page_size = min(page_size, MAX_PAGE_SIZE)
        ids = self._category_index.get(category, [])
        products = [self._store[pid] for pid in ids if pid in self._store]
        available = [p for p in products if p.is_available()]
        start = (page - 1) * page_size
        return available[start: start + page_size]

    def list_all(self, only_active: bool = True, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> list[Product]:
        page_size = min(page_size, MAX_PAGE_SIZE)
        products = list(self._store.values())
        if only_active:
            products = [p for p in products if p.is_available()]
        start = (page - 1) * page_size
        return products[start: start + page_size]

    def delete(self, product_id: str) -> bool:
        product = self._store.pop(product_id, None)
        if product:
            cat = self._category_index.get(product.category, [])
            if product_id in cat:
                cat.remove(product_id)
            if self._audit_enabled:
                self._audit.append(AuditEntry("product", product_id, "delete", {}))
            return True
        return False

    def count(self) -> int:
        return len(self._store)


class OrderRepository:
    """CRUD operations for Order entities."""

    def __init__(self, audit_log: bool = ENABLE_AUDIT_LOG):
        self._store: dict[str, Order] = {}
        self._user_index: dict[str, list[str]] = defaultdict(list)
        self._audit: list[AuditEntry] = []
        self._audit_enabled = audit_log

    def save(self, order: Order) -> Order:
        is_new = order.id not in self._store
        self._store[order.id] = order
        if order.id not in self._user_index[order.user_id]:
            self._user_index[order.user_id].append(order.id)
        if self._audit_enabled:
            self._audit.append(AuditEntry("order", order.id, "create" if is_new else "update", order.to_dict()))
        return order

    def get_by_id(self, order_id: str) -> Optional[Order]:
        return self._store.get(order_id)

    def list_by_user(self, user_id: str, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> list[Order]:
        page_size = min(page_size, MAX_PAGE_SIZE)
        order_ids = self._user_index.get(user_id, [])
        orders = [self._store[oid] for oid in order_ids if oid in self._store]
        start = (page - 1) * page_size
        return orders[start: start + page_size]

    def count_by_user(self, user_id: str) -> int:
        return len(self._user_index.get(user_id, []))

    def delete(self, order_id: str) -> bool:
        order = self._store.pop(order_id, None)
        if order:
            user_orders = self._user_index.get(order.user_id, [])
            if order_id in user_orders:
                user_orders.remove(order_id)
            if self._audit_enabled:
                self._audit.append(AuditEntry("order", order_id, "delete", {}))
            return True
        return False

    def count(self) -> int:
        return len(self._store)
