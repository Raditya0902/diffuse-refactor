"""
config.py
Global configuration constants for the ORM stress-test project.
Used by: models.py, repository.py, services.py, api.py, validators.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# ── Database constants ─────────────────────────────────────────────────────────
DB_HOST: str = "localhost"
DB_PORT: int = 5432
DB_NAME: str = "stress_db"
DB_POOL_SIZE: int = 10
DB_MAX_OVERFLOW: int = 20
DB_TIMEOUT_SECONDS: int = 30

# ── Pagination ────────────────────────────────────────────────────────────────
DEFAULT_PAGE_SIZE: int = 25
MAX_PAGE_SIZE: int = 200

# ── Validation thresholds ────────────────────────────────────────────────────
MAX_USERNAME_LEN: int = 64
MAX_EMAIL_LEN: int = 255
MAX_PRODUCT_NAME_LEN: int = 128
MIN_PRICE: float = 0.01
MAX_PRICE: float = 999_999.99
MAX_ORDER_ITEMS: int = 50

# ── Feature flags ─────────────────────────────────────────────────────────────
ENABLE_AUDIT_LOG: bool = True
ENABLE_PRICE_CACHE: bool = True
ENABLE_USER_RATE_LIMIT: bool = True
RATE_LIMIT_PER_MINUTE: int = 60

# ── Status codes ──────────────────────────────────────────────────────────────
STATUS_ACTIVE: str = "active"
STATUS_INACTIVE: str = "inactive"
STATUS_PENDING: str = "pending"
STATUS_CANCELLED: str = "cancelled"
STATUS_COMPLETED: str = "completed"
VALID_STATUSES: frozenset[str] = frozenset({
    STATUS_ACTIVE, STATUS_INACTIVE, STATUS_PENDING,
    STATUS_CANCELLED, STATUS_COMPLETED,
})


@dataclass
class DatabaseConfig:
    """Encapsulates database connection parameters."""
    host: str = DB_HOST
    port: int = DB_PORT
    name: str = DB_NAME
    pool_size: int = DB_POOL_SIZE
    max_overflow: int = DB_MAX_OVERFLOW
    timeout: int = DB_TIMEOUT_SECONDS

    def get_dsn(self) -> str:
        return f"postgresql://{self.host}:{self.port}/{self.name}"

    def get_pool_options(self) -> dict:
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "connect_args": {"connect_timeout": self.timeout},
        }

    def is_local(self) -> bool:
        return self.host in ("localhost", "127.0.0.1")

    def clone_with(self, **overrides) -> "DatabaseConfig":
        import copy
        cloned = copy.copy(self)
        for k, v in overrides.items():
            setattr(cloned, k, v)
        return cloned


@dataclass
class AppConfig:
    """Top-level application configuration."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    debug: bool = False
    enable_audit: bool = ENABLE_AUDIT_LOG
    enable_cache: bool = ENABLE_PRICE_CACHE
    rate_limit: int = RATE_LIMIT_PER_MINUTE

    def validate(self) -> list[str]:
        errors = []
        if self.rate_limit < 1:
            errors.append("rate_limit must be >= 1")
        if self.db.pool_size < 1:
            errors.append("pool_size must be >= 1")
        return errors

    def get_feature_flags(self) -> dict[str, bool]:
        return {
            "audit": self.enable_audit,
            "cache": self.enable_cache,
            "rate_limit": ENABLE_USER_RATE_LIMIT,
        }


# ── Singleton default config ───────────────────────────────────────────────────
DEFAULT_CONFIG = AppConfig()
