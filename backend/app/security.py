"""Password hashing, signed tokens, filename sanitising and a small in-memory rate limiter."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from collections import defaultdict, deque

_SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_hex, digest_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


_DUMMY_HASH: str | None = None


def verify_password_or_dummy(password: str, stored: str | None) -> bool:
    """Spend the same time whether or not the account exists, so login timing does not reveal registered emails."""
    global _DUMMY_HASH
    if stored is None:
        if _DUMMY_HASH is None:
            _DUMMY_HASH = hash_password("dummy-password-for-timing")
        verify_password(password, _DUMMY_HASH)
        return False
    return verify_password(password, stored)


def create_token(user_id: str, secret: str, ttl_minutes: int, epoch: int = 0) -> str:
    body = _b64(json.dumps({"sub": user_id, "exp": int(time.time()) + ttl_minutes * 60, "ep": epoch}).encode())
    sig = _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_token(token: str, secret: str) -> str | None:
    """Return the user id if the token is authentic and unexpired, else None."""
    payload = verify_token_payload(token, secret)
    return str(payload["sub"]) if payload else None


def verify_token_payload(token: str, secret: str) -> dict | None:
    """Return {'sub', 'exp', 'ep'} if the token is authentic and unexpired, else None."""
    try:
        body, sig = token.split(".")
        expected = _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_unb64(body))
        if payload["exp"] < time.time():
            return None
        return {"sub": str(payload["sub"]), "exp": payload["exp"], "ep": int(payload.get("ep", 0))}
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def safe_display_name(filename: str | None) -> str:
    """A display-only name. Files are never stored under this name."""
    name = (filename or "document.pdf").replace("\\", "/").split("/")[-1]
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip() or "document.pdf"
    return name[:120]


class RateLimiter:
    """Sliding-window limiter kept in process memory.

    Limitation: counters are per-process and reset on restart. Use a reverse proxy or a shared
    store (e.g. Redis) when running several backend replicas.
    """

    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True
