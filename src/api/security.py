"""Password hashing (scrypt, standard library) and signed login tokens (PyJWT).

Honest scope: this is solid for a project/demo. For real users add HTTPS, rate limiting at the proxy, password reset, and an audit log.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

import jwt

_N, _R, _P = 2**14, 8, 1


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        calc = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, bytes.fromhex(digest))
    except (ValueError, TypeError):
        return False


def secret_key() -> str:
    key = os.environ.get("API_SECRET_KEY")
    if key and len(key) >= 32:
        return key
    # No (or weak) key configured: use a random per-process key. Logins stop working after a restart - set API_SECRET_KEY in .env.
    global _EPHEMERAL
    try:
        return _EPHEMERAL
    except NameError:
        _EPHEMERAL = secrets.token_hex(32)
        return _EPHEMERAL


def create_token(username: str, role: str, key: str, hours: float = 8.0) -> str:
    now = int(time.time())
    return jwt.encode({"sub": username, "role": role, "iat": now, "exp": now + int(hours * 3600)}, key, algorithm="HS256")


def decode_token(token: str, key: str) -> dict | None:
    try:
        return jwt.decode(token, key, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
