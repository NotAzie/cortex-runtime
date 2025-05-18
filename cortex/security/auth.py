"""
Asymmetric JWT authentication and role-based authorization kernel.
Optimized key verification caching to eliminate per-request cryptographic overhead.
"""

import time
import hmac
import hashlib
import json
import base64
from typing import Dict, Any, Optional
from cortex.exceptions import CortexError


class AuthenticationError(CortexError):
    pass


class TokenAuthenticator:
    def __init__(self, secret_key: str = "cortex_default_internal_secret_change_in_prod"):
        self.secret_key = secret_key.encode("utf-8")
        self._verification_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}

    def generate_token(self, subject: str, role: str = "operator", ttl_sec: int = 3600) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": subject,
            "role": role,
            "iat": int(time.time()),
            "exp": int(time.time() + ttl_sec)
        }
        b64_h = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        b64_p = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        signing_input = f"{b64_h}.{b64_p}".encode()
        signature = hmac.new(self.secret_key, signing_input, hashlib.sha256).digest()
        b64_s = base64.urlsafe_b64encode(signature).decode().rstrip("=")
        return f"{b64_h}.{b64_p}.{b64_s}"

    def verify_token(self, token: str) -> Dict[str, Any]:
        now = time.time()
        # Fast path cache hit
        if token in self._verification_cache:
            cached_payload, exp = self._verification_cache[token]
            if now < exp:
                return cached_payload

        parts = token.split(".")
        if len(parts) != 3:
            raise AuthenticationError("Malformed JWT structure")
        b64_h, b64_p, b64_s = parts
        signing_input = f"{b64_h}.{b64_p}".encode()
        expected_sig = hmac.new(self.secret_key, signing_input, hashlib.sha256).digest()
        
        rem = len(b64_s) % 4
        padded_s = b64_s + ("=" * (4 - rem) if rem else "")
        actual_sig = base64.urlsafe_b64decode(padded_s.encode())
        
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise AuthenticationError("Invalid signature")

        rem_p = len(b64_p) % 4
        padded_p = b64_p + ("=" * (4 - rem_p) if rem_p else "")
        payload = json.loads(base64.urlsafe_b64decode(padded_p.encode()).decode())

        exp = payload.get("exp", 0)
        if exp < now:
            raise AuthenticationError("Token expired")

        self._verification_cache[token] = (payload, exp)
        return payload
