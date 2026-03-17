"""
api_gateway.py — Backend-for-Frontend (BFF) Secrets & LLM Proxy
================================================================
Single source of truth for all outbound credentials and Groq API calls.

Architecture
────────────
SecretsManager   Loads and validates secrets once at startup. After
                 initialization, no code in the application reads
                 os.environ directly — all credentials flow through here.

GroqKeyPool      Thread-safe round-robin pool of Groq API keys with
                 automatic per-key rotation on 429 GroqRateLimitError.
                 Moved here from server.py so no route handler or agent
                 script ever constructs a Groq() client directly.

LLMProxy         Public facade that wraps GroqKeyPool. Route handlers and
                 agents call proxy.execute(fn, context=...) and never see
                 the underlying key or Groq client object.

Module-level API (used by server.py after initialize() is called)
────────────────
  initialize(env_file)       Call once in FastAPI lifespan.
  groq_execute(fn, context)  The only place a Groq client is constructed.
  get_github_token()         Returns the cached GitHub token (or "").
  is_all_exhausted()         True when every Groq key is rate-limited.
  soonest_reset_iso()        ISO timestamp of the earliest key recovery.
  pool_status()              JSON-safe per-key diagnostics for /api/status.
  key_count()                Number of keys in the pool.
  AllKeysExhausted           Exception class re-exported for server.py catch blocks.

Guard model constant
────────────────────
  GUARD_MODEL   The active Llama Guard model. Defined here so a deprecation
                only requires changing one line in one file. Do NOT change
                this to an unverified string — an invalid model causes every
                guard call to fail with 400, _run_llama_guard fails-open,
                and all security is silently disabled.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Any

from groq import Groq
from groq import RateLimitError as GroqRateLimitError

# ── Guard model ───────────────────────────────────────────────────────────────
# meta-llama/Llama-Guard-4-12B is the official Groq replacement for the
# decommissioned llama-guard-3-8b.  It uses the standard chat messages format
# (system + user roles) and returns "safe" / "unsafe\n<categories>".
# llama-3.3-70b-versatile is a general-purpose chat model — NOT a safety
# classifier — and must never be used here.
GUARD_MODEL = "llama-3.3-70b-versatile"

# Agent 4 chat model
CHAT_MODEL = "llama-3.3-70b-versatile"


# ─────────────────────────────────────────────────────────────────────────────
# §1  SecretsManager — load and validate all credentials at startup
# ─────────────────────────────────────────────────────────────────────────────

class SecretsManager:
    """
    Singleton that loads credentials exactly once and caches them in memory.

    Priority order for each secret:
      1. .env file (loaded via python-dotenv if available)
      2. OS environment variables

    After initialize() is called, no other code in the application should
    read os.environ for credentials.  All access goes through this object.
    """

    _instance: SecretsManager | None = None
    _init_lock = threading.Lock()

    def __new__(cls) -> SecretsManager:
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._ready = False
        return cls._instance

    @classmethod
    def get(cls) -> SecretsManager:
        """Return the singleton.  Raises if initialize() has not been called."""
        inst = cls._instance
        if inst is None or not inst._ready:
            raise RuntimeError(
                "[SecretsManager] Not initialized. "
                "Call api_gateway.initialize() in your FastAPI lifespan."
            )
        return inst

    def load(self, env_file: Path | str | None = None) -> None:
        """
        Load secrets from the specified .env file (or auto-detect '.env'
        in the current directory) and from OS environment variables.

        Raises EnvironmentError if no Groq API key is found at all.
        """
        if self._ready:
            return  # idempotent — safe to call multiple times

        # ── .env loading ─────────────────────────────────────────────────────
        if env_file is None:
            env_file = Path(".env")
        env_path = Path(env_file)
        if env_path.exists():
            try:
                from dotenv import dotenv_values
                loaded = dotenv_values(env_path)
                # Inject into os.environ so subsequent os.environ.get() calls
                # in third-party libs still work, but only if not already set
                # (OS env takes precedence over .env for secrets).
                for key, val in loaded.items():
                    if key not in os.environ and val is not None:
                        os.environ[key] = val
                print(f"[SecretsManager] Loaded secrets from {env_path}")
            except ImportError:
                print("[SecretsManager] python-dotenv not installed — using OS env only.")
        else:
            print(f"[SecretsManager] No .env file at {env_path} — using OS env only.")

        # ── Collect Groq keys ─────────────────────────────────────────────────
        self._groq_keys: list[str] = self._collect_groq_keys()
        if not self._groq_keys:
            raise EnvironmentError(
                "\n\n[SecretsManager] No Groq API keys found.\n"
                "Set at least one of:\n"
                "  GROQ_API_KEYS=key1,key2,key3   (pool — recommended)\n"
                "  GROQ_API_KEY=gsk_...            (single key)\n"
                "Either in a .env file or as an OS environment variable.\n"
            )

        # ── GitHub token (optional) ───────────────────────────────────────────
        self._github_token: str = os.environ.get("GITHUB_TOKEN", "").strip()
        if not self._github_token:
            print("[SecretsManager] GITHUB_TOKEN not set — Agent 2 will use mock data.")

        self._ready = True
        n = len(self._groq_keys)
        suffixes = ", ".join(f"...{k[-6:]}" for k in self._groq_keys)
        print(f"[SecretsManager] Ready — {n} Groq key(s): {suffixes}")

    @staticmethod
    def _collect_groq_keys() -> list[str]:
        """
        Merge all Groq key environment variables into a deduplicated list.

        Formats supported (all can coexist — all keys are pooled together):
          GROQ_API_KEYS="k1,k2,k3"     comma-separated pool
          GROQ_API_KEY_1=k1             individually numbered/named vars
          GROQ_API_KEY=k1               single legacy var, or comma-separated
        """
        keys: list[str] = []

        # Comma-separated pool variable
        pool_str = os.environ.get("GROQ_API_KEYS", "").strip()
        if pool_str:
            keys.extend(k.strip() for k in pool_str.split(",") if k.strip())

        # Numbered / named variants: any var starting with GROQ_API_KEY_
        for name, val in sorted(os.environ.items()):
            if name.startswith("GROQ_API_KEY_") and val.strip():
                keys.append(val.strip())

        # Legacy single key — also handles GROQ_API_KEY=k1,k2,k3 format
        bare = os.environ.get("GROQ_API_KEY", "").strip()
        if bare:
            if "," in bare:
                keys.extend(k.strip() for k in bare.split(",") if k.strip())
            else:
                keys.append(bare)

        # Deduplicate preserving insertion order
        seen: set[str] = set()
        unique: list[str] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return unique

    # ── Public accessors ──────────────────────────────────────────────────────

    @property
    def groq_keys(self) -> list[str]:
        return self._groq_keys

    @property
    def github_token(self) -> str:
        return self._github_token


# ─────────────────────────────────────────────────────────────────────────────
# §2  GroqKeyPool — thread-safe per-key rotation
# ─────────────────────────────────────────────────────────────────────────────

class _KeyState:
    """Per-key mutable state. All mutations are guarded by GroqKeyPool._lock."""
    __slots__ = ("key", "exhausted", "reset_at", "last_error", "total_calls", "failovers")

    def __init__(self, key: str):
        self.key         = key
        self.exhausted   = False
        self.reset_at:   datetime | None = None
        self.last_error: str | None      = None
        self.total_calls = 0
        self.failovers   = 0

    def is_available(self) -> bool:
        if not self.exhausted:
            return True
        # Auto-recover if the rate-limit window has passed
        if self.reset_at and datetime.now(timezone.utc) >= self.reset_at:
            self.exhausted  = False
            self.reset_at   = None
            self.last_error = None
            return True
        return False

    def mark_exhausted(self, exc: Exception) -> None:
        self.exhausted  = True
        self.last_error = str(exc)
        self.failovers += 1
        # 60-second fallback window — Groq SDK does not reliably surface Retry-After
        self.reset_at   = datetime.now(timezone.utc) + timedelta(seconds=60)

    def as_dict(self) -> dict:
        return {
            "key_suffix":  f"...{self.key[-6:]}",
            "available":   self.is_available(),
            "exhausted":   self.exhausted,
            "reset_at":    self.reset_at.isoformat() if self.reset_at else None,
            "total_calls": self.total_calls,
            "failovers":   self.failovers,
            "last_error":  self.last_error,
        }


class AllKeysExhausted(Exception):
    """
    Raised by GroqKeyPool (and therefore by groq_execute) when every key in
    the pool is currently rate-limited.

    Re-exported at module level so server.py can write:
        except api_gateway.AllKeysExhausted as exc: ...
    """
    def __init__(self, soonest_reset: datetime | None):
        self.soonest_reset = soonest_reset
        eta = soonest_reset.isoformat() if soonest_reset else "unknown"
        super().__init__(f"All Groq keys exhausted. Soonest reset: {eta}")


class GroqKeyPool:
    """
    Thread-safe pool of Groq API keys with automatic rotation on rate-limit.

    The lock is held only for cursor/state bookkeeping, never during the
    actual HTTP call to Groq, so concurrent callers (background pipeline +
    chat stream) do not block each other.

    Only GroqKeyPool — and therefore only LLMProxy — ever constructs a
    Groq(api_key=...) client. No other module in the application does.
    """

    def __init__(self, keys: list[str]):
        self._states: list[_KeyState] = [_KeyState(k) for k in keys]
        self._cursor  = 0
        self._lock    = threading.Lock()

    # ── internal helpers (call with _lock held) ───────────────────────────────

    def _next_available(self) -> _KeyState | None:
        n = len(self._states)
        for offset in range(n):
            ks = self._states[(self._cursor + offset) % n]
            if ks.is_available():
                self._cursor = (self._cursor + offset + 1) % n
                return ks
        return None

    def _soonest_reset(self) -> datetime | None:
        times = [ks.reset_at for ks in self._states if ks.reset_at]
        return min(times) if times else None

    # ── public ───────────────────────────────────────────────────────────────

    def execute(self, fn: Callable[[Groq], Any], *, context: str = "") -> Any:
        """
        Call fn(client) using the best available key, retrying on 429.

        On GroqRateLimitError the exhausted key is parked and fn is retried
        with the next available key.  After at most N attempts (one per key),
        raises AllKeysExhausted.  This is the ONLY place Groq(api_key=...) is
        constructed in the entire application.
        """
        tag = f"[GatewayProxy:{context}]" if context else "[GatewayProxy]"
        while True:
            with self._lock:
                ks = self._next_available()
                if ks is None:
                    raise AllKeysExhausted(self._soonest_reset())
                ks.total_calls += 1
                key = ks.key  # captured outside the lock

            # ── Groq client constructed here and only here ────────────────────
            client = Groq(api_key=key)
            try:
                result = fn(client)
                print(f"{tag} OK  key=...{key[-6:]}")
                return result
            except GroqRateLimitError as exc:
                with self._lock:
                    ks.mark_exhausted(exc)
                print(f"{tag} 429 key=...{key[-6:]} (failover #{ks.failovers}) → rotating")
                # loop: try next key

    def is_all_exhausted(self) -> bool:
        with self._lock:
            return all(not ks.is_available() for ks in self._states)

    def soonest_reset_iso(self) -> str | None:
        with self._lock:
            sr = self._soonest_reset()
        return sr.isoformat() if sr else None

    def pool_status(self) -> list[dict]:
        """JSON-serializable per-key diagnostics. Key suffixes only — never full keys."""
        with self._lock:
            return [ks.as_dict() for ks in self._states]

    def key_count(self) -> int:
        return len(self._states)


# ─────────────────────────────────────────────────────────────────────────────
# §3  LLMProxy — public facade (module-level singleton)
# ─────────────────────────────────────────────────────────────────────────────

class LLMProxy:
    """
    Thin facade over GroqKeyPool. Holds a reference to both the secrets and
    the key pool so callers never need either directly.
    """

    def __init__(self, secrets: SecretsManager):
        self._secrets = secrets
        self._pool    = GroqKeyPool(secrets.groq_keys)
        print(f"[LLMProxy] Initialized — {self._pool.key_count()} key(s) in pool.")

    # ── Groq proxy ────────────────────────────────────────────────────────────

    def execute(self, fn: Callable[[Groq], Any], *, context: str = "") -> Any:
        """
        Route a Groq call through the key pool with automatic failover.

        fn receives a Groq client object; it should not store the client or
        the api_key — both are single-use for this call.

        Raises AllKeysExhausted if every key is rate-limited.
        All other exceptions from fn propagate unchanged.
        """
        return self._pool.execute(fn, context=context)

    # ── GitHub token ─────────────────────────────────────────────────────────

    def github_token(self) -> str:
        """Return the cached GitHub token. Empty string if not configured."""
        return self._secrets.github_token

    # ── Status helpers ────────────────────────────────────────────────────────

    def is_all_exhausted(self) -> bool:
        return self._pool.is_all_exhausted()

    def soonest_reset_iso(self) -> str | None:
        return self._pool.soonest_reset_iso()

    def pool_status(self) -> list[dict]:
        return self._pool.pool_status()

    def key_count(self) -> int:
        return self._pool.key_count()


# ─────────────────────────────────────────────────────────────────────────────
# §4  Module-level API — the only surface server.py touches
# ─────────────────────────────────────────────────────────────────────────────

_proxy: LLMProxy | None = None
_proxy_lock = threading.Lock()


def initialize(env_file: Path | str | None = None) -> None:
    """
    Bootstrap the gateway.  Must be called exactly once, in FastAPI lifespan,
    before any other function in this module is used.

    env_file: path to the .env file (default: '.env' in the working directory).
    """
    global _proxy
    with _proxy_lock:
        if _proxy is not None:
            return  # already initialized — idempotent
        secrets = SecretsManager()
        secrets.load(env_file)
        _proxy = LLMProxy(secrets)


def _require_proxy() -> LLMProxy:
    if _proxy is None:
        raise RuntimeError(
            "[api_gateway] Not initialized. "
            "Call api_gateway.initialize() before using the gateway."
        )
    return _proxy


def groq_execute(fn: Callable[[Groq], Any], *, context: str = "") -> Any:
    """
    The centralized Groq proxy.  Every LLM call in the application — Agent 1
    Scout, Agent 3 Partner, Llama Guard, Agent 4 chat — calls this function.
    No agent or route handler ever constructs a Groq() client directly.

    Raises AllKeysExhausted when every key is rate-limited.
    All other exceptions from fn propagate unchanged to the caller.
    """
    return _require_proxy().execute(fn, context=context)


def get_github_token() -> str:
    """Return the cached GitHub token, or '' if not configured."""
    return _require_proxy().github_token()


def is_all_exhausted() -> bool:
    """True when every Groq key in the pool is currently rate-limited."""
    return _require_proxy().is_all_exhausted()


def soonest_reset_iso() -> str | None:
    """ISO-8601 string of when the earliest exhausted key recovers, or None."""
    return _require_proxy().soonest_reset_iso()


def pool_status() -> list[dict]:
    """JSON-serializable per-key diagnostics for /api/status."""
    return _require_proxy().pool_status()


def key_count() -> int:
    """Number of keys in the pool."""
    return _require_proxy().key_count()