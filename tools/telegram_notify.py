#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


TOKEN_ENV = "DONCH_TG_BOT_TOKEN"
CHAT_ENV = "DONCH_TG_CHAT_ID"
MAX_MSG_LEN = 3500

TOKEN_ENV_ALIASES = (
    "DONCH_TG_BOT_TOKEN",
    "DONCH_TG_TOKEN",
    "DONCH_TELEGRAM_BOT_TOKEN",
    "TG_BOT_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "BOT_TOKEN",
)
CHAT_ENV_ALIASES = (
    "DONCH_TG_CHAT_ID",
    "DONCH_CHAT_ID",
    "DONCH_TELEGRAM_CHAT_ID",
    "TG_CHAT_ID",
    "TELEGRAM_CHAT_ID",
    "CHAT_ID",
)
RECOGNIZED_ENV_KEYS = set(TOKEN_ENV_ALIASES) | set(CHAT_ENV_ALIASES) | {"DONCH_TG_FAIL_LOG"}
ENV_FILE_CANDIDATES = (
    Path.cwd() / ".telegram.env",
    Path(__file__).resolve().parents[1] / ".telegram.env",
    Path("/etc/default/donch-telegram"),
    Path("/etc/default/donch-autopar-backtester"),
)


def _safe(value: Optional[str]) -> str:
    return (value or "").strip()


def _first_env(names: tuple[str, ...]) -> str:
    for n in names:
        v = _safe(os.environ.get(n))
        if v:
            return v
    return ""


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    try:
        parts = shlex.split(value, comments=False, posix=True)
        if len(parts) == 1:
            return parts[0]
    except ValueError:
        pass
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def load_telegram_env_files(extra_paths: Optional[list[str | Path]] = None) -> list[dict[str, Any]]:
    """Load Telegram env vars from known local files without exposing values."""
    paths: list[Path] = []
    seen: set[Path] = set()
    for p in list(ENV_FILE_CANDIDATES) + [Path(x) for x in (extra_paths or [])]:
        try:
            rp = p.expanduser().resolve()
        except Exception:
            rp = p
        if rp in seen:
            continue
        seen.add(rp)
        paths.append(rp)

    reports: list[dict[str, Any]] = []
    for path in paths:
        rec: dict[str, Any] = {"path": str(path), "exists": path.exists(), "loaded_keys": []}
        if not path.exists() or not path.is_file():
            reports.append(rec)
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if s.startswith("export "):
                    s = s[len("export ") :].strip()
                if "=" not in s:
                    continue
                key, raw_val = s.split("=", 1)
                key = key.strip()
                if key not in RECOGNIZED_ENV_KEYS or _safe(os.environ.get(key)):
                    continue
                val = _parse_env_value(raw_val)
                if val:
                    os.environ[key] = val
                    rec["loaded_keys"].append(key)
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        reports.append(rec)
    return reports


def _clip(msg: str) -> str:
    txt = msg.strip()
    if len(txt) <= MAX_MSG_LEN:
        return txt
    return txt[: MAX_MSG_LEN - 24] + "\n... [truncated for Telegram]"


def _api_url(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def _post_json(url: str, payload: Dict[str, Any], timeout_sec: float = 12.0) -> Dict[str, Any]:
    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(raw)
    if not bool(obj.get("ok", False)):
        desc = str(obj.get("description", "unknown error"))
        raise RuntimeError(f"telegram api error: {desc}")
    return obj


def _get_json(url: str, timeout_sec: float = 12.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(raw)
    if not bool(obj.get("ok", False)):
        desc = str(obj.get("description", "unknown error"))
        raise RuntimeError(f"telegram api error: {desc}")
    return obj


def discover_chat_id(token: str, timeout_sec: float = 12.0) -> Optional[str]:
    tok = _safe(token)
    if not tok:
        return None
    try:
        url = _api_url(tok, "getUpdates") + "?timeout=1&limit=25"
        obj = _get_json(url, timeout_sec=timeout_sec)
        rows = list(obj.get("result", []))
        for upd in reversed(rows):
            for key in ("message", "edited_message", "channel_post", "edited_channel_post"):
                node = upd.get(key, {})
                chat = node.get("chat", {})
                cid = chat.get("id", None)
                if cid is not None:
                    return str(cid)
    except Exception:
        return None
    return None


def send_telegram_message(token: str, chat_id: str, text: str, timeout_sec: float = 12.0) -> None:
    tok = _safe(token)
    cid = _safe(chat_id)
    if not tok:
        raise ValueError("missing telegram bot token")
    if not cid:
        raise ValueError("missing telegram chat id")
    payload = {
        "chat_id": cid,
        "text": _clip(text),
        "disable_web_page_preview": "true",
    }
    _post_json(_api_url(tok, "sendMessage"), payload, timeout_sec=timeout_sec)


def _log_send_failure(run_label: str, title: str, err: Exception) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"[tg-notify] send_failed utc={ts} run_label={run_label} title={title} err={type(err).__name__}: {err}"
    try:
        print(line, file=sys.stderr, flush=True)
    except Exception:
        pass
    path = _safe(os.environ.get("DONCH_TG_FAIL_LOG"))
    if path:
        try:
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            with Path(path).expanduser().resolve().open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


@dataclass
class TelegramNotifier:
    token: str
    chat_id: str
    enabled: bool
    run_label: str
    auto_chat: bool = False

    @classmethod
    def from_args(
        cls,
        args: Any,
        *,
        run_label: str,
        token_attr: str = "tg_bot_token",
        chat_attr: str = "tg_chat_id",
        auto_chat_attr: str = "tg_auto_chat",
    ) -> "TelegramNotifier":
        load_telegram_env_files()
        tok = _safe(getattr(args, token_attr, "")) or _first_env(TOKEN_ENV_ALIASES)
        cid = _safe(getattr(args, chat_attr, "")) or _first_env(CHAT_ENV_ALIASES)
        auto_chat = bool(getattr(args, auto_chat_attr, False))
        if tok and (not cid) and auto_chat:
            cid = _safe(discover_chat_id(tok) or "")
        enabled = bool(tok and cid)
        return cls(
            token=tok,
            chat_id=cid,
            enabled=enabled,
            run_label=run_label.strip() or "run",
            auto_chat=auto_chat,
        )

    def status_line(self) -> str:
        if self.enabled:
            return "enabled"
        if self.token and not self.chat_id:
            return f"disabled: missing chat id (set {CHAT_ENV} or --tg-chat-id, or use --tg-auto-chat)"
        if self.chat_id and not self.token:
            return f"disabled: missing bot token (set {TOKEN_ENV} or --tg-bot-token)"
        return "disabled"

    def send(self, title: str, body: str = "") -> bool:
        if (not self.enabled) and self.token and (not self.chat_id) and self.auto_chat:
            self.chat_id = _safe(discover_chat_id(self.token) or "")
            self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            return False
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        host = socket.gethostname()
        cwd = str(Path.cwd())
        msg = f"[{self.run_label}] {title}\nUTC: {ts}\nHost: {host}\nCWD: {cwd}"
        if body.strip():
            msg += f"\n\n{body.strip()}"
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                send_telegram_message(self.token, self.chat_id, msg)
                return True
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(1.0 + attempt)
        if last_err is not None:
            _log_send_failure(self.run_label, title, last_err)
        return False
