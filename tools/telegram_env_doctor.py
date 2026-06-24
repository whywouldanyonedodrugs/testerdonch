#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from telegram_notify import (
    CHAT_ENV_ALIASES,
    TOKEN_ENV_ALIASES,
    TelegramNotifier,
)


def _mask_token(v: str) -> str:
    if not v:
        return ""
    if len(v) <= 10:
        return "***"
    return v[:6] + "..." + v[-4:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose Telegram env wiring for Donch tools.")
    p.add_argument("--send-test", action="store_true", help="Send a test message if notifier is enabled.")
    p.add_argument("--label", default="tg-doctor", help="Run label for test message.")
    return p.parse_args()


def main() -> int:
    a = parse_args()

    print("Token env aliases:")
    token_found = False
    for n in TOKEN_ENV_ALIASES:
        v = (os.environ.get(n) or "").strip()
        if v:
            token_found = True
            print(f"  {n}=set ({_mask_token(v)})")
        else:
            print(f"  {n}=<empty>")

    print("\nChat env aliases:")
    chat_found = False
    for n in CHAT_ENV_ALIASES:
        v = (os.environ.get(n) or "").strip()
        if v:
            chat_found = True
            print(f"  {n}=set ({v})")
        else:
            print(f"  {n}=<empty>")

    class _Args:
        tg_bot_token = ""
        tg_chat_id = ""
        tg_auto_chat = True

    notifier = TelegramNotifier.from_args(_Args(), run_label=str(a.label).strip() or "tg-doctor")
    print(f"\nResolved notifier status: {notifier.status_line()}")
    if notifier.enabled:
        print(f"Resolved token: {_mask_token(notifier.token)}")
        print(f"Resolved chat:  {notifier.chat_id}")

    if a.send_test:
        msg = (
            f"doctor_test=ok\n"
            f"utc={datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}\n"
            f"source=tools/telegram_env_doctor.py"
        )
        ok = notifier.send("TEST", body=msg)
        print(f"\nSend test: {'ok' if ok else 'failed'}")
        return 0 if ok else 2

    if token_found and chat_found:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
