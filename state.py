#!/usr/bin/env python3
"""Manage warpembot processed message state."""

import argparse
import json
import sys
from pathlib import Path

STATE_FILE = Path(__file__).parent / "state.json"


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed_message_ids": []}


def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")


def cmd_check(args):
    state = load_state()
    if args.message_id in state["processed_message_ids"]:
        print(f"YES — already processed")
        return 0
    else:
        print(f"NO — not yet processed")
        return 1


def cmd_mark(args):
    state = load_state()
    if args.message_id in state["processed_message_ids"]:
        print(f"Already processed, no change.")
        return 0
    state["processed_message_ids"].append(args.message_id)
    save_state(state)
    print(f"Marked as processed. Total: {len(state['processed_message_ids'])}")
    return 0


def cmd_list(args):
    state = load_state()
    for mid in state["processed_message_ids"]:
        print(mid)
    return 0


def cmd_count(args):
    state = load_state()
    print(len(state["processed_message_ids"]))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Manage warpembot state")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="Check if a message ID has been processed")
    p_check.add_argument("message_id")
    p_check.set_defaults(func=cmd_check)

    p_mark = sub.add_parser("mark", help="Mark a message ID as processed")
    p_mark.add_argument("message_id")
    p_mark.set_defaults(func=cmd_mark)

    p_list = sub.add_parser("list", help="List all processed message IDs")
    p_list.set_defaults(func=cmd_list)

    p_count = sub.add_parser("count", help="Count processed message IDs")
    p_count.set_defaults(func=cmd_count)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
