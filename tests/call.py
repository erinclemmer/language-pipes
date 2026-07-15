#!/usr/bin/env python3
"""Call an OpenAI-compatible server's Responses endpoint and stream to stdout.

Usage:
    python call.py "your prompt here"
    python call.py --model my-model "your prompt"
    echo "your prompt" | python call.py

Environment variables:
    OPENAI_BASE_URL   Base URL of the server (default: http://localhost:8000/v1)
    OPENAI_API_KEY    API key sent as a Bearer token (default: "").
    OPENAI_MODEL      Default model id if --model is not given.
"""

import argparse
import json
import os
import sys

import requests


def stream_responses(base_url: str, api_key: str, model: str, prompt: str) -> None:
    url = base_url.rstrip("/") + "/responses"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "input": prompt,
        "stream": True,
    }

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=None) as resp:
        if resp.status_code != 200:
            sys.stderr.write(f"error {resp.status_code}: {resp.text}\n")
            sys.exit(1)

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                sys.stdout.write(event.get("delta", ""))
                sys.stdout.flush()
            elif event_type == "response.error" or event.get("error"):
                sys.stderr.write(f"\nstream error: {event.get('error') or event}\n")
                sys.exit(1)

    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Prompt text (or piped via stdin).")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        help="Server base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        help="Model id (default: %(default)s).",
    )
    args = parser.parse_args()

    prompt = args.prompt
    if prompt is None:
        prompt = sys.stdin.read().strip()
    if not prompt:
        parser.error("no prompt given (pass an argument or pipe via stdin)")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    stream_responses(args.base_url, api_key, args.model, prompt)


if __name__ == "__main__":
    main()
