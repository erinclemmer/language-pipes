import os
import time
from typing import List

import pytest
import requests

from language_pipes.cli import main
from language_pipes.util.chat import ChatMessage, ChatRole

openai = pytest.importorskip("openai")

MODEL = "Qwen/Qwen3-1.7B"

pytestmark = pytest.mark.integration


def _require_integration():
    if os.getenv("LP_RUN_INTEGRATION") != "1":
        pytest.skip("Set LP_RUN_INTEGRATION=1 to run integration tests.")


def start_node(
    node_id: str,
    max_memory: float,
    peer_port: int,
    job_port: int,
    oai_port: int | None = None,
    bootstrap_port: int | None = None,
):
    args = [
        "serve",
        "--node-id",
        node_id,
        "--hosted-models",
        f"id={MODEL},device=cpu,memory={max_memory},load_ends=true",
        "--peer-port",
        str(peer_port),
        "--job-port",
        str(job_port),
        "--app-dir",
        "./",
        "--model-validation",
    ]
    if oai_port is not None:
        args.extend(["--openai-port", str(oai_port)])

    if bootstrap_port is not None:
        args.extend(["--bootstrap-address", "localhost", "--bootstrap-port", str(bootstrap_port)])

    return main(args)


def oai_complete(port: int, messages: List[ChatMessage], retries: int = 0):
    try:
        client = openai.OpenAI(
            api_key="",
            base_url=f"http://127.0.0.1:{port}/v1",
        )
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.2,
            max_completion_tokens=100,
            messages=[m.to_json() for m in messages],
        )
        return response
    except Exception as exc:
        if retries < 5:
            time.sleep(5)
            return oai_complete(port, messages, retries + 1)
        raise exc


def test_400_codes():
    _require_integration()
    start_node("node-1", 5, 5000, 5050, 8000)
    messages = [
        ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
        ChatMessage(ChatRole.USER, "Hello, how are you?"),
    ]
    res = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"messages": [m.to_json() for m in messages]},
        timeout=10,
    )

    assert res.status_code == 400

    res = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"model": MODEL},
        timeout=10,
    )

    assert res.status_code == 400

    res = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={"model": MODEL, "messages": []},
        timeout=10,
    )

    assert res.status_code == 400


def test_single_node():
    _require_integration()
    start_node("node-1", 5, 5000, 5050, 8000)
    res = oai_complete(
        8000,
        [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?"),
        ],
    )
    assert len(res.choices) > 0
