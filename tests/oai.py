import os
import sys
import time
import requests
import threading
import unittest
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from language_pipes.cli import main
from language_pipes.util.chat import ChatMessage, ChatRole

MODEL = "Qwen/Qwen3-1.7B"

def start_node(node_id: str, max_memory: float, oai_port: int, peer_port: int, job_port: int):
    t = threading.Thread(target=main, args=(["run", 
        "--node-id", node_id, 
        "--hosted-models", f"{MODEL}:cpu:{max_memory}", 
        "--oai-port", str(oai_port),
        "--peer-port", str(peer_port),
        "--job-port", str(job_port)
    ], ))
    t.start()
    return t

def oai_complete(port: int, messages: List[ChatMessage], retries: int = 0):
    try:
        res = requests.post(f"http://localhost:{port}/v1/chat/completions", json={
            "model": MODEL,
            "max_completion_tokens": 10,
            "messages": [m.to_json() for m in messages]
        })
        if res.status_code != 200:
            raise Exception(f"Failed to complete: {res.text}")
        return res.json()
    except Exception as e:
        print(e)
        if retries < 5:
            time.sleep(5)
            return oai_complete(port, messages, retries + 1)


class OpenAITests(unittest.TestCase):
    def test_single_node(self):
        start_node("node-1", 5, 6000, 5000, 5050)
        res = oai_complete(6000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])
        print("\"" + res["choices"][0]["message"]["content"] + "\"")
        self.assertTrue(len(res["choices"]) > 0)

    def test_double_node(self):
        start_node("node-1", 2, 6000, 5000, 5050)
        time.sleep(10)
        start_node("node-2", 3, 6001, 5001, 5051)
        time.sleep(5)
        res = oai_complete(6000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])
        print("\"" + res["choices"][0]["message"]["content"] + "\"")
        self.assertTrue(len(res["choices"]) > 0)


if __name__ == '__main__':
    unittest.main()