import os
import sys
import time
import pathlib
import requests
import unittest
from typing import List

cd = pathlib.Path().resolve()

sys.path.append(os.path.join(cd, 'src'))

from language_pipes.cli import main
from language_pipes.util.chat import ChatMessage, ChatRole

MODEL = "Qwen/Qwen3-1.7B"
# MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
# MODEL = "meta-llama/Llama-3.2-1B-Instruct"

def start_node(node_id: str, max_memory: float, peer_port: int, job_port: int, oai_port: int = None, bootstrap_port: int = None):
    args = ["run", 
        "--node-id", node_id, 
        "--hosted-models", f"{MODEL}::cpu::{max_memory}", 
        "--peer-port", str(peer_port),
        "--job-port", str(job_port),
        "--model-validation", 
        "--https"
    ]
    if oai_port is not None:
        args.extend(["--oai-port", str(oai_port)])
    
    if bootstrap_port is not None:
        args.extend(["--bootstrap-address", "localhost", "--bootstrap-port", str(bootstrap_port)])

    return main(args)

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
        start_node("node-1", 5, 5000, 5050, 6000)
        res = oai_complete(6000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])
        print("\"" + res["choices"][0]["message"]["content"] + "\"")
        self.assertTrue(len(res["choices"]) > 0)

    def test_400_codes(self):
        start_node("node-1", 5, 5000, 5050, 6000)
        messages = [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ]
        res = requests.post("http://localhost:6000/v1/chat/completions", json={
            "messages": [m.to_json() for m in messages]
        })

        self.assertEqual(400, res.status_code)

        res = requests.post("http://localhost:6000/v1/chat/completions", json={
            "model": MODEL
        })

        self.assertEqual(400, res.status_code)

        res = requests.post("http://localhost:6000/v1/chat/completions", json={
            "model": MODEL,
            "messages": []
        })

        self.assertEqual(400, res.status_code)

    def test_double_node(self):
        start_node("node-1", 2, 5000, 5050, 6000)
        time.sleep(10)
        start_node("node-2", 3, 5001, 5051, None, 5000)
        time.sleep(10)
        res = oai_complete(6000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])
        print("\"" + res["choices"][0]["message"]["content"] + "\"")
        self.assertTrue(len(res["choices"]) > 0)

    def test_triple_node(self):
        start_node("node-1", 1, 5000, 5050, 6000)
        time.sleep(10)
        start_node("node-2", 1, 5001, 5051, None, 5000)
        time.sleep(10)
        start_node("node-3", 3, 5002, 5052, None, 5000)
        time.sleep(10)
        res = oai_complete(6000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])
        print("\"" + res["choices"][0]["message"]["content"] + "\"")
        self.assertTrue(len(res["choices"]) > 0)


    def test_reconnect(self):
        start_node("node-1", 1, 5000, 5050, 6000)
        time.sleep(10)
        node2 = start_node("node-2", 1, 5001, 5051, None, 5000)
        time.sleep(10)
        start_node("node-3", 3, 5002, 5052, None, 5000)
        time.sleep(10)
        node2.stop()
        time.sleep(10)
        start_node("node-4", 1, 5004, 5054, None, 5000)
        time.sleep(5)

        res = oai_complete(6000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])
        print("\"" + res["choices"][0]["message"]["content"] + "\"")
        self.assertTrue(len(res["choices"]) > 0)


if __name__ == '__main__':
    unittest.main()