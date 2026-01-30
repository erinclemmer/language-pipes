import yappi
import os
import sys
import time
import pathlib
import requests
import unittest
from typing import List, Optional

import openai

cd = pathlib.Path().resolve()

sys.path.append(os.path.join(cd, 'src'))

from language_pipes.cli import main
from language_pipes.util.chat import ChatMessage, ChatRole

MODEL = "Qwen/Qwen3-1.7B"

def start_node(node_id: str, max_memory: float, peer_port: int, oai_port: Optional[int] = None, bootstrap_port: Optional[int] = None):
    args = ["serve", 
        "--node-id", node_id, 
        "--hosted-models", f"id={MODEL},device=cpu,memory={max_memory},load_ends=true", 
        "--peer-port", str(peer_port),
        "--app-dir", "./",
        "--model-validation"
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
            max_completion_tokens=50,
            messages=[m.to_json() for m in messages]
        )
        return response
    except Exception as e:
        print(e)
        if retries < 5:
            time.sleep(5)
            return oai_complete(port, messages, retries + 1)


def oai_stream(port: int, messages: List[ChatMessage], retries: int = 0):
    try:
        client = openai.OpenAI(
            api_key="",
            base_url=f"http://127.0.0.1:{port}/v1",
        )
        stream = client.chat.completions.create(
            model=MODEL,
            stream=True,
            max_completion_tokens=100,
            messages=[m.to_json() for m in messages]
        )
        s = ""
        for chunk in stream:
            c = chunk.choices[0].delta.content
            print(c)
            s += c
        print(s)
        
    except Exception as e:
        print(e)
        if retries < 5:
            time.sleep(5)
            return oai_complete(port, messages, retries + 1)

class OpenAITests(unittest.TestCase):
    def test_cli(self):
        main([])

    def test_single_node(self):
        start_node("node-1", 5, 5000, 8000)
        res = oai_complete(8000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])
        print("\"" + res.choices[0].message.content + "\"")
        self.assertTrue(len(res.choices) > 0)

    def test_models_endpoint(self):
        start_node("node-1", 5, 5000, 8000)
        client = openai.OpenAI(
            api_key="",
            base_url="http://127.0.0.1:8000/v1",
        )
        models = client.models.list()
        self.assertEqual(models.object, "list")
        self.assertTrue(len(models.data) > 0)
        
        # Verify model object structure matches OpenAI format
        model = models.data[0]
        self.assertEqual(model.object, "model")
        self.assertIsNotNone(model.id)
        self.assertIsNotNone(model.created)
        
        # Also test raw HTTP request
        res = requests.get("http://localhost:8000/v1/models")
        self.assertEqual(200, res.status_code)
        data = res.json()
        self.assertEqual(data["object"], "list")
        self.assertIn("data", data)

    def test_400_codes(self):
        start_node("node-1", 5, 5000, 8000)
        messages = [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ]
        res = requests.post("http://localhost:8000/v1/chat/completions", json={
            "messages": [m.to_json() for m in messages]
        })

        self.assertEqual(400, res.status_code)

        res = requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": MODEL
        })

        self.assertEqual(400, res.status_code)

        res = requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": MODEL,
            "messages": []
        })

        self.assertEqual(400, res.status_code)

    def test_double_node(self):
        start_node("node-1", 1.5, 5000, 8000)
        time.sleep(5)
        start_node("node-2", 3, 5001, None, 5000)
        time.sleep(5)
        
        res = oai_complete(8000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])
        
        print("\"" + res.choices[0].message.content + "\"")
        self.assertTrue(len(res.choices) > 0)

    def test_double_long(self):
        start_node("node-1", 1.5, 5000, 8000)
        time.sleep(5)
        start_node("node-2", 3, 5001, None, 5000)
        time.sleep(5)
        with open('mcbeth.txt', 'r', encoding='utf-8') as f:
            mcbeth = f.read()
        res = oai_complete(8000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, f"What play is the following text the opening to?\n{mcbeth}")
        ])
        print("\"" + res.choices[0].message.content + "\"")
        self.assertTrue(len(res.choices) > 0)

    def test_stream(self):
        start_node("node-1", 2, 5000, 8000)
        time.sleep(5)
        start_node("node-2", 3, 5001, None, 5000)
        time.sleep(5)
        oai_stream(8000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])

    def test_triple_node(self):
        start_node("node-1", 1, 5000, 8000)
        time.sleep(10)
        start_node("node-2", 1, 5001, None, 5000)
        time.sleep(10)
        start_node("node-3", 3, 5002, None, 5000)
        time.sleep(10)
        res = oai_complete(8000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])
        print("\"" + res.choices[0].message.content + "\"")
        self.assertTrue(len(res.choices) > 0)


    def test_reconnect(self):
        start_node("node-1", 1, 5000, 8000)
        time.sleep(10)
        node2 = start_node("node-2", 1, 5001, None, 5000)
        time.sleep(10)
        start_node("node-3", 3, 5002, None, 5000)
        time.sleep(10)
        node2.stop()
        time.sleep(10)
        start_node("node-4", 1, 5004, None, 5000)
        time.sleep(5)

        res = oai_complete(8000, [
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant"),
            ChatMessage(ChatRole.USER, "Hello, how are you?")
        ])
        print("\"" + res.choices[0].message.content + "\"")
        self.assertTrue(len(res.choices) > 0)


if __name__ == '__main__':
    unittest.main()
