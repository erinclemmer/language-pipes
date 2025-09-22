import requests
import json

url = "http://127.0.0.1:6000/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

payload = {
    "model": "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "max_completion_tokens": 10,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a haiku about distributed systems."}
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(payload))

print("Status Code:", response.status_code)
print("Response JSON:", response.json())