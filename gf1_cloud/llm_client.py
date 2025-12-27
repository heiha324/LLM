from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List


def _post_json(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def chat_completion(
    url: str,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 60,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return _post_json(url, payload, timeout)


def get_message_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices", [])
    if not choices:
        raise ValueError("empty response choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return str(content).strip()


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("empty text")
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start == -1:
        raise ValueError("no json object in text")
    depth = 0
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("unterminated json object")
    return json.loads(text[start:end])


def file_url(path: str) -> str:
    if path.startswith("file://"):
        return path
    return "file://" + path
