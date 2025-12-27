#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from gf1_cloud.llm_client import chat_completion, extract_json, get_message_text
from gf1_cloud.llm_prompts import router_prompt
from gf1_cloud.progress import tqdm
from gf1_cloud.utils import ensure_dir, load_json, save_json


def _iter_feature_files(path: Path):
    if path.is_file():
        yield path
    else:
        yield from path.rglob("*.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="features json file or dir")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", default="outputs/llm_router")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    features_path = Path(args.features)
    ensure_dir(args.out_dir)

    paths = list(_iter_feature_files(features_path))
    if args.limit:
        paths = paths[: args.limit]

    for path in tqdm(paths, desc="llm_router", total=len(paths), leave=True):
        payload = load_json(str(path))
        features = payload.get("features", payload)
        prompt = router_prompt(features)
        messages = [{"role": "user", "content": prompt}]
        response = chat_completion(args.api_url, args.model, messages, temperature=0.0)
        text = get_message_text(response)
        route = extract_json(text)

        payload["llm_route"] = route
        out_path = Path(args.out_dir) / path.name
        save_json(payload, str(out_path))


if __name__ == "__main__":
    main()
