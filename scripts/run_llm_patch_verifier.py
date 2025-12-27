#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from gf1_cloud.llm_client import chat_completion, extract_json, file_url, get_message_text
from gf1_cloud.llm_prompts import patch_verifier_prompt
from gf1_cloud.progress import tqdm
from gf1_cloud.utils import ensure_dir, save_json


def _collect_patches(patch_dir: Path) -> Dict[str, Dict[str, str]]:
    groups: Dict[str, Dict[str, str]] = {}
    for path in patch_dir.glob("*_rgb.png"):
        patch_id = path.name[:-8]
        nirrg = patch_dir / f"{patch_id}_nirrg.png"
        overlay = patch_dir / f"{patch_id}_overlay.png"
        if not nirrg.exists() or not overlay.exists():
            continue
        groups[patch_id] = {
            "rgb": str(path.resolve()),
            "nirrg": str(nirrg.resolve()),
            "overlay": str(overlay.resolve()),
        }
    return groups


def _scene_key(patch_id: str) -> str:
    base, _, suffix = patch_id.rpartition("_")
    return base if suffix.isdigit() else patch_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dir", required=True)
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", default="outputs/patch_verify")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    patch_dir = Path(args.patch_dir)
    groups = _collect_patches(patch_dir)
    ensure_dir(args.out_dir)

    prompt = patch_verifier_prompt()
    stats: Dict[str, List[Dict]] = defaultdict(list)

    items = list(groups.items())
    if args.limit:
        items = items[: args.limit]

    for patch_id, paths in tqdm(items, desc="patch_verify", total=len(items), leave=False):
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": file_url(paths["rgb"])}} ,
            {"type": "image_url", "image_url": {"url": file_url(paths["nirrg"])}} ,
            {"type": "image_url", "image_url": {"url": file_url(paths["overlay"])}} ,
        ]
        messages = [{"role": "user", "content": content}]
        response = chat_completion(args.api_url, args.model, messages, temperature=0.0)
        text = get_message_text(response)
        result = extract_json(text)

        result["patch_id"] = patch_id
        out_path = Path(args.out_dir) / f"{patch_id}.json"
        save_json(result, str(out_path))
        stats[_scene_key(patch_id)].append(result)

    for scene_id, items in stats.items():
        cloud_probs = [float(i.get("cloud_like_prob", 0.0)) for i in items]
        confuse_probs = [float(i.get("confusing_surface_prob", 0.0)) for i in items]
        summary = {
            "scene_id": scene_id,
            "patch_count": len(items),
            "patch_cloud_like_mean": sum(cloud_probs) / max(1, len(cloud_probs)),
            "patch_confusing_surface_mean": sum(confuse_probs) / max(1, len(confuse_probs)),
        }
        out_path = Path(args.out_dir) / f"{scene_id}_summary.json"
        save_json(summary, str(out_path))


if __name__ == "__main__":
    main()
