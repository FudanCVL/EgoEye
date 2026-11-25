"""CPU-only validation shared by the two inference routes."""
import csv
import json
from pathlib import Path


def read_prompts(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"tpv", "prompt"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected CSV columns tpv,prompt")
        result = {}
        for row in reader:
            name = Path(row["tpv"].strip()).name
            prompt = row["prompt"].strip()
            if not name.lower().endswith(".mp4") or not prompt:
                raise ValueError(f"{path}: invalid video name or empty prompt: {name!r}")
            if name in result:
                raise ValueError(f"{path}: duplicate video basename: {name}")
            result[name] = prompt
    if not result:
        raise ValueError(f"{path}: empty CSV")
    return result


def select_inputs(video_root, prompts, names_file=None, limit=0):
    if limit < 0:
        raise ValueError("limit must be non-negative")
    root = Path(video_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {root}")
    all_names = sorted(p.name for p in root.glob("*.mp4") if p.is_file())
    names = json.loads(Path(names_file).read_text()) if names_file else sorted(prompts)
    if (not isinstance(names, list) or not names or
            any(not isinstance(n, str) or Path(n).name != n for n in names) or
            len(set(names)) != len(names)):
        raise ValueError("Video selection must be a nonempty JSON list of unique basenames")
    missing = set(names) - set(all_names)
    if missing:
        raise ValueError(f"Missing input videos: {sorted(missing)}")
    missing = set(names) - set(prompts)
    if missing:
        raise ValueError(f"Missing prompts/subjects: {sorted(missing)}")
    # Match the historical sorted input traversal, including for --limit.
    wanted = set(names)
    selected = [n for n in all_names if n in wanted]
    if limit:
        selected = selected[:limit]
    return all_names, selected


def write_prompts(path, prompts, names):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tpv", "prompt"])
        writer.writeheader()
        writer.writerows({"tpv": n, "prompt": prompts[n]} for n in names)
