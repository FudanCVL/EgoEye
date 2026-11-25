"""Run TPV -> Qwen -> prompt -> Wan, or TPV + saved prompt -> Wan."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from inference_inputs import read_prompts, select_inputs, write_prompts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["one", "two"], required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--output", required=True, help="New output directory")
    p.add_argument("--benchmark", choices=["EgoEval", "EgoCross"])
    p.add_argument("--subject-csv", help="tpv,prompt CSV; prompt is a subject description")
    p.add_argument("--prompt-csv", help="tpv,prompt CSV; prompt is a generated caption")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--wan-model", default="/tmp/egoeye_models/wan22")
    p.add_argument("--qwen-model", default="/tmp/egoeye_models/qwen3vl")
    p.add_argument("--qwen-python", default=sys.executable, help="Python executable with the Qwen dependencies")
    p.add_argument("--qwen-lora", default=str(ROOT / "weights/qwen_lora.ckpt"))
    p.add_argument("--wan-lora", default=str(ROOT / "weights/wan22_lora.ckpt"))
    p.add_argument("--dry-run", action="store_true", help="Check input selection and print commands without loading models or writing output")
    a = p.parse_args()
    if (a.stage == "one" and a.prompt_csv) or (a.stage == "two" and a.subject_csv):
        p.error("one uses --subject-csv; two uses --prompt-csv")
    source = a.subject_csv if a.stage == "one" else a.prompt_csv
    if not source and a.stage == "two" and a.benchmark:
        source = ROOT / "prompts" / a.benchmark / "PROMPT.csv"
    if not source:
        p.error("Supply --subject-csv for stage one, or --prompt-csv / --benchmark for stage two")
    rows = read_prompts(source)
    all_names, names = select_inputs(a.video_root, rows, None, a.limit)
    out = Path(a.output).resolve()
    if out.exists():
        p.error(f"Output already exists: {out}; choose a new directory")
    wan_python = Path(sys.executable)
    qwen_python = Path(a.qwen_python).resolve()
    qwen_model, wan_model = Path(a.qwen_model).resolve(), Path(a.wan_model).resolve()
    commands = []
    if a.stage == "one":
        commands.append([str(qwen_python), str(ROOT / "code/qwen_prompt/eval_generate.py"),
                         "--base-model", str(qwen_model), "--lora", str(Path(a.qwen_lora).resolve()),
                         "--subject-csv", str(out / "subjects.csv"), "--video-root", str(Path(a.video_root).resolve()),
                         "--output", str(out / "prompts.csv"), "--seed", str(a.seed)])
    wan_cmd = [str(wan_python), str(ROOT / "scripts/infer_wan22.py"),
               "--model", str(wan_model), "--lora", str(Path(a.wan_lora).resolve()),
               "--video-root", str(Path(a.video_root).resolve()), "--prompt-csv", str(out / "prompts.csv"),
               "--names-file", str(out / "videos.json"), "--output", str(out / "videos"), "--seed", str(a.seed)]
    commands.append(wan_cmd)
    print(f"Stage={a.stage}; selected={len(names)}; source videos={len(all_names)}", flush=True)
    for command in commands:
        print(shlex.join(command), flush=True)
    if a.dry_run:
        return
    required = [wan_python, wan_model / "model_index.json", Path(a.wan_lora)]
    if a.stage == "one":
        required += [qwen_python, qwen_model / "config.json", Path(a.qwen_lora), Path(a.qwen_lora).parent / "qwen_lora/adapter_config.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing inference asset: {path}; see README setup instructions")
    out.mkdir(parents=True)
    (out / "videos.json").write_text(json.dumps(names, indent=2) + "\n")
    if a.stage == "one":
        write_prompts(out / "subjects.csv", rows, names)
    else:
        write_prompts(out / "prompts.csv", rows, names)
    manifest = {"stage": a.stage, "benchmark": a.benchmark, "input_csv": str(Path(source).resolve()),
                "source_video_order": all_names, "selected_videos": names, "commands": commands, "status": "running"}
    manifest_path = out / "run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for command in commands:
            subprocess.run(command, check=True)
    except BaseException:
        manifest["status"] = "failed"
        raise
    else:
        manifest["status"] = "complete"
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
