"""Generate egocentric captions with the released Qwen LoRA."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from inference_inputs import read_prompts, select_inputs
from system_prompt import SYSTEM_PROMPT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", required=True)
    p.add_argument("--lora", default=str(ROOT / "weights/qwen_lora.ckpt"))
    p.add_argument("--subject-csv", required=True)
    p.add_argument("--video-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1234)
    a = p.parse_args()
    subjects = read_prompts(a.subject_csv)
    _, names = select_inputs(a.video_root, subjects)
    output = Path(a.output)
    if output.exists():
        p.error(f"Output already exists: {output}")
    # Imports are deferred so --help and input checks do not need a GPU environment.
    import csv
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    from lora_weights import load_lora_checkpoint
    from llamafactory.chat import ChatModel
    from prepare_inputs import prepare_inputs

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen inference requires a CUDA GPU")
    # Match the previously exercised video preprocessing and no-thinking template.
    chat = ChatModel(dict(
        model_name_or_path=a.base_model, template="qwen3_vl_nothink",
        infer_backend="huggingface", trust_remote_code=True, enable_thinking=False,
        cutoff_len=4096, video_max_pixels=16384, video_fps=2.0, video_maxlen=128))
    engine = chat.engine
    config = LoraConfig.from_pretrained(str(Path(a.lora).parent / "qwen_lora"))
    config.inference_mode = True
    model = get_peft_model(engine.model, config).eval()
    expected = load_lora_checkpoint(a.lora)
    set_peft_model_state_dict(model, expected)
    model.requires_grad_(False)
    actual = get_peft_model_state_dict(model)
    if set(expected) != set(actual) or len(expected) != 896:
        raise ValueError("Qwen adapter tensor keys do not match the model")
    if not all(torch.equal(v.cpu(), expected[k].to(v.dtype)) for k, v in actual.items()):
        raise ValueError("Loaded Qwen adapter differs from the supplied tensors")
    matched = len(expected)
    del actual, expected
    layers = [m for m in model.modules() if hasattr(m, "disable_adapters") and hasattr(m, "lora_A")]
    if not layers or any(m.disable_adapters for m in layers):
        raise RuntimeError("Qwen LoRA is not enabled")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tpv", "prompt"])
        writer.writeheader()
        records = []
        for index, name in enumerate(names):
            message = (f"Subject description: {subjects[name]}\n"
                       "Please generate a first-person egocentric reasoning caption from this subject's viewpoint.")
            kwargs, prompt_len = prepare_inputs(
                model, engine.tokenizer, engine.processor, engine.template, engine.generating_args,
                [{"role": "user", "content": message}], SYSTEM_PROMPT, None, None,
                [str(Path(a.video_root) / name)], None,
                dict(max_new_tokens=512, temperature=0.0, top_p=1.0, top_k=50, repetition_penalty=1.0))
            gc = kwargs["generation_config"]
            gc.do_sample = False
            gc.temperature = gc.top_p = gc.top_k = None
            kwargs["use_model_defaults"] = False
            torch.manual_seed(a.seed + index)
            torch.cuda.manual_seed_all(a.seed + index)
            start = time.time()
            with torch.inference_mode():
                result = model.generate(**kwargs)
            ids = result[0, prompt_len:]
            caption = engine.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
            if not caption:
                raise RuntimeError(f"Empty Qwen output for {name}")
            writer.writerow({"tpv": name, "prompt": caption})
            handle.flush()
            records.append({"video": name, "subject": subjects[name], "prompt": caption,
                            "token_ids": ids.cpu().tolist(), "seconds": time.time() - start})
            print("DONE", name, flush=True)
    output.with_suffix(".metadata.json").write_text(json.dumps({
        "base_model": a.base_model, "adapter": a.lora, "matched_lora_tensors": matched,
        "generation": {"do_sample": False, "use_model_defaults": False, "max_new_tokens": 512},
        "video_preprocessing": {"max_pixels": 16384, "fps": 2.0, "max_frames": 128},
        "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
