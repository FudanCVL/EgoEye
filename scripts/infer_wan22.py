"""Wan2.2 EgoEye inference script."""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from diffusers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code/wan22"))
from inference_inputs import read_prompts, select_inputs
from inference_model import EgoEyeWanModel
from models.wan2.custom_pipeline import CustomWanPipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-root", required=True)
    p.add_argument("--names-file")
    p.add_argument("--model", default="/tmp/egoeye_models/wan22")
    p.add_argument("--lora", default=str(ROOT / "weights/wan22_lora.ckpt"))
    p.add_argument("--prompt-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--default-prompt", action="store_true")
    p.add_argument("--rng-mode", choices=["serial", "per_video"], default="serial")
    a = p.parse_args()

    assert torch.cuda.is_available(), "CUDA is required for inference"
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    if a.height % 32 or a.width % 32 or min(a.height, a.width) <= 0:
        p.error("height and width must be positive multiples of 32")
    if a.shards < 1 or not 0 <= a.shard < a.shards:
        p.error("Require shards >= 1 and 0 <= shard < shards")

    prompts = read_prompts(a.prompt_csv)
    all_names, names = select_inputs(a.video_root, prompts, a.names_file, a.limit)
    out = Path(a.output)
    selected = [n for n in all_names if n in names and names.index(n) % a.shards == a.shard]
    if not selected:
        p.error("This shard has no selected inputs")

    for name in selected:
        if (out / name).exists():
            raise FileExistsError(f"Output already exists: {out / name}")
        reader = VideoReader(str(Path(a.video_root) / name), ctx=cpu(0), num_threads=1)
        if len(reader) < 49:
            raise ValueError(f"{name}: at least 49 frames are required")
        del reader

    model = EgoEyeWanModel(a.model, a.lora).to("cuda").eval()
    scheduler = UniPCMultistepScheduler.from_config(
        FlowMatchEulerDiscreteScheduler.from_pretrained(a.model, subfolder="scheduler").config,
        flow_shift=5,
    )
    pipe = CustomWanPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        transformer=model.transformer,
        scheduler=scheduler,
    )
    out.mkdir(parents=True, exist_ok=True)

    pre = transforms.Compose([
        transforms.Resize((a.height, a.width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    shape = (1, model.vae.config.z_dim, 13, a.height // 16, a.width // 16)
    posterior_dtype = torch.bfloat16

    completed = 0
    for index, name in enumerate(all_names):
        is_selected = name in names and names.index(name) % a.shards == a.shard
        if a.rng_mode == "per_video":
            torch.cuda.manual_seed_all(a.seed + index)
        if not is_selected:
            if a.rng_mode == "serial":
                torch.randn(shape, device="cuda", dtype=posterior_dtype)
                torch.randn(shape, device="cuda", dtype=torch.float32)
            continue
        if a.limit and completed >= a.limit:
            break

        prompt = "Transform it into the first-person perspective." if a.default_prompt else prompts[name]
        reader = VideoReader(str(Path(a.video_root) / name), ctx=cpu(0), num_threads=1)
        frames = reader.get_batch(list(range(49))).asnumpy()
        pixels = torch.stack([pre(Image.fromarray(f)) for f in frames]).permute(1, 0, 2, 3).unsqueeze(0).cuda()

        start = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            posterior = model.vae.encode(pixels).latent_dist
            posterior_dtype = posterior.mean.dtype
            lat = posterior.sample()
            lat = (lat - model.latents_mean) / model.latents_std
            result = pipe(
                prompt=[prompt],
                height=a.height,
                width=a.width,
                num_frames=49,
                num_inference_steps=50,
                guidance_scale=5.0,
                attention_kwargs={
                    "encoder_condition_states": lat,
                    "encoder_ref_states": None,
                    "use_collaborative_position_encoding": True,
                },
            ).frames[0]

        export_to_video(result, str(out / name), fps=8)
        print("DONE", name, f"{time.time() - start:.2f}s", flush=True)
        completed += 1


if __name__ == "__main__":
    main()
