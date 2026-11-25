"""Wan components and strict EgoEye LoRA loading, without a training system."""
import torch
from torch import nn
from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5EncoderModel
from peft import LoraConfig
from lora_weights import load_lora_checkpoint
from models.wan2.transformer_wan import CustomWanTransformer3DModel


class EgoEyeWanModel(nn.Module):
    def __init__(self, model_path, lora_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        # Preserve the audited fp32 parameter / bf16 autocast inference behavior.
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch.float32)
        self.vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.float32)
        self.transformer = CustomWanTransformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.float32)
        self.register_buffer("latents_mean", torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1))
        self.register_buffer("latents_std", torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1))
        self.transformer.add_adapter(LoraConfig(
            r=80, lora_alpha=80, init_lora_weights=True,
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]))
        lora = load_lora_checkpoint(lora_path)
        expected = {k for k in self.transformer.state_dict() if "lora_" in k}
        if set(lora) != expected or len(lora) != 600:
            raise ValueError(f"LoRA mismatch: missing={sorted(expected-set(lora))}, unexpected={sorted(set(lora)-expected)}")
        result = self.transformer.load_state_dict(lora, strict=False)
        if result.unexpected_keys:
            raise ValueError(result.unexpected_keys)
        self.matched_lora_tensors = len(lora)
        self.requires_grad_(False)
        self.eval()
