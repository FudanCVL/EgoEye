"""Read released tensor-only LoRA checkpoints without loading arbitrary objects."""
from pathlib import Path


def load_lora_checkpoint(path):
    import torch
    state = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty LoRA tensor dictionary")
    if not all(isinstance(key, str) and "lora_" in key and isinstance(value, torch.Tensor)
               for key, value in state.items()):
        raise ValueError("Checkpoint contains entries other than LoRA tensors")
    return state
