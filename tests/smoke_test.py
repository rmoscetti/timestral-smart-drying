import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import numpy as np
import torch

import timestral
from timestral import TimestralForecaster

print(f"python={sys.version.split()[0]}")
print(f"timestral={timestral.__version__}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    print(f"cuda_capability={torch.cuda.get_device_capability(0)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"selected_device={device}")

forecaster = TimestralForecaster.from_pretrained(
    "mistralai/Timestral-209M",
    device=device,
    cache_dir=str(HF_CACHE),
)
print(f"model_device={forecaster.device}")
print(f"num_parameters={sum(p.numel() for p in forecaster.model.parameters())}")

rng = np.random.default_rng(42)
context = (rng.standard_normal((2, 1, 512)).cumsum(-1) + 100.0).astype(np.float32)
forecast = forecaster.forecast(context, prediction_length=96)

quantiles = forecast.quantiles
print(f"quantiles_shape={tuple(quantiles.shape)}")
print(f"quantile_levels={forecast.quantile_levels}")
print(f"finite={bool(torch.isfinite(quantiles).all())}")
print(f"sorted={bool((quantiles[..., 1:] >= quantiles[..., :-1]).all())}")
print(f"median_shape={tuple(forecast.median.shape)}")
print(f"median_head={forecast.median[0, 0, :5].tolist()}")

if torch.cuda.is_available():
    torch.cuda.synchronize()
    print(f"cuda_memory_allocated_mb={torch.cuda.memory_allocated() / 1024**2:.1f}")
    print(f"cuda_memory_reserved_mb={torch.cuda.memory_reserved() / 1024**2:.1f}")

assert timestral.__version__ == "1.0.0"
assert quantiles.shape == (2, 1, 96, 21)
assert torch.isfinite(quantiles).all()
assert (quantiles[..., 1:] >= quantiles[..., :-1]).all()
print("SMOKE_TEST=PASS")
