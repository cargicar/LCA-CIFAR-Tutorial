"""
lcapt inference — loads a saved LCAConv2D (lca_cifar_lcapt.pth) and runs sparse coding
on a fresh random batch of CIFAR-10 images.

Usage
-----
    python lcapt_inference.py <path/to/lca_cifar_lcapt.pth> [config_lcapt.yaml]

If config_lcapt.yaml is omitted, looks for one in the experiment directory
that contains the .pth file (two levels up from models/), then falls back to
./config_lcapt.yaml.

Output is saved to a new timestamped experiments/lcapt_inference_<datetime>/ directory.
"""

import glob
import os
import random
import shutil
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from lcapt.lca import LCAConv2D
from lcapt.metric import compute_l1_sparsity, compute_l2_error

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python lcapt_inference.py <lca_cifar_lcapt.pth> [config_lcapt.yaml]")
    sys.exit(1)

pth_path = sys.argv[1]

if len(sys.argv) > 2:
    cfg_path = sys.argv[2]
else:
    candidate = os.path.join(os.path.dirname(os.path.dirname(pth_path)), 'config_lcapt.yaml')
    cfg_path  = candidate if os.path.exists(candidate) else 'config_lcapt.yaml'

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

exp_dir   = os.path.join('experiments', 'lcapt_inference_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
plots_dir = os.path.join(exp_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
shutil.copy(cfg_path, os.path.join(exp_dir, 'config_lcapt.yaml'))


class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


_log = open(os.path.join(exp_dir, 'run.log'), 'w')
sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype  = torch.float16 if cfg['training']['dtype'] == 'float16' else torch.float32

print(f"Experiment dir: {exp_dir}")
print(f"Model:          {pth_path}")
print(f"Config:         {cfg_path}")
print(f"Device:         {device}  dtype={dtype}\n")

BATCH = cfg['data']['batch_size']

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

lca = LCAConv2D(
    out_neurons=cfg['model']['features'],
    in_neurons=cfg['model']['in_channels'],
    result_dir=os.path.join(exp_dir, 'lca_results'),
    kernel_size=cfg['model']['kernel_size'],
    stride=cfg['model']['stride'],
    lambda_=cfg['model']['lambda_'],
    tau=cfg['model']['tau'],
    lca_iters=cfg['model']['lca_iters'],
    return_vars=['inputs', 'acts', 'recons', 'recon_errors'],
).to(dtype=dtype, device=device)

lca.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
lca.eval()

print(f"Loaded LCAConv2D  weights shape: {lca.weights.shape}  from {pth_path}\n")

# ---------------------------------------------------------------------------
# Load images  (no seed → different batch every run)
# ---------------------------------------------------------------------------

class _CIFARPNGDataset(Dataset):
    def __init__(self, image_glob):
        self.paths = sorted(glob.glob(image_glob))
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr.transpose(2, 0, 1))  # (C, H, W)

dset      = _CIFARPNGDataset(cfg['data']['image_glob'])
loader    = DataLoader(dset, batch_size=BATCH, shuffle=True, num_workers=0)
images    = next(iter(loader)).to(dtype=dtype, device=device)
print(f"Loaded {images.shape[0]} CIFAR-10 images, shape: {images.shape}\n")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

with torch.no_grad():
    inputs, code, recon, recon_error = lca(images)

n_total  = code.shape[1] * code.shape[2] * code.shape[3]
sparsity = (code == 0).float().mean().item()
active   = (code != 0).float().sum(dim=(1, 2, 3)).mean().item()
l1       = compute_l1_sparsity(code, lca.lambda_).item()
l2       = compute_l2_error(inputs, recon).item()
rel_err  = (
    (inputs - recon).pow(2).sum(dim=(1, 2, 3)) /
    (inputs.pow(2).sum(dim=(1, 2, 3)) + 1e-8)
).mean().item()

print(f"=== LCAConv2D inference  (λ={lca.lambda_:.3f}) ===")
print(f"  Sparsity (fraction zero):  {sparsity:.3f}")
print(f"  Relative recon error:      {rel_err:.6f}")
print(f"  Active coefficients/item:  {active:.1f} / {n_total}")
print(f"  L2 recon error:            {l2:.4f}")
print(f"  L1 sparsity cost:          {l1:.4f}")
print(f"  Total energy (L2+L1):      {l2+l1:.4f}")
print()

# ---------------------------------------------------------------------------
# Reconstruction plot
# ---------------------------------------------------------------------------

n = cfg['output']['n_images']

def to_rgb(tensor):
    arr = tensor.float().cpu().numpy().transpose(1, 2, 0)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
axes[0, 0].set_title('Input')
axes[0, 1].set_title('Reconstruction')
axes[0, 2].set_title('Recon Error')

for i in range(n):
    inp = recon[i] + recon_error[i]   # lcapt normalizes internally; recover original
    axes[i, 0].imshow(to_rgb(inp))
    axes[i, 1].imshow(to_rgb(recon[i]))
    axes[i, 2].imshow(to_rgb(recon_error[i]))
    for ax in axes[i]:
        ax.axis('off')

plt.tight_layout()
out = os.path.join(plots_dir, 'reconstructions.png')
plt.savefig(out)
plt.close()
print(f"Saved {out}")

print("\nDone.")
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
_log.close()
