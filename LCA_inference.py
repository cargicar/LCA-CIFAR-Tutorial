"""
LCA inference — loads a saved dictionary (lca_dl.pth) and runs sparse coding.

Usage
-----
    python LCA_inference.py <path/to/lca_dl.pth> [config.yaml]

If config.yaml is omitted, the script looks for one in the experiment directory
that contains the .pth file (i.e. two levels up from models/lca_dl.pth), then
falls back to ./config.yaml.

Output mirrors LCA_torch.py: same printed metrics and reconstruction plots,
saved to a new timestamped directory under experiments/inference_<datetime>/.
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

from LCA_torch import LCA

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

if len(sys.argv) < 2:
    print("Usage: python LCA_inference.py <lca_dl.pth> [config.yaml]")
    sys.exit(1)

pth_path = sys.argv[1]

if len(sys.argv) > 2:
    cfg_path = sys.argv[2]
else:
    # Auto-detect: pth lives at <exp_dir>/models/lca_dl.pth
    candidate = os.path.join(os.path.dirname(os.path.dirname(pth_path)), 'config.yaml')
    cfg_path = candidate if os.path.exists(candidate) else 'config.yaml'

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

exp_dir   = os.path.join('experiments', 'inference_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
plots_dir = os.path.join(exp_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)
shutil.copy(cfg_path, os.path.join(exp_dir, 'config.yaml'))


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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(42)

print(f"Experiment dir: {exp_dir}")
print(f"Model:          {pth_path}")
print(f"Config:         {cfg_path}")
print(f"Device:         {device}\n")

N         = cfg['model']['n_features']
M         = cfg['model']['n_atoms']
BATCH     = cfg['data']['batch']
threshold = cfg['dictionary_learning']['threshold']

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

# Construct with learn_dict=True to match the saved state_dict key layout
# (Phi saved as nn.Parameter, no G buffer).
lca = LCA(
    torch.zeros(N, M),
    lam=cfg['model']['lam'],
    threshold=threshold,
    tau=cfg['model']['tau'],
    n_iter=cfg['model']['n_iter'],
    dt=cfg['model']['dt'],
    learn_dict=True,
).to(device)

lca.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))

# Patch to pure-inference mode: pre-compute G so _forward_inference works.
with torch.no_grad():
    G = lca.Phi.detach().T @ lca.Phi.detach() - torch.eye(M, device=device)
lca.register_buffer('G', G)
lca.learn_dict = False
lca.eval()

print(f"Loaded dictionary  Φ  shape: {lca.Phi.shape}  from {pth_path}\n")

# ---------------------------------------------------------------------------
# Load images
# ---------------------------------------------------------------------------

image_paths = glob.glob(cfg['data']['image_glob'])
sampled     = random.sample(image_paths, BATCH)

images = []
for path in sampled:
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr - arr.mean()
    images.append(arr.flatten())

s = torch.tensor(np.stack(images), device=device)
print(f"Loaded {BATCH} CIFAR-10 images, signal shape: {s.shape}\n")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

with torch.no_grad():
    a, s_hat = lca(s)

print(f"=== LCA inference  ({threshold} threshold, λ={cfg['model']['lam']}) ===")
print(f"  Sparsity (fraction zero):  {lca.sparsity(a):.3f}")
print(f"  Relative recon error:      {lca.reconstruction_error(s, s_hat):.6f}")
print(f"  Active coefficients/item:  {(a != 0).float().sum(dim=1).mean():.1f} / {M}")
print()

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def to_image(vec):
    img = vec.cpu().numpy().reshape(32, 32, 3)
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img


n_images = cfg['output']['n_images']
fig, axes = plt.subplots(n_images, 2, figsize=(4, 2 * n_images))
axes[0, 0].set_title("Original")
axes[0, 1].set_title(f"LCA recon ({threshold})")
for i in range(n_images):
    axes[i, 0].imshow(to_image(s[i]))
    axes[i, 0].axis('off')
    axes[i, 1].imshow(to_image(s_hat[i]))
    axes[i, 1].axis('off')
plt.tight_layout()
out = os.path.join(plots_dir, 'reconstructions.png')
plt.savefig(out)
plt.close()
print(f"Saved {out}")

print("\nDone.")
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
_log.close()
