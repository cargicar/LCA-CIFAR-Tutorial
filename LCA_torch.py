"""
Locally Competitive Algorithm (LCA) — PyTorch implementation
Based on: Rozell et al., "Sparse Coding via Thresholding and Local Competition
          in Neural Circuits", Neural Computation 20, 2526-2563 (2008)

Two variants are implemented:
  - SLCA : soft-thresholding  → minimises ℓ₁ cost  (equivalent to BPDN/LASSO)
  - HLCA : hard-thresholding  → minimises ℓ₀-like cost (more aggressive sparsity)

Quick-start
-----------
    from lca import LCA
    import torch

    # Random overcomplete dictionary (N=64 input dim, M=128 atoms)
    dictionary = torch.randn(64, 128)
    dictionary = dictionary / dictionary.norm(dim=0, keepdim=True)   # unit-norm atoms

    lca = LCA(dictionary, lam=0.1, threshold='soft', tau=10.0, n_iter=300)

    s = torch.randn(16, 64)   # batch of 16 signals
    a, recon = lca(s)         # a: sparse codes (16,128),  recon: (16,64)

Usage — single GPU:
    python LCA_torch.py [config.yaml]

Usage — N GPUs (e.g. 4):
    torchrun --nproc_per_node=4 LCA_torch.py [config.yaml]
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Literal, Tuple


# ---------------------------------------------------------------------------
# Threshold functions
# ---------------------------------------------------------------------------

def soft_threshold(u: torch.Tensor, lam: float) -> torch.Tensor:
    """Soft (shrinkage) threshold: max(|u| - λ, 0) * sign(u).
    Corresponds to ℓ₁ cost function C(a) = |a|.
    """
    return torch.sign(u) * torch.relu(u.abs() - lam)


def hard_threshold(u: torch.Tensor, lam: float) -> torch.Tensor:
    """Hard threshold: u if |u| > λ, else 0.
    Corresponds to ℓ₀-like cost function C(a) = λ²/2 · 𝟙(|a|>λ).
    """
    return u * (u.abs() > lam).float()


# ---------------------------------------------------------------------------
# Core LCA module
# ---------------------------------------------------------------------------

class LCA(nn.Module):
    """Locally Competitive Algorithm for sparse coding.

    Parameters
    ----------
    dictionary : Tensor, shape (n_features, n_atoms)
        The dictionary Φ. Columns should be unit-norm atoms.
    lam : float
        Threshold / sparsity trade-off λ. Larger → sparser codes.
    threshold : 'soft' | 'hard'
        Which thresholding function to use (SLCA vs HLCA).
    tau : float
        Neural time constant τ (controls integration speed).
    n_iter : int
        Number of Euler integration steps.
    dt : float
        Step size for Euler integration. Should satisfy dt < tau.
    track_energy : bool
        If True, forward also returns the energy E at each step (inference only).
    learn_dict : bool
        If True, the dictionary is updated via gradient descent each forward pass.
    dict_lr : float
        Learning rate for dictionary update (used only when learn_dict=True).
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        lam: float = 0.1,
        threshold: Literal['soft', 'hard'] = 'soft',
        tau: float = 10.0,
        n_iter: int = 300,
        dt: float = 1.0,
        track_energy: bool = False,
        learn_dict: bool = False,
        dict_lr: float = 1e-3,
    ):
        super().__init__()

        self.lam = lam
        self.tau = tau
        self.n_iter = n_iter
        self.dt = dt
        self.track_energy = track_energy
        self.learn_dict = learn_dict
        self.dict_lr = dict_lr

        if threshold == 'soft':
            self.T = soft_threshold
        elif threshold == 'hard':
            self.T = hard_threshold
        else:
            raise ValueError(f"threshold must be 'soft' or 'hard', got '{threshold}'")

        if learn_dict:
            # Learnable: gradient will flow through Phi, G recomputed each forward
            self.Phi = nn.Parameter(dictionary.float())
        else:
            # Fixed: stored as buffer, G pre-computed once
            self.register_buffer('Phi', dictionary.float())
            G = self.Phi.T @ self.Phi
            G = G - torch.eye(G.shape[0], device=G.device)
            self.register_buffer('G', G)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run LCA inference (and optionally update the dictionary).

        Parameters
        ----------
        s : Tensor, shape (..., N)

        Returns
        -------
        a : Tensor, shape (..., M)  — sparse coefficients
        s_hat : Tensor, shape (..., N)  — reconstruction ŝ = Φ·a
        energies : list[float]  — only returned when track_energy=True and learn_dict=False
        """
        if self.learn_dict:
            return self._forward_with_learning(s)
        return self._forward_inference(s)

    def _forward_inference(self, s: torch.Tensor):
        b = s @ self.Phi
        u = torch.zeros_like(b)
        energies = [] if self.track_energy else None

        for _ in range(self.n_iter):
            a = self.T(u, self.lam)
            inhibition = a @ self.G.T
            u = u + self.dt * (b - u - inhibition) / self.tau
            if self.track_energy:
                energies.append(self._energy(s, a).mean().item())

        a = self.T(u, self.lam)
        s_hat = a @ self.Phi.T
        if self.track_energy:
            return a, s_hat, energies
        return a, s_hat

    def _forward_with_learning(self, s: torch.Tensor):
        Phi = self.Phi

        # Inference — no gradients through the ODE
        with torch.no_grad():
            G = Phi.T @ Phi - torch.eye(Phi.shape[1], device=Phi.device)
            b = s @ Phi
            u = torch.zeros_like(b)
            for _ in range(self.n_iter):
                a = self.T(u, self.lam)
                u = u + self.dt * (b - u - a @ G.T) / self.tau
            a = self.T(u, self.lam)

        # Dictionary update — gradient flows only through Phi here
        s_hat = a @ Phi.T
        recon_loss = 0.5 * (s - s_hat).pow(2).mean()
        recon_loss.backward()

        with torch.no_grad():
            self.Phi.data -= self.dict_lr * self.Phi.grad
            self.Phi.grad.zero_()
            self._normalise_dict()

        return a.detach(), s_hat.detach()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_dict(self):
        self.Phi.data = self.Phi.data / self.Phi.data.norm(dim=0, keepdim=True).clamp(min=1e-8)

    def _energy(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        s_hat = a @ self.Phi.T
        recon_loss = 0.5 * (s - s_hat).pow(2).sum(dim=-1)
        if self.T is soft_threshold:
            sparsity_cost = self.lam * a.abs().sum(dim=-1)
        else:
            sparsity_cost = (self.lam ** 2 / 2) * (a.abs() > self.lam).float().sum(dim=-1)
        return recon_loss + sparsity_cost

    @property
    def n_features(self) -> int:
        return self.Phi.shape[0]

    @property
    def n_atoms(self) -> int:
        return self.Phi.shape[1]

    def sparsity(self, a: torch.Tensor) -> float:
        return (a == 0).float().mean().item()

    def reconstruction_error(self, s: torch.Tensor, s_hat: torch.Tensor, relative: bool = True) -> float:
        mse = (s - s_hat).pow(2).mean().item()
        if relative:
            mse /= (s.pow(2).mean().item() + 1e-8)
        return mse


# ---------------------------------------------------------------------------
# Demo / usage example
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import glob
    import os
    import random
    import shutil
    import sys
    import time
    import yaml
    from datetime import datetime
    from PIL import Image
    import numpy as np
    from torch.utils.data import DataLoader, Dataset, DistributedSampler

    # ------------------------------------------------------------------ #
    # DDP helpers
    # ------------------------------------------------------------------ #

    def setup_ddp():
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        return dist.get_rank(), local_rank, dist.get_world_size()

    def cleanup_ddp():
        dist.destroy_process_group()

    class _Tee:
        """Write to multiple streams simultaneously (stdout + log file)."""
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    class _CIFARPNGDataset(Dataset):
        def __init__(self, image_glob):
            self.paths = sorted(glob.glob(image_glob))
            assert len(self.paths) > 0, f"No images found at {image_glob}"
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr - arr.mean()
            return torch.from_numpy(arr.flatten())  # (3072,)

    def main():
        # ------------------------------------------------------------------ #
        # Distributed setup
        # ------------------------------------------------------------------ #
        using_ddp = dist.is_available() and 'LOCAL_RANK' in os.environ
        if using_ddp:
            rank, local_rank, world_size = setup_ddp()
            device = torch.device(f'cuda:{local_rank}')
        else:
            rank = local_rank = 0
            world_size = 1
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        is_main = (rank == 0)

        # ------------------------------------------------------------------ #
        # Config
        # ------------------------------------------------------------------ #
        cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        # ------------------------------------------------------------------ #
        # Experiment directory — rank 0 generates name, broadcasts to all
        # ------------------------------------------------------------------ #
        if is_main:
            exp_dir = os.path.join('experiments', 'LCA_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        else:
            exp_dir = None

        if using_ddp:
            container = [exp_dir]
            dist.broadcast_object_list(container, src=0)
            exp_dir = container[0]

        plots_dir = os.path.join(exp_dir, 'plots')

        if is_main:
            os.makedirs(plots_dir, exist_ok=True)
            shutil.copy(cfg_path, os.path.join(exp_dir, 'config.yaml'))
            _log = open(os.path.join(exp_dir, 'run.log'), 'w')
            sys.stdout = _Tee(sys.__stdout__, _log)
            sys.stderr = _Tee(sys.__stderr__, _log)

        if using_ddp:
            dist.barrier()

        torch.manual_seed(42)
        random.seed(42)

        if is_main:
            print(f"Experiment dir: {exp_dir}")
            print(f"Device: {device}  GPUs: {world_size}  (config: {cfg_path})\n")

        # ------------------------------------------------------------------ #
        # 1. Load a batch of random CIFAR-10 images as signals
        # ------------------------------------------------------------------ #
        N          = cfg['model']['n_features']
        M          = cfg['model']['n_atoms']
        BATCH      = cfg['data']['batch']
        niter      = cfg['model']['n_iter']
        learn_dict = cfg['dictionary_learning']['enabled']
        learning_rate = cfg['dictionary_learning']['learning_rate']

        image_paths = glob.glob(cfg['data']['image_glob'])
        sampled = random.sample(image_paths, BATCH)

        images = []
        for path in sampled:
            img = Image.open(path).convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr - arr.mean()
            images.append(arr.flatten())

        s = torch.tensor(np.stack(images), device=device)
        if is_main:
            print(f"Loaded {BATCH} CIFAR-10 images, signal shape: {s.shape}\n")

        # ------------------------------------------------------------------ #
        # 2. Build a random overcomplete dictionary
        # ------------------------------------------------------------------ #
        Phi = torch.randn(N, M, device=device)
        Phi = Phi / Phi.norm(dim=0, keepdim=True)

        # ------------------------------------------------------------------ #
        # 3. Run SLCA (soft threshold) — rank 0 only, comparison purposes
        # ------------------------------------------------------------------ #
        slca = LCA(Phi, lam=cfg['model']['lam'],
                   threshold=cfg['inference']['slca_threshold'],
                   tau=cfg['model']['tau'], n_iter=niter,
                   track_energy=cfg['inference']['track_energy']).to(device)
        a_soft, s_hat_soft, energies_soft = slca(s)

        if is_main:
            print("=== SLCA (soft threshold) ===")
            print(f"  Sparsity (fraction zero):  {slca.sparsity(a_soft):.3f}")
            print(f"  Relative recon error:      {slca.reconstruction_error(s, s_hat_soft):.6f}")
            print(f"  Active coefficients/item:  {(a_soft != 0).float().sum(dim=1).mean():.1f} / {M}")
            print(f"  Energy (first→last iter):  {energies_soft[0]:.4f} → {energies_soft[-1]:.4f}\n")

        # ------------------------------------------------------------------ #
        # 4. Run HLCA (hard threshold)
        # ------------------------------------------------------------------ #
        hlca = LCA(Phi, lam=cfg['model']['lam'],
                   threshold=cfg['inference']['hlca_threshold'],
                   tau=cfg['model']['tau'], n_iter=niter,
                   track_energy=cfg['inference']['track_energy']).to(device)
        a_hard, s_hat_hard, energies_hard = hlca(s)

        if is_main:
            print("=== HLCA (hard threshold) ===")
            print(f"  Sparsity (fraction zero):  {hlca.sparsity(a_hard):.3f}")
            print(f"  Relative recon error:      {hlca.reconstruction_error(s, s_hat_hard):.6f}")
            print(f"  Active coefficients/item:  {(a_hard != 0).float().sum(dim=1).mean():.1f} / {M}")
            print(f"  Energy (first→last iter):  {energies_hard[0]:.4f} → {energies_hard[-1]:.4f}\n")

        # ------------------------------------------------------------------ #
        # 5. Reconstruction error comparison
        # ------------------------------------------------------------------ #
        if is_main:
            print("=== Reconstruction error comparison ===")
            print(f"  SLCA relative MSE: {slca.reconstruction_error(s, s_hat_soft):.6f}")
            print(f"  HLCA relative MSE: {hlca.reconstruction_error(s, s_hat_hard):.6f}")
            print()

        # ------------------------------------------------------------------ #
        # 6. Plot original vs reconstructed images
        # ------------------------------------------------------------------ #
        import matplotlib.pyplot as plt

        def plot_loss(losses, title="Loss", xlabel="Step", ylabel="MSE", filename="loss.png"):
            plt.figure()
            plt.plot(losses)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.savefig(filename)
            plt.close()
            print(f"Saved {filename}")

        def plot_reconstructions(s, s_hat_soft, s_hat_hard, n_images=4, filename="reconstructions.png", same_T=False):
            def to_image(vec):
                img = vec.detach().cpu().numpy().reshape(32, 32, 3)
                img = img - img.min()
                img = img / (img.max() + 1e-8)
                return img

            _, axes = plt.subplots(n_images, 3, figsize=(6, 2 * n_images))
            axes[0, 0].set_title("Original")
            if same_T:
                axes[0, 1].set_title("LCA_inference")
                axes[0, 2].set_title("LCA_dictionary_learning")
            else:
                axes[0, 1].set_title("SLCA recon")
                axes[0, 2].set_title("HLCA recon")

            for i in range(n_images):
                for ax, vec in zip(axes[i], [s[i], s_hat_soft[i], s_hat_hard[i]]):
                    ax.imshow(to_image(vec))
                    ax.axis('off')

            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
            print(f"Saved {filename}")

        if is_main:
            plot_reconstructions(s, s_hat_soft, s_hat_hard,
                                 n_images=cfg['output']['n_images'],
                                 filename=os.path.join(plots_dir, "reconstructions_inference.png"))

        # ------------------------------------------------------------------ #
        # 7. Dictionary learning — full dataset, epoch+batch loop
        # ------------------------------------------------------------------ #
        if learn_dict:
            epochs       = cfg['dictionary_learning']['epochs']
            print_freq   = cfg['dictionary_learning']['print_freq']
            anneal_every = cfg['dictionary_learning']['lambda_anneal_every']
            anneal_step  = cfg['dictionary_learning']['lambda_anneal_step']

            if is_main:
                print(f"=== Dictionary learning ({epochs} epochs, {world_size} GPU(s)) ===")
                print(f"    effective batch = {cfg['data']['batch_size']} × {world_size} = "
                      f"{cfg['data']['batch_size'] * world_size}\n")

            lca_dl = LCA(
                Phi, lam=cfg['model']['lam'],
                threshold=cfg['dictionary_learning']['threshold'],
                tau=cfg['model']['tau'],
                n_iter=cfg['dictionary_learning']['n_iter'],
                dt=cfg['model']['dt'],
                dict_lr=learning_rate,
                learn_dict=True
            ).to(device)

            # All ranks must start with identical weights.
            # _forward_with_learning uses backprop + manual update (not DDP),
            # so we keep replicas in sync with manual all_reduce after each batch.
            if using_ddp:
                dist.broadcast(lca_dl.Phi.data, src=0)

            dset = _CIFARPNGDataset(cfg['data']['image_glob'])
            sampler = (
                DistributedSampler(dset, num_replicas=world_size, rank=rank, shuffle=True)
                if using_ddp else None
            )
            dl_loader = DataLoader(
                dset,
                batch_size=cfg['data']['batch_size'],
                shuffle=(sampler is None),
                sampler=sampler,
                num_workers=cfg['data']['num_workers'],
                pin_memory=torch.cuda.is_available(),
                persistent_workers=cfg['data']['num_workers'] > 0,
            )

            losses = []
            lam = cfg['model']['lam']

            for epoch in range(epochs):
                if sampler is not None:
                    sampler.set_epoch(epoch)

                t0 = time.time()

                if epoch > 0 and epoch % anneal_every == 0:
                    lam += anneal_step
                    lca_dl.lam = lam
                    if is_main:
                        print(f"  [anneal] λ → {lam:.3f}")

                ep_mse = ep_sparsity = ep_active = ep_rel_err = 0.0

                for batch_s in dl_loader:
                    batch_s = batch_s.to(device)
                    a_dl, s_hat_dl = lca_dl(batch_s)
                    err = (batch_s - s_hat_dl).pow(2).mean().item()

                    # Each rank has applied its own gradient update to Phi.
                    # Average the updated weights across all GPUs, then
                    # re-normalise (mean of unit-norm vectors is not unit-norm).
                    if using_ddp:
                        dist.all_reduce(lca_dl.Phi.data, op=dist.ReduceOp.SUM)
                        lca_dl.Phi.data /= world_size
                        lca_dl._normalise_dict()

                    if is_main:
                        losses.append(err)

                    ep_mse      += err
                    ep_sparsity += (a_dl == 0).float().mean().item()
                    ep_active   += (a_dl != 0).float().sum(dim=1).mean().item()
                    ep_rel_err  += err / (batch_s.pow(2).mean().item() + 1e-8)

                nb = len(dl_loader)
                epoch_time = time.time() - t0

                if is_main:
                    print(f"Epoch {epoch:02d} | {epoch_time:.1f}s ({epoch_time/nb:.2f}s/batch) | "
                          f"Sparsity: {ep_sparsity/nb:.3f}  "
                          f"Active: {ep_active/nb:.1f}/{M}  "
                          f"Rel.err: {ep_rel_err/nb:.6f}  "
                          f"recon MSE: {ep_mse/nb:.6f}  "
                          f"λ={lam:.3f}")

                    models_dir = os.path.join(exp_dir, 'models')
                    os.makedirs(models_dir, exist_ok=True)
                    torch.save(lca_dl.state_dict(), os.path.join(models_dir, 'lca_dl.pth'))

            # Final summary — run inference on the fixed comparison batch s
            if is_main:
                print(f"\n=== LCA Dictionary Learning "
                      f"({cfg['dictionary_learning']['threshold']} threshold, λ={lam:.3f}) ===")

            with torch.no_grad():
                G = lca_dl.Phi.detach().T @ lca_dl.Phi.detach() - torch.eye(M, device=device)
            lca_dl.register_buffer('G', G)
            lca_dl.learn_dict = False
            a_dl, s_hat_dl = lca_dl(s)

            if is_main:
                print(f"  Sparsity (fraction zero):  {lca_dl.sparsity(a_dl):.3f}")
                print(f"  Relative recon error:      {lca_dl.reconstruction_error(s, s_hat_dl):.6f}")
                print(f"  Active coefficients/item:  {(a_dl != 0).float().sum(dim=1).mean():.1f} / {M}")

                plot_loss(losses, title="Dictionary Learning — Reconstruction MSE",
                          filename=os.path.join(plots_dir, "dict_learning_loss.png"))

                plot_reconstructions(s, s_hat_hard, s_hat_dl,
                                     n_images=cfg['output']['n_images'],
                                     filename=os.path.join(plots_dir, "reconstructions_dictionary_learning.png"),
                                     same_T=True)

                model_path = os.path.join(models_dir, 'lca_dl.pth')
                print(f"Saved learned dictionary → {model_path}")

        if is_main:
            print("\nDone.")
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            _log.close()

        if using_ddp:
            cleanup_ddp()

    main()
