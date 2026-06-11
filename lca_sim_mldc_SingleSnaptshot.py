"""
3D convolutional LCA dictionary learning on a single JHTDB simulation snapshot.

Loads one 3D pressure (or velocity) field from an HDF5 file, extracts random
3D patches, and trains LCAConv3D via Hebbian updates.

Supports single-GPU and multi-GPU training via torch.distributed (manual all_reduce).

Usage — single GPU:
    python lca_sim_mldc.py [config_simmldc.yaml]

Usage — N GPUs (e.g. 4):
    torchrun --nproc_per_node=4 lca_sim_mldc.py [config_simmldc.yaml]
"""

import os
import shutil
import sys
import time
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from lcapt.lca import LCAConv3D
from lcapt.metric import compute_l1_sparsity, compute_l2_error


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int]:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), local_rank, dist.get_world_size()


def cleanup_ddp() -> None:
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Dataset — random 3D patch crops from a single HDF5 snapshot
# ---------------------------------------------------------------------------

class _HDF5PatchDataset(Dataset):
    """
    Loads one timestep from an HDF5 volume and serves random 3D crops.

    The HDF5 field is expected at key 'pressure' (or 'Vx'/'Vy'/'Vz') with
    shape (nt, nx, ny, nz).  Each __getitem__ returns a (1, P, P, P) patch
    drawn from a random location within the volume.

    n_patches sets the virtual epoch size — there is no fixed enumeration of
    patches; each call samples independently, giving effectively infinite
    augmentation from a single snapshot.
    """
    def __init__(self, h5_path: str, field_key: str, timestep: int,
                 patch_size: int, n_patches: int):
        with h5py.File(h5_path, 'r') as f:
            vol = f[field_key][timestep]          # (nx, ny, nz)  float32
        # Normalise to zero mean, unit variance so LCA starts in a stable range
        vol = (vol - vol.mean()) / (vol.std() + 1e-8)
        self.vol        = torch.from_numpy(vol.astype(np.float32))   # (nx, ny, nz)
        self.patch_size = patch_size
        self.n_patches  = n_patches
        nx, ny, nz      = vol.shape
        assert min(nx, ny, nz) >= patch_size, \
            f"Volume {vol.shape} smaller than patch_size={patch_size}"
        self.limits = (nx - patch_size, ny - patch_size, nz - patch_size)

    def __len__(self):
        return self.n_patches

    def __getitem__(self, _idx):
        x = torch.randint(0, self.limits[0] + 1, (1,)).item()
        y = torch.randint(0, self.limits[1] + 1, (1,)).item()
        z = torch.randint(0, self.limits[2] + 1, (1,)).item()
        p = self.patch_size
        patch = self.vol[x:x+p, y:y+p, z:z+p]   # (P, P, P)
        return patch.unsqueeze(0)                  # (1, P, P, P) — channel dim


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ------------------------------------------------------------------ #
    # DDP
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
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'config_simmldc.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # Experiment directory
    # ------------------------------------------------------------------ #
    if is_main:
        exp_dir = os.path.join(
            'experiments', 'simmldc_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        )
    else:
        exp_dir = None

    if using_ddp:
        container = [exp_dir]
        dist.broadcast_object_list(container, src=0)
        exp_dir = container[0]

    plots_dir  = os.path.join(exp_dir, 'plots')
    models_dir = os.path.join(exp_dir, 'models')

    if is_main:
        os.makedirs(plots_dir,  exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        shutil.copy(cfg_path, os.path.join(exp_dir, 'config_simmldc.yaml'))
        _log       = open(os.path.join(exp_dir, 'run.log'), 'w')
        sys.stdout = _Tee(sys.__stdout__, _log)
        sys.stderr = _Tee(sys.__stderr__, _log)

    if using_ddp:
        dist.barrier()

    # ------------------------------------------------------------------ #
    # dtype
    # ------------------------------------------------------------------ #
    dtype_str = cfg['training'].get('dtype', 'float32')
    dtype = {'float16': torch.float16,
             'bfloat16': torch.bfloat16}.get(dtype_str, torch.float32)

    if is_main:
        print(f"Experiment dir : {exp_dir}")
        print(f"Config         : {cfg_path}")
        print(f"Device         : {device}  dtype={dtype}")
        print(f"GPUs           : {world_size}\n")

    # ------------------------------------------------------------------ #
    # Data — single 3D snapshot, served as random patches
    # ------------------------------------------------------------------ #
    dcfg = cfg['data']
    dset = _HDF5PatchDataset(
        h5_path    = dcfg['h5_path'],
        field_key  = dcfg['field_key'],
        timestep   = dcfg['timestep'],
        patch_size = dcfg['patch_size'],
        n_patches  = dcfg['n_patches'],
    )

    sampler = (
        DistributedSampler(dset, num_replicas=world_size, rank=rank, shuffle=True)
        if using_ddp else None
    )
    dataloader = DataLoader(
        dset,
        batch_size  = dcfg['batch_size'],
        shuffle     = (sampler is None),
        sampler     = sampler,
        num_workers = dcfg['num_workers'],
        pin_memory  = True,
        persistent_workers = dcfg['num_workers'] > 0,
    )

    if is_main:
        nx, ny, nz = dset.vol.shape
        print(f"Volume   : {nx}×{ny}×{nz}  |  field={dcfg['field_key']}  t={dcfg['timestep']}")
        print(f"Patches  : {dcfg['n_patches']} patches/epoch  |  size={dcfg['patch_size']}³")
        print(f"Batches  : {len(dataloader)}/GPU/epoch  |  "
              f"effective batch={dcfg['batch_size'] * world_size}\n")

    # ------------------------------------------------------------------ #
    # Model — LCAConv3D
    # ------------------------------------------------------------------ #
    mcfg = cfg['model']
    lca = LCAConv3D(
        out_neurons  = mcfg['features'],
        in_neurons   = mcfg['in_channels'],
        result_dir   = os.path.join(exp_dir, 'lca_results'),
        kernel_size  = mcfg['kernel_size'],
        stride       = mcfg['stride'],
        lambda_      = mcfg['lambda_'],
        tau          = mcfg['tau'],
        lca_iters    = mcfg['lca_iters'],
        eta          = mcfg['learning_rate'],
        track_metrics = False,
        return_vars  = ['inputs', 'acts', 'recons', 'recon_errors'],
    ).to(dtype=dtype, device=device)

    if using_ddp:
        dist.broadcast(lca.weights.data, src=0)

    lca_ddp   = lca
    lca_inner = lca

    if is_main:
        k = mcfg['kernel_size']
        print(f"LCAConv3D : {mcfg['features']} atoms | "
              f"kernel {k}³ | stride {mcfg['stride']} | λ={mcfg['lambda_']}\n")

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    epochs       = cfg['training']['epochs']
    anneal_every = cfg['training']['lambda_anneal_every']
    anneal_step  = cfg['training']['lambda_anneal_step']
    anneal_start = cfg['training'].get('lambda_anneal_start', 0)
    anneal_stop  = cfg['training'].get('lambda_anneal_stop', epochs)

    all_l2, all_l1, all_energy = [], [], []

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        t0 = time.time()

        if epoch > 0 and anneal_start <= epoch < anneal_stop \
                and (epoch - anneal_start) % anneal_every == 0:
            lca_inner.lambda_ += anneal_step
            if is_main:
                print(f"  [anneal] λ → {lca_inner.lambda_:.3f}")

        ep_l2 = ep_l1 = ep_energy = ep_sparsity = ep_active = ep_rel_err = 0.0

        for patches in dataloader:
            patches = patches.to(dtype=dtype, device=device)  # (B, 1, P, P, P)
            inputs, code, recon, recon_error = lca_ddp(patches)

            lca_inner.update_weights(code, recon_error)

            if using_ddp:
                dist.all_reduce(lca_inner.weights.data, op=dist.ReduceOp.SUM)
                lca_inner.weights.data /= world_size
                lca_inner.normalize_weights()

            l1     = compute_l1_sparsity(code, lca_inner.lambda_).item()
            l2     = compute_l2_error(inputs, recon).item()
            energy = l2 + l1

            if is_main:
                all_l2.append(l2)
                all_l1.append(l1)
                all_energy.append(energy)

            # code shape: (B, features, D_out, H_out, W_out)
            n_total      = code.shape[1] * code.shape[2] * code.shape[3] * code.shape[4]
            ep_l2       += l2
            ep_l1       += l1
            ep_energy   += energy
            ep_sparsity += (code == 0).float().mean().item()
            ep_active   += (code != 0).float().sum(dim=(1, 2, 3, 4)).mean().item()
            ep_rel_err  += (
                (inputs - recon).pow(2).sum(dim=(1, 2, 3, 4)) /
                (inputs.pow(2).sum(dim=(1, 2, 3, 4)) + 1e-8)
            ).mean().item()

        nb         = len(dataloader)
        epoch_time = time.time() - t0

        if is_main:
            print(
                f"Epoch {epoch:02d} | {epoch_time:.1f}s ({epoch_time/nb:.2f}s/batch) | "
                f"Sparsity: {ep_sparsity/nb:.3f}  "
                f"Active: {ep_active/nb:.1f}/{n_total}  "
                f"Rel.err: {ep_rel_err/nb:.6f}  "
                f"L2: {ep_l2/nb:.4f}  L1: {ep_l1/nb:.4f}  "
                f"Energy: {ep_energy/nb:.4f}  λ={lca_inner.lambda_:.3f}"
            )
            torch.save(lca_inner.state_dict(), os.path.join(models_dir, 'lca_simmldc.pth'))

    # ------------------------------------------------------------------ #
    # Post-training output
    # ------------------------------------------------------------------ #
    if is_main:
        print("\nTraining complete.")

        sparsity = (code == 0).float().mean().item()
        active   = (code != 0).float().sum(dim=(1, 2, 3, 4)).mean().item()
        rel_err  = (
            (inputs - recon).pow(2).sum(dim=(1, 2, 3, 4)) /
            (inputs.pow(2).sum(dim=(1, 2, 3, 4)) + 1e-8)
        ).mean().item()
        k = mcfg['kernel_size']
        print(f"\n=== LCAConv3D ({mcfg['features']} atoms, kernel {k}³, λ={lca_inner.lambda_:.3f}) ===")
        print(f"  Sparsity (fraction zero):  {sparsity:.3f}")
        print(f"  Relative recon error:      {rel_err:.6f}")
        print(f"  Active coefficients/item:  {active:.1f} / {n_total}")
        print(f"  Energy (L2 + L1):          {all_energy[-1]:.4f}  "
              f"(first batch: {all_energy[0]:.4f})")

        # Plot 1 — loss curves
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for ax, values, label in zip(
            axes, [all_l2, all_l1, all_energy],
            ['L2 Recon Error', 'L1 Sparsity', 'Total Energy']
        ):
            ax.plot(values)
            ax.set_ylabel(label)
        axes[-1].set_xlabel('Batch (across all epochs)')
        axes[0].set_title('LCAConv3D — Training Metrics')
        plt.tight_layout()
        out = os.path.join(plots_dir, 'training_metrics.png')
        plt.savefig(out)
        plt.close()
        print(f"Saved {out}")

        # Plot 2 — dictionary atoms (mid-plane slice of each 3D kernel)
        # weights shape: (features, in_channels, kD, kH, kW)
        weights = lca_inner.get_weights().float().cpu().numpy()
        n_feat  = weights.shape[0]
        kD      = weights.shape[2]
        mid     = kD // 2                # show the central depth slice
        atoms   = weights[:, 0, mid, :, :]   # (features, kH, kW)

        cols = int(np.ceil(np.sqrt(n_feat)))
        rows = int(np.ceil(n_feat / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
        axes = np.array(axes).ravel()
        vmax = np.percentile(np.abs(atoms), 99)
        for i, ax in enumerate(axes):
            if i < n_feat:
                ax.imshow(atoms[i], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.axis('off')
        fig.suptitle(f'Dictionary atoms — mid-plane slice  ({n_feat} atoms, kernel {k}³)',
                     fontsize=10)
        plt.tight_layout()
        out = os.path.join(plots_dir, 'dictionary_atoms.png')
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Saved {out}")

        # Plot 3 — reconstruction examples (mid-plane slices of last batch)
        def mid_slice(t):
            # t: (C, D, H, W) tensor → 2D array at mid-depth
            arr = t.float().cpu().numpy()[0, t.shape[1] // 2]  # (H, W)
            return arr

        n = min(cfg['output']['n_images'], inputs.shape[0])
        fig, axes = plt.subplots(n, 3, figsize=(6, 2 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        axes[0, 0].set_title('Input (mid-slice)')
        axes[0, 1].set_title('Reconstruction')
        axes[0, 2].set_title('Recon Error')
        for i in range(n):
            inp_s  = mid_slice(recon[i] + recon_error[i])
            rec_s  = mid_slice(recon[i])
            err_s  = mid_slice(recon_error[i])
            vmax_i = np.percentile(np.abs(inp_s), 99)
            for ax, data in zip(axes[i], [inp_s, rec_s, err_s]):
                ax.imshow(data, cmap='RdBu_r', vmin=-vmax_i, vmax=vmax_i)
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

    if using_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
