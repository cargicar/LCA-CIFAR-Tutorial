# Single Simulation Single Snapshop
We take a single shop from a single simulation and do a few forward passes to optimize the Phi
### scripts used 
- lca_sim_mldc_SingleSnaptshot.py
- config_simmldc.yaml
### Key Design Decisions lca_sim_mldc_SingleSnaptshot.py
- **Patch extraction**: Each `__getitem__` draws a new random 32³ crop from the single 128³ volume, giving effectively unlimited augmentation. `n_patches=2000` sets the virtual epoch size.
- **Normalization**: Volume is z-scored on load (zero mean, unit variance) so LCA's internal normalization works correctly with scalar pressure data.
- **`in_channels=1`**: Pressure is a scalar field, unlike the 3-channel RGB images in the CIFAR version.
- **Dictionary atom plots**: 3D kernels are visualized as their central depth slice (`kD//2`), the standard way to display 3D filters.
- **`patch_size=32`, `stride=4`**: Output code is `(32/4)³ = 8³ = 512` positions/patch with 64 atoms → 32,768 total code values per patch.
- **LCAConv3D**: Same API as LCAConv2D but takes `(B, C, D, H, W)` input. Multi-GPU sync via manual `all_reduce` after each Hebbian update, identical to the CIFAR pipeline.
- **Full-volume reconstruction**: The inference script tiles the 128³ volume into non-overlapping 32³ patches (4×4×4 = 64 tiles), runs LCA on each batch, then undoes LCA's per-patch normalization (`recon_global = recon_lca × patch_std + patch_mean`) before stitching back to 128³. Tile-boundary discontinuities are visible as a faint grid artifact in the reconstruction.
- **Compression metrics**: Sparse COO storage assigns each non-zero a float32 value (4 bytes) + a flat index into the full code tensor of `features × (D/P) × (H/P) × (W/P) × (P/stride)³` positions. With 2,097,152 addressable positions, each index costs 22 bits (3 bytes), giving 7 bytes per non-zero. Standard quality/rate metrics reported: PSNR (dB), RMSE, bits-per-voxel (BPV, baseline = 32), and compression ratio vs raw float32.

### Results
**Experiment:** `simmldc_2026-06-11_13-28-15`  
**Config:** 64 atoms, kernel 7³, stride 4, patch 32³, λ warmup 0.05→0.55 (ep15–40, hold ep40–54), 2000 patches/epoch, t=15

| Metric | Value |
|---|---|
| Sparsity (fraction zero) | 88.7% |
| Relative recon error | 0.98% |
| Active coefficients/patch | 3,718 / 32,768 |
| L2 recon error | 160.13 |
| L1 sparsity cost | 1,092.46 |
| Final λ | 0.55 |

**Observations:**
- Reconstruction fidelity is excellent at 0.98% relative error — significantly better than the CIFAR case, likely because a single pressure snapshot is a much more homogeneous and structured field than natural images.
- Dictionary atoms (mid-plane slices of 7³ kernels) show diverse gradient and edge-like filters oriented along all three spatial directions, consistent with the smooth, slowly-varying pressure structures in isotropic turbulence.
- Recon error column in the reconstruction plots is nearly blank, confirming the sparse code is capturing most of the variance.



# Single Simulation Multiple Snapshops

# Multiple Simulations 

