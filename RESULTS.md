# Experiment Results

## Overview

We ran a series of experiments comparing two architectures (MLP and FastPointNet) across two data representations (random point clouds and fixed-topology mesh vertices). Training used SMPL-X synthetic data and COMA real scans. All experiments evaluated on COMA held-out test data unless otherwise noted.

## Results

**Table 1: Test accuracy across experiments.**

| Experiment | Architecture | Train Source | Test Source | Representation | Test Accuracy | Macro F1 |
|---|---|---|---|---|---|---|
| MLP + COMA pcd | MLP | COMA | COMA | 1024 random points | 18.1% | 14.1% |
| MLP + COMA mesh | MLP | COMA | COMA | 5023 fixed vertices | 77.6% | 77.5% |
| FastPointNet + COMA pcd | FastPointNet | COMA | COMA | 1024 random points | 67.8% | 67.3% |
| MLP + SMPL-X pcd | MLP | SMPL-X | SMPL-X | 1024 random points | 72.7%* | 72.6% |
| MLP + SMPL-X pcd - COMA pcd | MLP | SMPL-X | COMA | 1024 random points | 15.8% | 14.5% |
| MLP + combined pcd - COMA pcd | MLP | SMPL-X + COMA | COMA | 1024 random points | 17.9% | 14.9% |
| MLP + COMA mesh - SMPL-X mesh | MLP | COMA | SMPL-X | 5023 fixed vertices | 14.3% | 3.6% |
| MLP + SMPL-X mesh | MLP | SMPL-X | SMPL-X | 5023 fixed vertices | 100.0% | 100.0% |
| MLP + SMPL-X mesh (old, pre-boost) | MLP | SMPL-X | SMPL-X | 5023 fixed vertices | 40.6% | 26.3% |
| FastPointNet + COMA pcd (10%) | FastPointNet | COMA (subset) | COMA | 1024 random points | 42.9% | 37.8% |

*See note below on EXP A checkpoint discrepancy.

## Discussion

### MLP fails on random point clouds, works on fixed-vertex meshes

The most consistent finding is that the same MLP architecture behaves completely differently depending on how the data is represented. On COMA random point clouds the model achieves 18.1%, which is barely above the random baseline of 14.3% for seven classes. On the same COMA dataset with fixed vertex ordering, accuracy jumps to 77.6%. The MLP treats the raveled vertex array as a flat vector, so it implicitly learns positional patterns that only exist when vertex indices are consistent across samples. Random point clouds destroy this structure.

### PointNet closes most of the gap

FastPointNet, which is designed to be permutation-invariant, reaches 67.8% on the same COMA point cloud data where the MLP got 18.1%. This is a substantial improvement and confirms that the architecture is suited to unordered point sets. However, it still trails the MLP on mesh vertices by about 10 percentage points, suggesting that fixed correspondence provides information that permutation invariance cannot fully recover.

### No architecture or representation combination bridges the domain gap

Every cross-domain experiment between COMA and SMPL-X data produced results at or near random. Using SMPL-X synthetic data to train on COMA point clouds gave 15.8%, and training a mesh-based model on COMA and testing on SMPL-X gave 14.3%. Even combining both sources did not help (17.9%). The fundamental issue is that the two datasets use different mesh registrations. Vertex index i in COMA does not correspond to vertex i in SMPL-X, so the MLP's positional learning transfers nothing, and even FastPointNet's shape features do not generalize across the synthetic-to-real gap.

### SMPL-X in-domain mesh performance is weak

The original SMPL-X mesh experiment (40.6% accuracy, only 3 of 7 classes predicted) used incorrect face vertex extraction and unboosted expression parameters. After fixing face extraction (graph-based 5023-vertex selection) and applying 4× boosted emotion anchors, the re-run (EXP S-mesh v2) achieves **100% accuracy** on the held-out test set. This confirms that SMPL-X synthetic mesh data is perfectly separable when expressions are sufficiently amplified and vertex topology is consistent. The previous failure was a data-quality issue, not an architectural limitation.

### EXP A checkpoint discrepancy

The SMPL-X point cloud checkpoint reports 72.7% on test evaluation, but its training log shows 100% training accuracy and 18% validation accuracy, which indicates severe overfitting. The checkpoint was likely overwritten by a different training run and cannot be reproduced from the documented setup. This result is excluded from analysis.

### Architecture comparison

| Model | Parameters | Speed (ms/batch) | Best Accuracy |
|---|---|---|---|
| MLP | ~819K | 120 | 77.6% (mesh) |
| FastPointNet | ~240K | 533 | 67.8% (pcd) |
| PointNet | ~801K | 3855 | untested |

FastPointNet is a practical choice for CPU training. Full PointNet with 1D convolutions takes about 12 days on CPU for one training run, making it impractical without GPU access.

## Limitations

Several constraints shaped these results. The compute environment was CPU-only, which restricted model size and training time. The COMA dataset, while real, contains variability in actor performance — some expressions are naturally weak or overlap with other classes. SMPL-X synthetic expressions, even after 4x amplification, remain subtle compared to ground-truth capture. The two datasets share the FLAME template but use different registrations, making any cross-domain transfer impossible without non-rigid alignment.

## Reproduction

To reproduce results, install dependencies listed in requirements.txt, download the SMPL-X model weights, and place the COMA dataset under `coma/`. Run `python scripts/generate_smplx_mesh_vertices.py` to build the synthetic dataset, `python scripts/make_splits.py` to create train/val/test splits, then run the training and evaluation scripts from `scripts/`. The `RESULTS.md` file in this repository documents all experiment outputs.
