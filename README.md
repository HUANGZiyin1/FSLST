# FSLST

PyTorch implementation of **Frame Similarity-Based Screen Content Video Quality Enhancement via Adaptive Long Short-Term Fusion**, published at **IEEE VCIP 2024**.

[Paper / DOI](https://doi.org/10.1109/VCIP63160.2024.10849791)

FSLST selects three short-term frames according to their similarity to the target frame, extracts short-term features with multi-scale residual blocks and long-term features from five frames with 3D residual blocks, and combines the two streams using local spatial and global channel attention. The model predicts a residual for the target luminance frame.

## Files

- `SNFS_A.py`: FSLST model; the original class name `HAWT` is retained.
- `dataset/HAWT.py`: training/test data loading and similarity-based neighbor frame selection.
- `train.py`: training entry point.
- `test_fyp.py`: PSNR evaluation entry point.
- `SNFS_A_QP27.yml`, `SNFS_A_QP27_test.yml`: example QP27 configurations.
- `create_lmdb_mfqev2.py`: original LMDB preparation script.
- `common.py`, `octconv.py`, `utils/`: supporting modules retained from the original implementation.
- `exp/SCC_LD_SNFS_A_QP27_enlarg300x/`: QP27 checkpoint at iteration 300000 and the original `log.log` / `log_test.log`.

## Environment

The training and evaluation scripts require Python, CUDA-enabled PyTorch, NumPy, SciPy, PyYAML, tqdm, LMDB, OpenCV and scikit-image. Install a PyTorch build compatible with your GPU environment, then install the listed Python dependencies:

```bash
python -m pip install -r requirements.txt
```

No custom CUDA extension is needed for the active FSLST model. Exact dependency versions and GPU execution have not been validated for this release.

## Data and training

Replace `/path/to/dataset/` in the YAML files with your **absolute** dataset path. Training consumes paired GT/LQ LMDB databases; validation and testing read YUV videos. Keep GT and compressed videos paired in the same sorted order. The YUV test loader expects each compressed filename to be `rec` followed by the corresponding raw filename. Datasets are not included.

The original LMDB builder is invoked as follows:

```bash
python create_lmdb_mfqev2.py --opt_path SNFS_A_QP27.yml
```

Before using that builder, check its retained format and filename assumptions: it defaults to 8-bit YUV420p, whereas the test reader defaults to YUV444p; it limits each training video to 300 frames and groups frames in windows of five. Its filename parsing and automatic LMDB map-size calculation also need to match your files. The builder is included unchanged; the command is not a verified end-to-end data preparation recipe for arbitrary YUV datasets. Existing compatible LMDB databases may be specified directly in the configuration.

With prepared LMDB databases and validation data:

```bash
python train.py --opt_path SNFS_A_QP27.yml
```

The supplied YAML is an example adapted for QP27, rather than an exact reconstruction of the historical training configuration. In particular, it uses `batch_size_per_gpu: 4`; the archived training log records batch size 32 on one GPU. The training example writes to a separate experiment directory.

## Evaluation

Set the test GT/LQ paths in `SNFS_A_QP27_test.yml`, then run:

```bash
python test_fyp.py --opt_path SNFS_A_QP27_test.yml
```

This loads `exp/SCC_LD_SNFS_A_QP27_enlarg300x/ckp_300000.pt` and computes luminance PSNR. It writes `log_test.log` in the checkpoint directory, overwriting that file; copy the original log before running a new evaluation.

The bundled logs are preserved as historical records. The archived test log contains values near 0.99 despite its `PSNR` / `dB` labels; its actual historical metric cannot be established from the current PSNR script alone. Those values should not be interpreted as PSNR in dB.

## Citation

```bibtex
@inproceedings{huang2024fslst,
  author    = {Ziyin Huang and Yui-Lam Chan and Ngai-Wing Kwong and
               Sik-Ho Tsang and Kin-Man Lam and Wing-Kuen Ling},
  title     = {Frame Similarity-Based Screen Content Video Quality Enhancement
               via Adaptive Long Short-Term Fusion},
  booktitle = {2024 IEEE International Conference on Visual Communications
               and Image Processing (VCIP)},
  year      = {2024},
  pages     = {1--5},
  doi       = {10.1109/VCIP63160.2024.10849791}
}
```

## Acknowledgments

The training, data and utility infrastructure includes code adapted from [STDF](https://github.com/ryanxingql/stdf-pytorch). Original third-party copyright and license notices are retained in their source files.
