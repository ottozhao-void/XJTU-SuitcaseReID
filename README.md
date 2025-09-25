<img src="docs/open_mmlab.png" align="right" width="30%">

# OpenUnReID

## Introduction
`OpenUnReID` is an open-source PyTorch-based codebase for both unsupervised learning (**USL**) and unsupervised domain adaptation (**UDA**) in the context of object re-ID tasks. It provides strong baselines and multiple state-of-the-art methods with highly refactored codes for both *pseudo-label-based* and *domain-translation-based* frameworks. It works with **Python >=3.5** and **PyTorch >=1.1**.

This fork additionally ships suitcase-centric supervised baselines for the Multi-View Baggage (MVB) dataset, optional Gaussian mask regression, and hardened distributed tooling used in our internal experiments.

We are actively updating this repo, and more methods will be supported soon. Contributions are welcome.

<p align="center">
    <img src="docs/openunreid.png" width="60%">
</p>

### Major features
- [x] Distributed training & testing with multiple GPUs and multiple machines.
- [x] High flexibility on various combinations of datasets, backbones, losses, etc.
- [x] GPU-based pseudo-label generation and k-reciprocal re-ranking with quite high speed.
- [x] Plug-and-play domain-specific BatchNorms for any backbones, sync BN is also supported.
- [x] Mixed precision training is supported, achieving higher efficiency.
- [x] A strong cluster baseline, providing high extensibility on designing new methods.
- [x] State-of-the-art methods and performances for both USL and UDA problems on object re-ID.
- [x] Suitcase-specific supervised pipeline with optional Gaussian mask regression and robust multi-GPU training utilities.

### Supported methods

Please refer to [MODEL_ZOO.md](docs/MODEL_ZOO.md) for trained models and download links, and please refer to [LEADERBOARD.md](docs/LEADERBOARD.md) for the leaderboard on public benchmarks.

| Method | Reference | USL | UDA |
| ------ | :---: | :-----: | :-----: |
| [UDA_TP](tools/UDA_TP) | [PR'20 (arXiv'18)](https://arxiv.org/abs/1807.11334) | ✓ | ✓ |
| [SPGAN](tools/SPGAN)  | [CVPR'18](https://arxiv.org/abs/1711.07027) | n/a  |  ✓ |  
| SSG | [ICCV'19](https://arxiv.org/abs/1811.10144) | ongoing  | ongoing  |  
| [strong_baseline](tools/strong_baseline) | Sec. 3.1 in [ICLR'20](https://openreview.net/pdf?id=rJlnOhVYPS) | ✓ | ✓ |
| [MMT](tools/MMT/) | [ICLR'20](https://openreview.net/pdf?id=rJlnOhVYPS) | ✓  | ✓  |  
| [SpCL](tools/SpCL/) | [NeurIPS'20](https://arxiv.org/abs/2006.02713) | ✓ |  ✓  |  
| SDA  | [arXiv'20](https://arxiv.org/abs/2003.06650) | n/a  |  ongoing |  
| [MVB Supervised](tools/MVB_supervised) | Internal (2025) | ✓ |  n/a |


## Updates

[2025-05-01] Added suitcase-focused supervised baselines, Gaussian mask regression option, and NCCL stability tooling. See [Suitcase ReID Extensions](#suitcase-reid-extensions).

[2020-08-02] Add the leaderboard on public benchmarks: [LEADERBOARD.md](docs/LEADERBOARD.md)

[2020-07-30] `OpenUnReID` v0.1.1 is released:
+ Support domain-translation-based frameworks, [CycleGAN](tools/CycleGAN) and [SPGAN](tools/SPGAN).
+ Support mixed precision training (`torch.cuda.amp` in PyTorch>=1.6), use it by adding `TRAIN.amp True` at the end of training commands.

[2020-07-01] `OpenUnReID` v0.1.0 is released.

## Suitcase ReID Extensions

The `tools/MVB_supervised*` directories provide a supervised pipeline tailored for the Multi-View Baggage (MVB) dataset plus an optional Gaussian mask regression variant. These additions keep the original OpenUnReID APIs intact while introducing suitcase-specific training defaults, logging, and distributed training fixes.

### MVB Supervised Baseline

- **Scripts & config**: `tools/MVB_supervised/strong_baseline_mvb.py`, `tools/MVB_supervised/mvb_config.yaml`
- **Backbone & heads**: ResNet-101 with GeM pooling, dropout, ImageNet pretraining, and standard softmax + triplet losses
- **Logging**: Integrated TensorBoard and Weights & Biases tracking through `MVBBaseRunner`
- **Dataset analysis**: `tools/MVB_supervised/analyze_mvb_dataset.py` generates charts and `analysis_results/mvb_dataset_analysis.md` summarising gallery/probe balance, materials, and view angles

#### Quick start

```bash
# Single GPU
python tools/MVB_supervised/strong_baseline_mvb.py \
        --config tools/MVB_supervised/mvb_config.yaml

# Multi GPU (torchrun)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=4 \
        tools/MVB_supervised/strong_baseline_mvb.py \
        --config tools/MVB_supervised/mvb_config.yaml \
        --launcher pytorch
```

Use `--resume-from /path/to/checkpoint.pth` to continue training. Training logs and checkpoints land under `logs/suitcase_supervised_mvb_*`. Run evaluation only via:

```bash
python tools/MVB_supervised/mvb_test.py \
        --config tools/MVB_supervised/mvb_config.yaml \
        --set MODEL.source_pretrained=/path/to/model_best.pth
```

### Gaussian Mask Regression Variant

The directory `tools/MVB_supervised_with_gaussian_mask` adds an attention-style mask decoder, mask-aware losses, and NCCL hardening:

- **Model changes**: `resnet101_with_mask` backbone with intermediate feature taps and a decoder that outputs sigmoid masks (see `openunreid/models/backbones/resnet.py`)
- **Losses**: `GaussianMaskRegressionLoss`, `FocalMaskLoss`, and `CombinedMaskLoss` in `openunreid/models/losses/mask_regression.py`
- **Builder updates**: `build_model_with_mask()` and `ReIDModelWithMask` for joint feature + mask outputs

Enable the mask head through config:

```yaml
MODEL:
    backbone: "resnet101_with_mask"
    mask_enabled: True
    mask_output_size: [256, 256]

TRAIN:
    LOSS:
        losses:
            cross_entropy: 1.0
            softmax_triplet: 1.0
            gaussian_mask_regression: 0.05
        mask_loss_type: mse
        sigma_ratio: 0.1
        mask_weight: 0.5
        mask_threshold: 0.1
```

Launch training with the hardened distributed settings and NCCL environment defaults pre-configured in the script:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=4 \
        tools/MVB_supervised_with_gaussian_mask/strong_baseline_mvb_with_gaussian.py \
        --config tools/MVB_supervised_with_gaussian_mask/mvb_config.yaml \
        --launcher pytorch
```

If bounding boxes are unavailable, the runner synthesises adaptive boxes but real annotations yield better masks. Expect higher memory use; adjust batch size accordingly. Validate the mask head via `tools/MVB_supervised_with_gaussian_mask/test_mask_model.py`.

### Distributed Training Fixes

Recurrent NCCL timeouts were mitigated via explicit environment guards, safer DDP defaults, and diagnostics. Highlights are summarised in `tools/MVB_supervised_with_gaussian_mask/NCCL_FIXES_SUMMARY.md`.

Sanity-check collective ops before long runs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
        --nproc_per_node=2 \
        tools/MVB_supervised_with_gaussian_mask/test_distributed_fix.py
```

The script mirrors previous ALLGATHER failures and reports NCCL state to help catch cluster issues early.

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Get Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of `OpenUnReID`.

## License

`OpenUnReID` is released under the [Apache 2.0 license](LICENSE).

## Citation

If you use this toolbox or models in your research, please consider cite:
```
@inproceedings{ge2020mutual,
  title={Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification},
  author={Yixiao Ge and Dapeng Chen and Hongsheng Li},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=rJlnOhVYPS}
}

@inproceedings{ge2020selfpaced,
    title={Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID},
    author={Yixiao Ge and Feng Zhu and Dapeng Chen and Rui Zhao and Hongsheng Li},
    booktitle={Advances in Neural Information Processing Systems},
    year={2020}
}
```
<!-- @misc{ge2020structured,
    title={Structured Domain Adaptation with Online Relation Regularization for Unsupervised Person Re-ID},
    author={Yixiao Ge and Feng Zhu and Rui Zhao and Hongsheng Li},
    year={2020},
    eprint={2003.06650},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
} -->


## Acknowledgement

Some parts of `openunreid` are learned from [torchreid](https://github.com/KaiyangZhou/deep-person-reid) and [fastreid](https://github.com/JDAI-CV/fast-reid). We would like to thank for their projects, which have boosted the research of supervised re-ID a lot. We hope that `OpenUnReID` could well benefit the research community of unsupervised re-ID by providing strong baselines and state-of-the-art methods.

## Contact

This project is developed by Yixiao Ge ([@yxgeee](https://github.com/yxgeee)), Tong Xiao ([@Cysu](https://github.com/Cysu)), Zhiwei Zhang ([@zwzhang121](https://github.com/zwzhang121)).
