# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal PyTorch deep learning study repository. Content is organized as Jupyter notebooks with supporting Python utility files. No build/test/lint pipeline — notebooks are run interactively via Jupyter.

## Running notebooks

```bash
jupyter notebook        # launch Jupyter in browser
jupyter lab             # alternative: JupyterLab
```

To run a single notebook non-interactively:
```bash
jupyter nbconvert --to notebook --execute <notebook>.ipynb
```

## Repository structure

- `torch/` — foundational PyTorch notebooks (linear regression, binary/multiclass classification, CNN, custom datasets, augmentation, TensorBoard) and reusable helper modules
- `Legend13_code/Legend13/` — CNN architecture implementations (ZFNet, VGGNet, Inception v1/v3/v4, ResNet, WideResNet, ResNeXt, DenseNet, SENet, MobileNet V1/V2/V3, EfficientNet, PointNet)
- `ATT_code/ATT/` — attention-based architecture implementations (ViT, Swin Transformer, ConvNeXt, BERT, GPT-2)

## Helper modules (`torch/multiclass_functions*.py`)

Three versions of shared training utilities used across multiclass notebooks:

- `Train(model, train_DL, criterion, optimizer, EPOCH)` — training loop, returns per-epoch loss history
- `Test(model, test_DL)` — evaluates accuracy on test dataloader
- `Test_plot(model, test_DL)` — plots 6 sample predictions with color-coded correct/wrong labels
- `count_params(model)` — counts trainable parameters
- `get_conf(model, test_DL)` — computes confusion matrix as numpy array
- `plot_confusion_matrix(confusion, classes=None)` — visualizes confusion matrix

Device selection is automatic: `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`.

## Key patterns

- Models are defined with `nn.Sequential` or custom `nn.Module` subclasses
- Data loading uses `torchvision.datasets` (MNIST, CIFAR-10, STL-10) and custom `Dataset` classes
- Loss functions: `BCELoss`/`BCEWithLogitsLoss` for binary classification, `CrossEntropyLoss` for multiclass
- Saved model weights go to `torch/results/` (e.g., `MLP.pt`)

## Commits to Github
- 수정 파일 커밋/푸쉬 시, 중요한 변경사항이 있을 경우는 커밋 메시지 맨 앞에 별표를 붙임
