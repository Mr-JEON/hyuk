# PyTorch Deep Learning Study

PyTorch를 활용한 딥러닝 학습 저장소입니다. 기초 개념부터 최신 아키텍처까지 Jupyter Notebook으로 구현하고 실습합니다.

## 구성

### `torch/` — PyTorch 기초
- 선형 회귀, 이진 분류, 다중 분류
- CNN, Custom Dataset, 데이터 증강(Augmentation)
- MNIST, CIFAR-10, STL-10, COVID X-ray 데이터셋 실습
- TensorBoard 시각화

### `Legend13_code/` — CNN 아키텍처
| 모델 | 논문 |
|------|------|
| ZFNet | Visualizing and Understanding Convolutional Networks |
| VGGNet | Very Deep Convolutional Networks |
| Inception v1/v3/v4 | GoogLeNet, Inception 시리즈 |
| ResNet / WideResNet / ResNeXt | Residual Networks 계열 |
| DenseNet | Densely Connected Networks |
| SENet | Squeeze-and-Excitation Networks |
| MobileNet V1/V2/V3 | Efficient Mobile Architectures |
| EfficientNet | EfficientNet: Rethinking Model Scaling |
| PointNet | PointNet: 3D Point Cloud |

### `ATT_code/` — Attention 기반 아키텍처
| 모델 | 설명 |
|------|------|
| ViT | Vision Transformer |
| Swin Transformer | Shifted Window Transformer |
| ConvNeXt | A ConvNet for the 2020s |
| BERT | Bidirectional Encoder Representations |
| GPT-2 | Generative Pre-trained Transformer 2 |

## 실행 환경

```bash
pip install torch torchvision matplotlib numpy tensorboard
jupyter notebook
```

## 유틸리티 (`torch/multiclass_functions*.py`)

다중 분류 노트북에서 공통으로 사용하는 함수 모음입니다.

```python
Train(model, train_DL, criterion, optimizer, EPOCH)  # 학습 루프
Test(model, test_DL)                                  # 정확도 평가
Test_plot(model, test_DL)                             # 예측 결과 시각화
count_params(model)                                   # 파라미터 수 계산
get_conf(model, test_DL)                              # 혼동 행렬 계산
plot_confusion_matrix(confusion, classes)             # 혼동 행렬 시각화
```
