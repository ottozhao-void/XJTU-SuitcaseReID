# 不同阈值设置示例配置

## 示例 1: 无阈值（原始行为）
```yaml
TRAIN:
  LOSS:
    losses: { "cross_entropy": 1., "softmax_triplet": 1., "gaussian_mask_regression": 0.05 }
    mask_loss_type: "mse"
    sigma_ratio: 0.1
    mask_weight: 0.5
    mask_threshold: 0.0  # 所有像素参与损失计算
```

## 示例 2: 低阈值（包含更多上下文）
```yaml
TRAIN:
  LOSS:
    losses: { "cross_entropy": 1., "softmax_triplet": 1., "gaussian_mask_regression": 0.05 }
    mask_loss_type: "mse"
    sigma_ratio: 0.1
    mask_weight: 0.5
    mask_threshold: 0.1  # 约14%的像素参与损失计算
```

## 示例 3: 中等阈值（平衡关注度）
```yaml
TRAIN:
  LOSS:
    losses: { "cross_entropy": 1., "softmax_triplet": 1., "gaussian_mask_regression": 0.05 }
    mask_loss_type: "mse"
    sigma_ratio: 0.1
    mask_weight: 0.5
    mask_threshold: 0.3  # 约7.5%的像素参与损失计算
```

## 示例 4: 高阈值（聚焦核心区域）
```yaml
TRAIN:
  LOSS:
    losses: { "cross_entropy": 1., "softmax_triplet": 1., "gaussian_mask_regression": 0.05 }
    mask_loss_type: "mse"
    sigma_ratio: 0.1
    mask_weight: 0.5
    mask_threshold: 0.5  # 约4.3%的像素参与损失计算
```

## 使用场景建议

### 训练初期（Epochs 1-30）
- 使用 `mask_threshold: 0.1` 
- 包含更多上下文信息，帮助模型学习基础特征

### 训练中期（Epochs 31-70）  
- 使用 `mask_threshold: 0.3`
- 平衡核心区域和上下文信息

### 训练后期（Epochs 71-100）
- 使用 `mask_threshold: 0.5`
- 聚焦最重要的核心区域，精细化训练

### 数据质量差/噪音多的情况
- 使用 `mask_threshold: 0.4-0.7`
- 忽略边缘噪音，专注高质量区域

### 数据质量好/细节重要的情况
- 使用 `mask_threshold: 0.1-0.2`
- 利用更多细节信息进行训练