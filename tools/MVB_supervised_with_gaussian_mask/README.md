# ResNet101 with Gaussian Mask Regression for Suitcase ReID

这个项目扩展了原始的OpenUnReID项目，添加了ResNet101骨干网络的掩码生成功能，用于生成高斯掩码并与真值掩码进行回归损失计算。

## 新增功能

### 1. ResNet101WithMask 骨干网络
- **位置**: `openunreid/models/backbones/resnet.py`
- **功能**: 
  - 基于ResNet101架构
  - 从layer3（第三层）提取中间特征
  - 包含解码器网络生成注意力掩码
  - 支持ImageNet预训练权重加载

### 2. 掩码解码器 (MaskDecoder)
- **功能**:
  - 将ResNet layer3的特征（1024通道）上采样到目标掩码尺寸
  - 使用卷积+批量归一化+ReLU的层次结构
  - 最终输出单通道掩码（经过sigmoid激活）

### 3. 高斯掩码回归损失函数
- **位置**: `openunreid/models/losses/mask_regression.py`
- **包含类**:
  - `GaussianMaskRegressionLoss`: 基础高斯掩码回归损失
  - `FocalMaskLoss`: 处理不平衡区域的焦点损失
  - `CombinedMaskLoss`: 结合回归和焦点损失的混合损失

### 4. 扩展的ReID模型
- **位置**: `openunreid/models/builder.py`
- **新增类**: `ReIDModelWithMask`
- **新增函数**: `build_model_with_mask()`

## 使用方法

### 1. 配置文件设置
```yaml
MODEL:
  backbone: "resnet101_with_mask"  # 使用带掩码的ResNet101
  mask_enabled: True
  mask_output_size: [256, 256]

TRAIN:
  LOSS:
    losses: { 
      "cross_entropy": 1., 
      "softmax_triplet": 1., 
      "gaussian_mask_regression": 0.5 
    }
    # 掩码损失设置
    mask_loss_type: "mse"  # 选项: mse, l1, smooth_l1
    sigma_ratio: 0.1       # 高斯sigma作为边界框大小的比例
    mask_weight: 0.5       # 掩码回归损失的权重
```

### 2. 训练脚本使用
```bash
# 单GPU训练
python tools/MVB_supervised_with_gaussian_mask/strong_baseline_mvb_with_gaussian.py \
    --config tools/MVB_supervised_with_gaussian_mask/mvb_config.yaml

# 多GPU训练
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nproc_per_node=4 \
    tools/MVB_supervised_with_gaussian_mask/strong_baseline_mvb_with_gaussian.py \
    --config tools/MVB_supervised_with_gaussian_mask/mvb_config.yaml \
    --launcher pytorch
```

### 3. 代码示例
```python
# 创建带掩码的模型
from openunreid.models import build_model_with_mask
model = build_model_with_mask(cfg, num_classes)

# 前向传播
results = model(inputs)
features = results["feat"]      # ReID特征
masks = results["mask"]         # 生成的掩码 (如果训练模式)
probs = results["prob"]         # 分类概率 (如果训练模式)

# 掩码损失计算
from openunreid.models.losses import GaussianMaskRegressionLoss
mask_loss_fn = GaussianMaskRegressionLoss()
mask_loss = mask_loss_fn(masks, target_bboxes=bboxes)
```

## 文件结构

```
tools/MVB_supervised_with_gaussian_mask/
├── mvb_config.yaml                          # 配置文件
├── strong_baseline_mvb_with_gaussian.py     # 训练脚本
├── test_mask_model.py                       # 测试脚本
└── README.md                                # 本文件

openunreid/models/
├── backbones/
│   └── resnet.py                           # 扩展的ResNet (新增ResNetWithMask)
├── losses/
│   ├── __init__.py                         # 更新的损失构建器
│   └── mask_regression.py                  # 新增掩码回归损失
└── builder.py                              # 扩展的模型构建器
```

## 关键特性

### 1. 中间层特征提取
- 从ResNet layer3提取特征用于掩码生成
- 保持layer4特征用于ReID任务

### 2. 高斯掩码生成
- 支持从边界框自动生成高斯掩码
- 可配置的sigma比例参数

### 3. 多损失函数支持
- MSE、L1、Smooth L1回归损失
- 焦点损失处理前景/背景不平衡
- 组合损失函数

### 4. 向后兼容
- 与原始OpenUnReID框架完全兼容
- 支持所有现有的ReID损失函数
- 可选择性启用掩码功能

## 测试

运行测试脚本验证功能：
```bash
cd /data1/zhaofanghan/SuitcaseReID/OpenUnReID
PYTHONPATH=/data1/zhaofanghan/SuitcaseReID/OpenUnReID python tools/MVB_supervised_with_gaussian_mask/test_mask_model.py
```

## 注意事项

1. **边界框标注**: 掩码损失计算需要边界框标注。如果数据集中没有，训练脚本会使用中心区域的虚拟边界框。

2. **内存使用**: 掩码生成会增加内存使用量，建议适当调整batch size。

3. **损失权重**: 掩码损失的权重需要根据具体任务调整，建议从0.1-1.0范围内尝试。

4. **预训练权重**: 只有ResNet骨干网络会加载ImageNet预训练权重，掩码解码器从随机初始化开始训练。

## 实现细节

- **特征图尺寸**: layer3输出 1024×H/8×W/8，layer4输出 2048×H/8×W/8
- **掩码尺寸**: 最终掩码尺寸可配置，默认256×256
- **上采样方法**: 使用双线性插值进行特征图上采样
- **激活函数**: 掩码输出使用sigmoid激活，确保值在[0,1]范围内