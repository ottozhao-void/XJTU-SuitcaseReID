#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高斯掩码可视化脚本
Visualization script for Gaussian masks using heatmaps
"""

import sys
import os

# 添加正确的路径
sys.path.insert(0, "/data1/zhaofanghan/SuitcaseReID/OpenUnReID")
os.chdir("/data1/zhaofanghan/SuitcaseReID/OpenUnReID")

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 直接导入mask_regression模块中的类
try:
    from openunreid.models.losses.mask_regression import GaussianMaskRegressionLoss
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试直接加载mask_regression.py文件...")
    
    # 如果模块导入失败，直接加载文件
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mask_regression", 
        "/data1/zhaofanghan/SuitcaseReID/OpenUnReID/openunreid/models/losses/mask_regression.py"
    )
    mask_regression = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mask_regression)
    GaussianMaskRegressionLoss = mask_regression.GaussianMaskRegressionLoss


def visualize_gaussian_mask_examples():
    """
    可视化不同参数下的高斯掩码示例
    """
    # 创建高斯掩码损失函数实例
    mask_loss = GaussianMaskRegressionLoss(sigma_ratio=0.1)
    
    # 设置掩码尺寸
    height, width = 128, 128
    
    # 定义不同的测试场景
    test_cases = [
        {
            'name': '中心位置 (0.5, 0.5)',
            'center_x': 0.5, 'center_y': 0.5,
            'sigma_x': 0.1, 'sigma_y': 0.1
        },
        {
            'name': '偏左上 (0.3, 0.3)',
            'center_x': 0.3, 'center_y': 0.3,
            'sigma_x': 0.15, 'sigma_y': 0.15
        },
        {
            'name': '偏右下 (0.7, 0.7)',
            'center_x': 0.7, 'center_y': 0.7,
            'sigma_x': 0.08, 'sigma_y': 0.08
        },
        {
            'name': '椭圆形 - 水平拉伸',
            'center_x': 0.5, 'center_y': 0.5,
            'sigma_x': 0.2, 'sigma_y': 0.1
        },
        {
            'name': '椭圆形 - 垂直拉伸',
            'center_x': 0.5, 'center_y': 0.5,
            'sigma_x': 0.1, 'sigma_y': 0.2
        },
        {
            'name': '小高斯核',
            'center_x': 0.5, 'center_y': 0.5,
            'sigma_x': 0.05, 'sigma_y': 0.05
        }
    ]
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('高斯掩码可视化 - 不同参数效果对比', fontsize=16, fontweight='bold')
    
    for i, test_case in enumerate(test_cases):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # 生成高斯掩码
        mask = mask_loss.generate_gaussian_mask(
            height, width,
            test_case['center_x'], test_case['center_y'],
            test_case['sigma_x'], test_case['sigma_y']
        )
        
        # 转换为numpy数组用于可视化
        mask_np = mask.detach().cpu().numpy()
        
        # 创建热力图
        im = ax.imshow(mask_np, cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('高斯值', rotation=270, labelpad=15)
        
        # 设置标题和标签
        ax.set_title(f'{test_case["name"]}\n'
                    f'center=({test_case["center_x"]:.1f}, {test_case["center_y"]:.1f}), '
                    f'σ=({test_case["sigma_x"]:.2f}, {test_case["sigma_y"]:.2f})', 
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('X坐标 (像素)')
        ax.set_ylabel('Y坐标 (像素)')
        
        # 添加中心点标记
        center_x_pixel = test_case['center_x'] * width
        center_y_pixel = test_case['center_y'] * height
        ax.plot(center_x_pixel, center_y_pixel, 'w+', markersize=15, markeredgewidth=3)
        ax.plot(center_x_pixel, center_y_pixel, 'k+', markersize=12, markeredgewidth=2)
        
        # 显示最大值和中心值
        max_val = mask_np.max()
        center_val = mask_np[int(center_y_pixel), int(center_x_pixel)]
        ax.text(0.02, 0.98, f'最大值: {max_val:.3f}\n中心值: {center_val:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised_with_gaussian_mask/gaussian_masks_overview.png', 
                dpi=300, bbox_inches='tight')
    print("✅ 高斯掩码概览图已保存: gaussian_masks_overview.png")
    plt.show()


def visualize_bbox_based_masks():
    """
    可视化基于边界框的高斯掩码生成
    """
    mask_loss = GaussianMaskRegressionLoss(sigma_ratio=0.1)
    
    # 定义不同的边界框场景 (模拟行李箱检测结果)
    bbox_cases = [
        {
            'name': '中心大行李箱',
            'bbox': [0.25, 0.25, 0.75, 0.75]  # [x1, y1, x2, y2]
        },
        {
            'name': '左侧小行李箱',
            'bbox': [0.1, 0.3, 0.4, 0.7]
        },
        {
            'name': '右上角行李箱',
            'bbox': [0.6, 0.1, 0.9, 0.4]
        },
        {
            'name': '细长行李箱',
            'bbox': [0.2, 0.4, 0.8, 0.6]
        }
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('基于边界框的高斯掩码生成', fontsize=16, fontweight='bold')
    
    for i, case in enumerate(bbox_cases):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # 转换边界框为张量
        bbox_tensor = torch.tensor([case['bbox']], dtype=torch.float32)
        
        # 生成高斯掩码
        masks = mask_loss.create_gaussian_masks_from_bbox(bbox_tensor, mask_size=(128, 128))
        mask_np = masks[0, 0].detach().cpu().numpy()
        
        # 创建热力图
        im = ax.imshow(mask_np, cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('高斯值', rotation=270, labelpad=15)
        
        # 绘制边界框
        x1, y1, x2, y2 = case['bbox']
        x1_pixel, y1_pixel = x1 * 128, y1 * 128
        x2_pixel, y2_pixel = x2 * 128, y2 * 128
        
        # 绘制边界框矩形
        from matplotlib.patches import Rectangle
        rect = Rectangle((x1_pixel, y1_pixel), x2_pixel - x1_pixel, y2_pixel - y1_pixel,
                        linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # 标记中心点
        center_x_pixel = (x1_pixel + x2_pixel) / 2
        center_y_pixel = (y1_pixel + y2_pixel) / 2
        ax.plot(center_x_pixel, center_y_pixel, 'w+', markersize=15, markeredgewidth=3)
        ax.plot(center_x_pixel, center_y_pixel, 'k+', markersize=12, markeredgewidth=2)
        
        # 设置标题和标签
        ax.set_title(f'{case["name"]}\nBBox: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('X坐标 (像素)')
        ax.set_ylabel('Y坐标 (像素)')
        
        # 显示统计信息
        max_val = mask_np.max()
        mean_val = mask_np.mean()
        ax.text(0.02, 0.98, f'最大值: {max_val:.3f}\n平均值: {mean_val:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised_with_gaussian_mask/bbox_gaussian_masks.png', 
                dpi=300, bbox_inches='tight')
    print("✅ 边界框高斯掩码图已保存: bbox_gaussian_masks.png")
    plt.show()


def visualize_gaussian_cross_section():
    """
    可视化高斯掩码的横截面和纵截面
    """
    mask_loss = GaussianMaskRegressionLoss(sigma_ratio=0.1)
    
    # 生成一个标准的高斯掩码
    height, width = 128, 128
    center_x, center_y = 0.5, 0.5
    sigma_x, sigma_y = 0.1, 0.15  # 不同的sigma值展示椭圆效果
    
    mask = mask_loss.generate_gaussian_mask(height, width, center_x, center_y, sigma_x, sigma_y)
    mask_np = mask.detach().cpu().numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('高斯掩码详细分析 - 截面图和3D视图', fontsize=16, fontweight='bold')
    
    # 1. 热力图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(mask_np, cmap='hot', interpolation='bilinear', vmin=0, vmax=1)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title('2D热力图视图', fontweight='bold')
    ax1.set_xlabel('X坐标 (像素)')
    ax1.set_ylabel('Y坐标 (像素)')
    
    # 添加截面线
    center_x_pixel = int(center_x * width)
    center_y_pixel = int(center_y * height)
    ax1.axhline(y=center_y_pixel, color='cyan', linestyle='-', linewidth=2, alpha=0.7, label='水平截面')
    ax1.axvline(x=center_x_pixel, color='lime', linestyle='-', linewidth=2, alpha=0.7, label='垂直截面')
    ax1.legend()
    
    # 2. 水平截面
    ax2 = axes[0, 1]
    horizontal_section = mask_np[center_y_pixel, :]
    x_coords = np.linspace(0, 1, width)
    ax2.plot(x_coords, horizontal_section, 'cyan', linewidth=3, label='水平截面')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='最大值=1')
    ax2.axvline(x=center_x, color='red', linestyle='--', alpha=0.5, label=f'中心x={center_x}')
    ax2.set_title('水平截面 (通过中心点)', fontweight='bold')
    ax2.set_xlabel('X坐标 (归一化)')
    ax2.set_ylabel('高斯值')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1.1)
    
    # 3. 垂直截面
    ax3 = axes[1, 0]
    vertical_section = mask_np[:, center_x_pixel]
    y_coords = np.linspace(0, 1, height)
    ax3.plot(y_coords, vertical_section, 'lime', linewidth=3, label='垂直截面')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='最大值=1')
    ax3.axvline(x=center_y, color='red', linestyle='--', alpha=0.5, label=f'中心y={center_y}')
    ax3.set_title('垂直截面 (通过中心点)', fontweight='bold')
    ax3.set_xlabel('Y坐标 (归一化)')
    ax3.set_ylabel('高斯值')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    
    # 4. 3D表面图
    ax4 = axes[1, 1]
    ax4.remove()  # 移除2D axes
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    X = np.linspace(0, 1, width)
    Y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(X, Y)
    
    surface = ax4.plot_surface(X, Y, mask_np, cmap='hot', alpha=0.8, 
                              linewidth=0, antialiased=True)
    ax4.set_title('3D表面视图', fontweight='bold')
    ax4.set_xlabel('X坐标 (归一化)')
    ax4.set_ylabel('Y坐标 (归一化)')
    ax4.set_zlabel('高斯值')
    
    # 添加颜色条到3D图
    fig.colorbar(surface, ax=ax4, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('/data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised_with_gaussian_mask/gaussian_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✅ 高斯掩码分析图已保存: gaussian_analysis.png")
    plt.show()


def print_gaussian_statistics():
    """
    打印高斯掩码的数值统计信息
    """
    print("\n" + "="*60)
    print("高斯掩码数值统计分析")
    print("="*60)
    
    mask_loss = GaussianMaskRegressionLoss(sigma_ratio=0.1)
    
    # 生成标准高斯掩码
    height, width = 64, 64  # 使用较小尺寸便于分析
    center_x, center_y = 0.5, 0.5
    sigma_x, sigma_y = 0.1, 0.1
    
    mask = mask_loss.generate_gaussian_mask(height, width, center_x, center_y, sigma_x, sigma_y)
    mask_np = mask.detach().cpu().numpy()
    
    print(f"掩码尺寸: {height} x {width}")
    print(f"中心位置: ({center_x}, {center_y})")
    print(f"Sigma值: σx={sigma_x}, σy={sigma_y}")
    print("-" * 40)
    
    # 中心点值
    center_pixel_x = int(center_x * width)
    center_pixel_y = int(center_y * height)
    center_value = mask_np[center_pixel_y, center_pixel_x]
    
    print(f"中心点像素坐标: ({center_pixel_x}, {center_pixel_y})")
    print(f"中心点高斯值: {center_value:.6f}")
    print(f"理论最大值: 1.000000")
    print(f"实际最大值: {mask_np.max():.6f}")
    print("-" * 40)
    
    # 统计信息
    print(f"最小值: {mask_np.min():.6f}")
    print(f"最大值: {mask_np.max():.6f}")
    print(f"平均值: {mask_np.mean():.6f}")
    print(f"标准差: {mask_np.std():.6f}")
    print("-" * 40)
    
    # 不同阈值下的像素比例
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    print("不同阈值下的像素比例:")
    for thresh in thresholds:
        ratio = (mask_np >= thresh).mean() * 100
        print(f"  >= {thresh:.1f}: {ratio:.2f}%")
    
    print("-" * 40)
    
    # 距离中心点不同位置的值
    print("距离中心点不同距离的高斯值:")
    distances = [0, 5, 10, 15, 20, 25]
    for dist in distances:
        if center_pixel_x + dist < width and center_pixel_y < height:
            value = mask_np[center_pixel_y, center_pixel_x + dist]
            print(f"  距离中心 {dist} 像素: {value:.6f}")
    
    print("="*60)


def main():
    """
    主函数 - 运行所有可视化
    """
    print("开始生成高斯掩码可视化...")
    
    # 确保输出目录存在
    import os
    output_dir = "/data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised_with_gaussian_mask"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置matplotlib的中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    try:
        # 1. 基础高斯掩码概览
        print("\n1. 生成基础高斯掩码概览...")
        visualize_gaussian_mask_examples()
        
        # 2. 基于边界框的掩码
        print("\n2. 生成基于边界框的高斯掩码...")
        visualize_bbox_based_masks()
        
        # 3. 详细分析图
        print("\n3. 生成高斯掩码详细分析...")
        visualize_gaussian_cross_section()
        
        # 4. 数值统计
        print_gaussian_statistics()
        
        print(f"\n✅ 所有可视化图像已保存到: {output_dir}")
        print("📊 生成的文件:")
        print("  - gaussian_masks_overview.png: 不同参数的高斯掩码对比")
        print("  - bbox_gaussian_masks.png: 基于边界框的高斯掩码")
        print("  - gaussian_analysis.png: 高斯掩码详细分析 (截面图和3D视图)")
        
    except Exception as e:
        print(f"❌ 可视化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()