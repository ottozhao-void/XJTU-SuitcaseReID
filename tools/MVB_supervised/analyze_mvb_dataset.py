#!/usr/bin/env python
"""
MVB数据集分析脚本

此脚本分析MVB数据集的结构和属性，并创建可视化图表以更好地理解其特征。
注意：训练集中20%用作验证集，原始验证集实际作为测试集使用。

创建日期: 2025年4月10日
"""

import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.gridspec as gridspec
from PIL import Image
import random
import matplotlib.font_manager as fm
from matplotlib import rcParams
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免一些字体问题

# 尝试加载适合中文的字体
# 首先尝试系统字体
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'AR PL UMing CN', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
available_fonts = [f.name for f in fm.fontManager.ttflist]

font_found = False
for font in chinese_fonts:
    if font in available_fonts:
        plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
        font_found = True
        break

if not font_found:
    # 如果没有找到中文字体，使用默认字体，注意可能导致中文显示为方块
    print("警告：未找到中文字体，中文可能显示不正确")

plt.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题

# 设置seaborn样式以获得更好的可视化效果
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# Paths
train_info_path = "/data1/zhaofanghan/SuitcaseReID/OpenUnReID/datasets/MVB/MVB_train/Info"
val_info_path = "/data1/zhaofanghan/SuitcaseReID/OpenUnReID/datasets/MVB/MVB_val/Info"
train_format_json = os.path.join(train_info_path, "format.json")
val_gallery_json = os.path.join(val_info_path, "val_gallery.json")
val_probe_json = os.path.join(val_info_path, "val_probe.json")
output_dir = "/data1/zhaofanghan/SuitcaseReID/OpenUnReID/tools/MVB_supervised/analysis_results"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_json_data(file_path):
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_id_from_filename(filename):
    """Extract the suitcase ID from a filename"""
    match = re.match(r'^(\d+)_', filename)
    if match:
        return match.group(1)
    return None

def extract_datatype_from_filename(filename):
    """Extract whether the image is from gallery (g) or probe (p)"""
    match = re.search(r'_([gp])_', filename)
    if match:
        return match.group(1)
    return None

def analyze_train_data():
    """分析训练数据集"""
    print("正在分析训练数据集...")
    train_data = load_json_data(train_format_json)
    
    # 初始化数据结构
    materials = defaultdict(int)
    datasource_count = {'g': 0, 'p': 0}
    view_angles = {'g': defaultdict(int), 'p': defaultdict(int)}
    id_to_images = defaultdict(list)
    image_dims = []
    
    # Process each image
    for img in train_data['image']:
        # Count materials
        material = img.get('material', 'unknown')
        materials[material] += 1
        
        # Count data sources
        datatype = img.get('datatype', 'unknown')
        datasource_count[datatype] += 1
        
        # Extract suitcase ID
        suitcase_id = img.get('id', 'unknown')
        id_to_images[suitcase_id].append(img)
        
        # Extract view angle
        img_name = img.get('image_name', '')
        view_match = re.search(r'_\d+\.jpg$', img_name)
        if view_match:
            view = view_match.group(0)[1:-4]  # Extract the view number
            view_angles[datatype][view] += 1
        
        # Record image dimensions
        dims = img.get('dimsize', [0, 0])
        image_dims.append(dims)
    
    # Calculate statistics
    num_ids = len(id_to_images)
    total_images = len(train_data['image'])
    avg_images_per_id = total_images / num_ids if num_ids > 0 else 0
    
    # Count images per ID
    images_per_id = [len(images) for _, images in id_to_images.items()]
    
    print(f"Number of unique suitcase IDs: {num_ids}")
    print(f"Total number of images: {total_images}")
    print(f"Average images per ID: {avg_images_per_id:.2f}")
    
    # 创建行李箱材料分布图
    plt.figure(figsize=(10, 6))
    materials_df = pd.Series(materials).reset_index()
    materials_df.columns = ['材料类型', '数量']
    plt.title("行李箱材料分布 (训练集)", fontsize=15)
    ax = sns.barplot(data=materials_df, x='材料类型', y='数量', palette='viridis')
    for i, count in enumerate(materials_df['数量']):
        ax.text(i, count + 50, f"{count} ({count/total_images*100:.1f}%)", 
                ha='center', va='bottom', fontsize=10)
    plt.ylabel("图像数量", fontsize=12)
    plt.xlabel("材料类型", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_materials_distribution.png"), dpi=300)
    plt.close()
    
    # 创建数据源分布饼图
    plt.figure(figsize=(8, 6))
    labels = ['安检前 (gallery)', '安检后 (probe)']
    counts = [datasource_count['g'], datasource_count['p']]
    plt.title("图像来源分布 (训练集)", fontsize=15)
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_datasource_distribution.png"), dpi=300)
    plt.close()
    
    # 创建每个ID的图像数量分布图
    plt.figure(figsize=(10, 6))
    plt.title("每个行李箱ID的图像数量分布 (训练集)", fontsize=15)
    sns.histplot(images_per_id, bins=20, kde=True)
    plt.xlabel("每个行李箱ID的图像数量", fontsize=12)
    plt.ylabel("频率", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_images_per_id_distribution.png"), dpi=300)
    plt.close()
    
    # 创建视角分布图
    plt.figure(figsize=(12, 6))
    plt.title("不同来源类型的视角分布 (训练集)", fontsize=15)
    
    # 转换为DataFrame以便绘图
    g_views = pd.Series(view_angles['g']).reset_index()
    g_views.columns = ['视角', '数量']
    g_views['来源'] = '安检前 (gallery)'
    
    p_views = pd.Series(view_angles['p']).reset_index()
    p_views.columns = ['视角', '数量']
    p_views['来源'] = '安检后 (probe)'
    
    views_df = pd.concat([g_views, p_views])
    
    sns.barplot(data=views_df, x='视角', y='数量', hue='来源', palette=['#ff9999','#66b3ff'])
    plt.xlabel("视角", fontsize=12)
    plt.ylabel("数量", fontsize=12)
    plt.legend(title="来源类型")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_view_angle_distribution.png"), dpi=300)
    plt.close()
    
    # Create image dimensions scatter plot
    plt.figure(figsize=(10, 6))
    plt.title("Image Dimensions Distribution (Training Set)", fontsize=15)
    widths = [dim[0] for dim in image_dims]
    heights = [dim[1] for dim in image_dims]
    plt.scatter(widths, heights, alpha=0.3, color='blue')
    plt.xlabel("Width (pixels)", fontsize=12)
    plt.ylabel("Height (pixels)", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_image_dimensions.png"), dpi=300)
    plt.close()
    
    # Material distribution by source type
    material_by_source = defaultdict(lambda: {'g': 0, 'p': 0})
    for img in train_data['image']:
        material = img.get('material', 'unknown')
        datatype = img.get('datatype', 'unknown')
        material_by_source[material][datatype] += 1
    
    # Create material by source plot
    plt.figure(figsize=(12, 7))
    materials_list = list(material_by_source.keys())
    g_counts = [material_by_source[m]['g'] for m in materials_list]
    p_counts = [material_by_source[m]['p'] for m in materials_list]
    
    index = np.arange(len(materials_list))
    bar_width = 0.35
    
    plt.bar(index, g_counts, bar_width, label='Pre-check (gallery)', color='#ff9999')
    plt.bar(index + bar_width, p_counts, bar_width, label='Post-check (probe)', color='#66b3ff')
    
    plt.xlabel('Material Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Material Distribution by Source Type (Training Set)', fontsize=15)
    plt.xticks(index + bar_width/2, materials_list)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_material_by_source.png"), dpi=300)
    plt.close()
    
    return {
        'num_ids': num_ids,
        'total_images': total_images,
        'avg_images_per_id': avg_images_per_id,
        'materials': materials,
        'datasource_count': datasource_count,
        'view_angles': view_angles,
        'images_per_id': images_per_id
    }

def analyze_val_data():
    """分析测试数据集（原验证集）"""
    print("正在分析测试数据集（原验证集）...")
    val_gallery_data = load_json_data(val_gallery_json)
    val_probe_data = load_json_data(val_probe_json)
    
    # Count data
    gallery_count = len(val_gallery_data['image'])
    probe_count = len(val_probe_data['image'])
    
    # Extract IDs from gallery
    gallery_ids = set()
    for img in val_gallery_data['image']:
        id_str = img.get('id')
        if id_str:
            gallery_ids.add(id_str)
    
    num_ids = len(gallery_ids)
    total_images = gallery_count + probe_count
    avg_gallery_per_id = gallery_count / num_ids if num_ids > 0 else 0
    avg_probe_per_id = probe_count / num_ids if num_ids > 0 else 0
    
    print(f"测试集中唯一行李箱ID数量: {num_ids}")
    print(f"测试集中安检前图像(gallery)总数: {gallery_count}")
    print(f"测试集中安检后图像(probe)总数: {probe_count}")
    print(f"每个行李箱ID的平均安检前图像数: {avg_gallery_per_id:.2f}")
    print(f"每个行李箱ID的平均安检后图像数: {avg_probe_per_id:.2f}")
    
    # 创建展示安检前后图像分布的饼图
    plt.figure(figsize=(8, 6))
    labels = ['安检前 (gallery)', '安检后 (probe)']
    counts = [gallery_count, probe_count]
    plt.title("测试集中安检前后图像分布", fontsize=15)
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_gallery_probe_distribution.png"), dpi=300)
    plt.close()
    
    # 统计安检前图像的视角分布
    gallery_view_angles = defaultdict(int)
    for img in val_gallery_data['image']:
        img_name = img.get('image_name', '')
        view_match = re.search(r'_(\d+)\.jpg$', img_name)
        if view_match:
            view = view_match.group(1)
            gallery_view_angles[view] += 1
    
    # 创建测试集安检前图像的视角分布图
    plt.figure(figsize=(10, 6))
    plt.title("测试集中安检前图像的视角分布", fontsize=15)
    view_df = pd.Series(gallery_view_angles).reset_index()
    view_df.columns = ['视角', '数量']
    
    sns.barplot(data=view_df, x='视角', y='数量', color='#ff9999')
    plt.xlabel("视角", fontsize=12)
    plt.ylabel("数量", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "val_gallery_view_angle_distribution.png"), dpi=300)
    plt.close()
    
    return {
        'num_ids': num_ids,
        'gallery_count': gallery_count,
        'probe_count': probe_count,
        'total_images': total_images,
        'avg_gallery_per_id': avg_gallery_per_id,
        'avg_probe_per_id': avg_probe_per_id,
        'gallery_view_angles': gallery_view_angles
    }

def compare_train_val():
    """比较训练集和测试集数据"""
    train_stats = analyze_train_data()
    val_stats = analyze_val_data()
    
    # 创建对比数据表
    comparison_data = {
        'Metric': [
            '唯一行李箱ID数量',
            '总图像数',
            '安检前图像数(gallery)',
            '安检后图像数(probe)',
            '每ID平均图像数'
        ],
        '训练集(80%)': [
            train_stats['num_ids'],
            train_stats['total_images'],
            train_stats['datasource_count']['g'],
            train_stats['datasource_count']['p'],
            f"{train_stats['avg_images_per_id']:.2f}"
        ],
        '测试集(原验证集)': [
            val_stats['num_ids'],
            val_stats['total_images'],
            val_stats['gallery_count'],
            val_stats['probe_count'],
            f"{(val_stats['avg_gallery_per_id'] + val_stats['avg_probe_per_id']):.2f}"
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # 创建汇总表格图
    plt.figure(figsize=(12, 6))
    plt.title("训练集与测试集对比", fontsize=15)
    
    # 关闭坐标轴
    plt.axis('off')
    
    # 创建表格
    the_table = plt.table(
        cellText=df.values, 
        colLabels=df.columns, 
        loc='center', 
        cellLoc='center', 
        colWidths=[0.5, 0.25, 0.25]
    )
    
    # Style the table
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1.5, 2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_val_comparison.png"), dpi=300)
    plt.close()
    
    # 创建分布对比
    categories = ['安检前图像 (Gallery)', '安检后图像 (Probe)']
    train_values = [train_stats['datasource_count']['g'], train_stats['datasource_count']['p']]
    val_values = [val_stats['gallery_count'], val_stats['probe_count']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_values, width, label='训练集')
    rects2 = ax.bar(x + width/2, val_values, width, label='测试集')
    
    ax.set_title('训练集与测试集图像类型对比', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "train_val_distribution_comparison.png"), dpi=300)
    plt.close()
    
    # Create a combined summary plot with multiple visualizations
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
    
    # 数据集大小对比
    ax1 = plt.subplot(gs[0, 0])
    labels = ['训练集', '测试集']
    sizes = [train_stats['total_images'], val_stats['total_images']]
    colors = ['#ff9999', '#66b3ff']
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.axis('equal')
    ax1.set_title('数据集规模对比', fontsize=14)
    
    # 唯一ID数量对比
    ax2 = plt.subplot(gs[0, 1])
    id_counts = [train_stats['num_ids'], val_stats['num_ids']]
    ax2.bar(labels, id_counts, color=colors)
    ax2.set_title('唯一行李箱ID数量', fontsize=14)
    for i, v in enumerate(id_counts):
        ax2.text(i, v + 50, str(v), ha='center')
    
    # 训练集中的图像类型分布
    ax3 = plt.subplot(gs[1, 0])
    train_dist = [
        train_stats['datasource_count']['g'], 
        train_stats['datasource_count']['p']
    ]
    train_labels = ['安检前\n(Gallery)', '安检后\n(Probe)']
    ax3.pie(train_dist, labels=train_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    ax3.axis('equal')
    ax3.set_title('训练集图像分布', fontsize=14)
    
    # 测试集中的图像类型分布
    ax4 = plt.subplot(gs[1, 1])
    val_dist = [val_stats['gallery_count'], val_stats['probe_count']]
    val_labels = ['安检前\n(Gallery)', '安检后\n(Probe)']
    ax4.pie(val_dist, labels=val_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    ax4.axis('equal')
    ax4.set_title('测试集图像分布', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mvb_dataset_summary.png"), dpi=300)
    plt.close()
    
    # 创建综合报告
    report_path = os.path.join(output_dir, "mvb_dataset_analysis.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# MVB数据集分析报告\n\n")
        f.write("## 数据集概述\n\n")
        f.write("MVB（Multi-View Baggage）数据集包含在机场拍摄的行李箱图像，")
        f.write("这些图像从多个视角拍摄，分别在安检前和安检后拍摄。训练过程中，原始训练集的80%用于实际训练，20%用作验证集；而原始验证集则被用于最终测试。\n\n")
        
        f.write("### 统计摘要\n\n")
        f.write("| 指标 | 训练集(80%) | 测试集(原验证集) |\n")
        f.write("|--------|-------------|---------------|\n")
        for i in range(len(comparison_data['Metric'])):
            f.write(f"| {comparison_data['Metric'][i]} | {comparison_data['训练集(80%)'][i]} | {comparison_data['测试集(原验证集)'][i]} |\n")
        
        f.write("\n## 训练集分析\n\n")
        f.write(f"- 训练集包含 {train_stats['num_ids']} 个唯一行李箱ID，共有 {train_stats['total_images']} 张图像。\n")
        f.write(f"- 平均每个行李箱ID有 {train_stats['avg_images_per_id']:.2f} 张图像。\n")
        f.write(f"- 其中安检前(gallery)图像 {train_stats['datasource_count']['g']} 张，安检后(probe)图像 {train_stats['datasource_count']['p']} 张。\n")
        f.write(f"- 注意：实际训练过程中，80%用于训练，20%用于验证。\n\n")
        
        f.write("### 材料分布\n\n")
        f.write("| 材料类型 | 数量 | 百分比 |\n")
        f.write("|----------|-------|------------|\n")
        total = sum(train_stats['materials'].values())
        for material, count in sorted(train_stats['materials'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total) * 100
            f.write(f"| {material} | {count} | {percentage:.2f}% |\n")
        
        f.write("\n## 测试集分析（原验证集）\n\n")
        f.write(f"- 测试集包含 {val_stats['num_ids']} 个唯一行李箱ID，共有 {val_stats['total_images']} 张图像。\n")
        f.write(f"- 其中安检前图像(gallery) {val_stats['gallery_count']} 张，安检后图像(probe) {val_stats['probe_count']} 张。\n")
        f.write(f"- 平均每个行李箱ID有 {val_stats['avg_gallery_per_id']:.2f} 张安检前图像和 {val_stats['avg_probe_per_id']:.2f} 张安检后图像。\n\n")
        
        f.write("\n## 可视化图表\n\n")
        
        f.write("以下可视化图表已生成，用于帮助理解数据集特征：\n\n")
        f.write("1. **材料分布**：训练集中不同行李箱材料的分布\n")
        f.write("2. **数据来源分布**：安检前(gallery)与安检后(probe)图像的比例\n")
        f.write("3. **每ID图像数量**：展示每个行李箱ID的可用图像数量分布\n")
        f.write("4. **视角分布**：gallery和probe图像中不同视角的分布\n")
        f.write("5. **图像尺寸**：显示图像宽度和高度分布的散点图\n")
        f.write("6. **比较图**：训练集和测试集之间的各种比较\n\n")
        
        f.write("所有可视化图表均保存在与本报告相同的目录中。\n")

    print(f"Analysis complete. Results saved to {output_dir}")
    return report_path

if __name__ == "__main__":
    compare_train_val()
