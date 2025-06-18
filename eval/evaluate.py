# evaluate.py
"""
用于评估和可视化 MultiEddyNet 模型性能的脚本。
加载预训练模型，对指定日期的数据进行预测，
并将结果与真实标签进行比较，生成差异可视化图。
"""
import logging
from pathlib import Path
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from skimage.measure import find_contours, label

# 从其他模块导入必要的组件
import config
from model import MultiEddyNet

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- 评估特定参数 ---
TARGET_DATE_STR = "19970918"  # 指定要评估的日期
IOU_MATCH_THRESHOLD = 0.1     # 匹配涡旋的 IoU 阈值
OUTPUT_DIR = config.BASE_DIR / "evaluation_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 绘图样式配置 ---
PLOT_STYLE = {
    'gt_warm_color': 'black', 'gt_cold_color': 'black', 'gt_linestyle': '-',
    'pred_warm_color': 'red', 'pred_cold_color': 'red', 'pred_linestyle': '--',
    'fp_warm_color': '#ff00ff', 'fp_cold_color': '#00ffff', # FP: 洋红, 青色
    'fn_warm_color': '#ff9900', 'fn_cold_color': '#0099ff', # FN: 橙色, 蓝色
    'linewidth': 1.2, 'dpi': 300,
    'extent': [-75, -55, 29.8, 42.64]
}

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """计算两个布尔掩膜之间的 IoU。"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0

def find_eddy_matches(gt_labels: np.ndarray, pred_labels: np.ndarray, iou_threshold: float):
    """
    比较GT和Pred标签图，识别匹配、FN、FP的涡旋。
    返回一个包含所有匹配信息和差异信息的字典。
    """
    results = {}
    for eddy_type, label_value in [('warm', 1), ('cold', 2)]:
        logging.info(f"--- 正在处理 {eddy_type} 涡旋 (标签={label_value}) ---")
        
        labeled_gt, num_gt = label(gt_labels == label_value, connectivity=2, return_num=True)
        labeled_pred, num_pred = label(pred_labels == label_value, connectivity=2, return_num=True)
        
        iou_matrix = np.zeros((num_gt, num_pred))
        for r in range(num_gt):
            for c in range(num_pred):
                iou_matrix[r, c] = calculate_iou(labeled_gt == r + 1, labeled_pred == c + 1)
        
        gt_matched = [False] * num_gt
        pred_matched = [False] * num_pred
        matched_pairs = []
        
        # 贪心匹配，优先选择IoU最高的对
        sorted_indices = np.dstack(np.unravel_index(np.argsort(iou_matrix.ravel())[::-1], iou_matrix.shape))[0]
        
        for r, c in sorted_indices:
            if iou_matrix[r, c] < iou_threshold:
                break
            if not gt_matched[r] and not pred_matched[c]:
                matched_pairs.append({'gt_id': r + 1, 'pred_id': c + 1, 'iou': iou_matrix[r, c]})
                gt_matched[r] = True
                pred_matched[c] = True
        
        results[eddy_type] = {
            'matched_pairs': matched_pairs,
            'fn_ids': [i + 1 for i, matched in enumerate(gt_matched) if not matched],
            'fp_ids': [i + 1 for i, matched in enumerate(pred_matched) if not matched],
            'labeled_gt': labeled_gt,
            'labeled_pred': labeled_pred
        }
        logging.info(f"匹配数: {len(matched_pairs)}, 漏检数: {len(results[eddy_type]['fn_ids'])}, "
                     f"误检数: {len(results[eddy_type]['fp_ids'])}")
                     
    return results

def setup_map_axis(ax: plt.Axes, extent: list, title: str):
    """设置地图坐标轴的基本样式。"""
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    land = cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', facecolor='lightgray')
    ax.add_feature(land, zorder=5)
    ax.coastlines(resolution='10m', color='black', linewidth=0.7, zorder=5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle=':')
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': 9, 'color': 'black'}

def get_contours(labeled_map, ids, x_grid, y_grid):
    """为指定的实例ID列表提取地理轮廓。"""
    contours = []
    for inst_id in ids:
        for contour in find_contours(labeled_map == inst_id, 0.5):
            x_coords = np.interp(contour[:, 1], np.arange(x_grid.shape[1]), x_grid[0, :])
            y_coords = np.interp(contour[:, 0], np.arange(y_grid.shape[0]), y_grid[:, 0])
            contours.append(np.column_stack([x_coords, y_coords]))
    return contours

def plot_contours(ax, contours, color, linestyle, linewidth):
    """在地图上绘制轮廓。"""
    for cnt in contours:
        ax.plot(cnt[:, 0], cnt[:, 1], color=color, linestyle=linestyle, lw=linewidth, transform=ccrs.PlateCarree(), zorder=10)

def create_filled_map(shape, labeled_map, ids, fill_value):
    """创建一个只包含指定ID填充区域的地图。"""
    filled_map = np.zeros(shape, dtype=np.uint8)
    for inst_id in ids:
        filled_map[labeled_map == inst_id] = fill_value
    return filled_map

def main():
    """主评估函数。"""
    # --- 1. 加载模型 ---
    if not config.MODEL_SAVE_PATH.exists():
        logging.error(f"模型权重文件未找到: {config.MODEL_SAVE_PATH}")
        return
        
    model = MultiEddyNet(in_ch=config.IN_CHANNELS, out_ch=config.NUM_CLASSES)
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.to(config.DEVICE).eval()
    logging.info(f"模型已从 {config.MODEL_SAVE_PATH} 加载。")

    # --- 2. 加载数据 ---
    with h5py.File(config.HDF5_FILE, 'r') as hf:
        x_center_1d, y_center_1d = hf['x'][:], hf['y'][:]
        x_grid, y_grid = np.meshgrid(x_center_1d, y_center_1d)
        
        time_data = hf['time'][:]
        date_idx = np.where(time_data == int(TARGET_DATE_STR))[0]
        if not date_idx.size:
            logging.error(f"日期 {TARGET_DATE_STR} 未在数据集中找到。")
            return
        time_idx = date_idx[0]

        features_np = hf['features'][time_idx, ..., config.FEATURE_INDICES_TO_USE]
        gt_labels = hf['labels'][time_idx, :, :]

    # --- 3. 模型预测 ---
    features_tensor = torch.from_numpy(np.nan_to_num(features_np)).unsqueeze(0).permute(0, 3, 1, 2).to(config.DEVICE)
    with torch.no_grad():
        pred_logits = model(features_tensor)
    pred_labels = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()

    # --- 4. 匹配与分析 ---
    match_results = find_eddy_matches(gt_labels, pred_labels, IOU_MATCH_THRESHOLD)

    # --- 5. 绘制结果 ---
    # 图1: 匹配的涡旋边界
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    setup_map_axis(ax1, PLOT_STYLE['extent'], f'Matched Eddy Boundaries - {TARGET_DATE_STR}')
    for eddy_type, style_prefix in [('warm', 'gt_warm'), ('cold', 'gt_cold')]:
        gt_ids = [p['gt_id'] for p in match_results[eddy_type]['matched_pairs']]
        contours = get_contours(match_results[eddy_type]['labeled_gt'], gt_ids, x_grid, y_grid)
        plot_contours(ax1, contours, PLOT_STYLE[f'{style_prefix}_color'], PLOT_STYLE['gt_linestyle'], PLOT_STYLE['linewidth'])
    for eddy_type, style_prefix in [('warm', 'pred_warm'), ('cold', 'pred_cold')]:
        pred_ids = [p['pred_id'] for p in match_results[eddy_type]['matched_pairs']]
        contours = get_contours(match_results[eddy_type]['labeled_pred'], pred_ids, x_grid, y_grid)
        plot_contours(ax1, contours, PLOT_STYLE[f'{style_prefix}_color'], PLOT_STYLE['pred_linestyle'], PLOT_STYLE['linewidth'])
    
    handles = [plt.Line2D([], [], color='black', ls='-', label='GT Boundary'),
               plt.Line2D([], [], color='red', ls='--', label='Pred Boundary')]
    ax1.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig1.savefig(OUTPUT_DIR / f"diff_matched_{TARGET_DATE_STR}.png", dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close(fig1)

    # 图2: 误检 (FP)
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    setup_map_axis(ax2, PLOT_STYLE['extent'], f'False Positives (FP) - {TARGET_DATE_STR}')
    fp_map = np.zeros_like(gt_labels, dtype=np.uint8)
    fp_map = np.maximum(fp_map, create_filled_map(fp_map.shape, match_results['warm']['labeled_pred'], match_results['warm']['fp_ids'], 1))
    fp_map = np.maximum(fp_map, create_filled_map(fp_map.shape, match_results['cold']['labeled_pred'], match_results['cold']['fp_ids'], 2))
    fp_cmap = mcolors.ListedColormap(['white', PLOT_STYLE['fp_warm_color'], PLOT_STYLE['fp_cold_color']])
    ax2.pcolormesh(x_grid, y_grid, fp_map, cmap=fp_cmap, norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], fp_cmap.N), transform=ccrs.PlateCarree())
    handles = [mpatches.Patch(color=PLOT_STYLE['fp_warm_color'], label='Extra Warm (FP)'),
               mpatches.Patch(color=PLOT_STYLE['fp_cold_color'], label='Extra Cold (FP)')]
    ax2.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig2.savefig(OUTPUT_DIR / f"diff_fp_{TARGET_DATE_STR}.png", dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close(fig2)

    # 图3: 漏检 (FN)
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    setup_map_axis(ax3, PLOT_STYLE['extent'], f'False Negatives (FN) - {TARGET_DATE_STR}')
    fn_map = np.zeros_like(gt_labels, dtype=np.uint8)
    fn_map = np.maximum(fn_map, create_filled_map(fn_map.shape, match_results['warm']['labeled_gt'], match_results['warm']['fn_ids'], 1))
    fn_map = np.maximum(fn_map, create_filled_map(fn_map.shape, match_results['cold']['labeled_gt'], match_results['cold']['fn_ids'], 2))
    fn_cmap = mcolors.ListedColormap(['white', PLOT_STYLE['fn_warm_color'], PLOT_STYLE['fn_cold_color']])
    ax3.pcolormesh(x_grid, y_grid, fn_map, cmap=fn_cmap, norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], fn_cmap.N), transform=ccrs.PlateCarree())
    handles = [mpatches.Patch(color=PLOT_STYLE['fn_warm_color'], label='Missed Warm (FN)'),
               mpatches.Patch(color=PLOT_STYLE['fn_cold_color'], label='Missed Cold (FN)')]
    ax3.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig3.savefig(OUTPUT_DIR / f"diff_fn_{TARGET_DATE_STR}.png", dpi=PLOT_STYLE['dpi'], bbox_inches='tight')
    plt.close(fig3)

    logging.info(f"评估图已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()