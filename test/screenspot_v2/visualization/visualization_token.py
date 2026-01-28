import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from matplotlib.colors import Normalize
import os
import argparse
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tester.qwen3vl_visionselector_tester import Qwen3VLVisionSelectorTester
from tester.qwen3vl_sparse_tester import Qwen3VLSparseTester
from util import align_size_to_patch


def draw_original_image(
    image: Image.Image,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Image.Image:
    img_copy = image.copy()
    if ax is not None:
        ax.imshow(img_copy)
        ax.axis("off")
        ax.set_title("Original Image")
    if save_path:
        img_copy.save(save_path)
    return img_copy


def draw_bbox_and_pred(
    image: Image.Image,
    bbox: List[float],
    pred: List[float] | tuple[float, float],
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Image.Image:
    img_copy = image.copy()
    W, H = img_copy.size
    draw = ImageDraw.Draw(img_copy)

    x1, y1, x2, y2 = bbox
    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1:
        x1, y1 = x1 * W, y1 * H
        x2, y2 = x2 * W, y2 * H
    draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)

    px, py = pred
    if 0 <= px <= 1 and 0 <= py <= 1:
        px, py = px * W, py * H
    r = 14
    draw.ellipse([px - r, py - r, px + r, py + r], fill="red", outline="red")

    if ax is not None:
        ax.imshow(img_copy)
        ax.axis("off")
        ax.set_title("BBox and Prediction")
    if save_path:
        img_copy.save(save_path)
    return img_copy


def draw_selected_tokens(
    image: Image.Image,
    selected_indices: List[int],
    patch_size: int,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> Image.Image:
    img_copy = image.copy().convert("RGBA")
    W, H = img_copy.size
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    total_tokens = num_patches_h * num_patches_w

    overlay = Image.new("RGBA", img_copy.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    selected_set = set(selected_indices)

    for idx in range(total_tokens):
        if idx not in selected_set:
            row = idx // num_patches_w
            col = idx % num_patches_w
            x1, y1 = col * patch_size, row * patch_size
            x2, y2 = (col + 1) * patch_size, (row + 1) * patch_size
            draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 255))

    img_copy = Image.alpha_composite(img_copy, overlay)

    if ax is not None:
        ax.imshow(img_copy)
        ax.axis("off")
        ax.set_title("Selected Tokens")
    if save_path:
        img_copy.save(save_path)
    return img_copy.convert("RGB")


def draw_token_heatmap(
    image: Image.Image,
    token_scores: torch.Tensor,
    patch_size: int,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
):
    """
    在 ax 中绘制 heatmap，并在 ax 内部“右侧”创建 colorbar，不覆盖 image
    """

    def _draw(ax, image, heatmap_resized):
        # 关键：为 colorbar 划空间
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            "right",
            size="6%",
            pad=0.1,
        )

        ax.imshow(image, alpha=0.5)
        im = ax.imshow(heatmap_resized, cmap="jet", alpha=0.5)

        ax.axis("off")
        ax.set_title("Token Importance Heatmap")

        plt.colorbar(im, cax=cax)

    # ---------- heatmap ----------
    W, H = image.size
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    scores = token_scores.float().cpu().numpy()
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    heatmap = scores.reshape(num_patches_h, num_patches_w)

    heatmap_resized = np.repeat(
        np.repeat(heatmap, patch_size, axis=0),
        patch_size,
        axis=1,
    )

    # 子图
    _draw(ax, image, heatmap_resized)

    # 单独保存
    if save_path is not None:
        fig, ax_single = plt.subplots(1, 1, figsize=(6, 6))
        _draw(ax_single, image, heatmap_resized)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def visualize_tokens(
    image: Image.Image,
    token_scores: torch.Tensor,
    bbox,
    pred,
    save_path,
    selected_indices: List[int],
    patch_size: int = 16,
    show: bool = True,
):
    """
    可视化原图、bbox/pred、选中tokens和token热力图
    """
    # 调整图像大小为patch的倍数
    width, height = align_size_to_patch(image, patch_size)
    print(f"调整后的图像大小: {width}x{height}")
    # 计算总token数量
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    total_tokens = num_patches_h * num_patches_w
    print(f"总token数量: {total_tokens}")
    print(f"选择比例: {len(selected_indices) / total_tokens * 100:.2f}%")

    image = image.resize((width, height)).convert("RGBA")
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # 1. 原始图像
    draw_original_image(image, ax=axes[0], save_path=os.path.join(save_path, "original.png"))
    draw_selected_tokens(
        image,
        selected_indices,
        patch_size,
        ax=axes[2],
        save_path=os.path.join(save_path, "selected_tokens.png"),
    )
    draw_bbox_and_pred(image, bbox, pred, ax=axes[1], save_path=os.path.join(save_path, "bbox_pred.png"))

    draw_token_heatmap(
        image,
        token_scores,
        patch_size,
        ax=axes[3],
        save_path=os.path.join(save_path, "heatmap.png"),
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, os.path.basename(save_path) + ".png"), bbox_inches="tight", dpi=300)
    if show:
        plt.show()


def visualize_visionselector_tokens(
    tester: Qwen3VLVisionSelectorTester,
    save_path: str,
    sample: dict,
    show: bool = True,
):
    """
    可视化Qwen3VLVisionSelector的visual token选择

    Args:
        tester: Qwen3VLVisionSelectorTester实例
        image: 输入图像
        instruction: 指令文本
        save_path: 保存路径（不含扩展名）
        show: 是否显示可视化结果
        sample: 包含bbox和pred的样本字典（可选）
    """
    # 获取模型的visual模块
    img_path = sample["img_path"]
    image = Image.open(img_path).convert("RGB")
    instruction = sample["text"]
    visual_model = tester.model.model.visual

    # 获取patch_size
    patch_size = visual_model.patch_size
    spatial_merge_size = visual_model.spatial_merge_size
    token_patch_size = patch_size * spatial_merge_size

    # 调用模型生成点击坐标（这会触发visual token选择）
    coordinates, response = tester.generate_click_coordinate(instruction, img_path)

    print(f"\n生成的响应: {response}")
    print(f"点击坐标: {coordinates}")

    # 检查是否有保存的token选择信息
    if hasattr(visual_model, "last_selected_indices"):
        selected_indices = visual_model.last_selected_indices.cpu().tolist()
        print(f"\n选中的token数量: {len(selected_indices)}")

    # 检查是否有保存的token分数
    if hasattr(visual_model, "learned_scores"):
        token_scores = visual_model.learned_scores

        print(f"\nToken分数形状: {token_scores.shape}")

        # 可视化token分数热力图
        print("正在可视化token分数热力图...")
        visualize_tokens(
            image=image,
            token_scores=token_scores,
            selected_indices=selected_indices,
            patch_size=token_patch_size,
            save_path=f"{save_path}" if save_path else None,
            show=show,
            bbox=sample["bbox"],
            pred=coordinates,
        )

    return coordinates, response


def visualize_sparse_tokens(
    tester: Qwen3VLSparseTester,
    save_path: str,
    sample: dict,
    show: bool = True,
):
    """
    可视化Qwen3VLSparse的visual token选择

    Args:
        tester: Qwen3VLSparseTester实例
        save_path: 保存路径（不含扩展名）
        show: 是否显示可视化结果
        sample: 包含bbox和pred的样本字典（可选）
    """
    # 获取模型的visual模块
    os.makedirs(save_path, exist_ok=True)
    img_path = sample["img_path"]
    image = Image.open(img_path).convert("RGB")
    instruction = sample["text"]
    text_model = tester.model.model.language_model
    visual_model = tester.model.model.visual
    # 获取patch_size
    patch_size = visual_model.patch_size
    spatial_merge_size = visual_model.spatial_merge_size
    token_patch_size = patch_size * spatial_merge_size

    # 调用模型生成点击坐标（这会触发visual token选择）
    coordinates, response = tester.generate_click_coordinate(instruction, img_path)

    print(f"\n生成的响应: {response}")
    print(f"点击坐标: {coordinates}")

    # 检查是否有保存的token选择信息
    if hasattr(text_model, "selected_idx_list"):
        selected_idx_list = text_model.selected_idx_list
    draw_bbox_and_pred(
        image=image,
        bbox=sample["bbox"],
        pred=coordinates,
        save_path=os.path.join(save_path, "bbox_pred.png"),
    )
    for layer_idx, selected_idx in selected_idx_list.items():
        selected_indices = selected_idx.tolist()
        vision_range = tester.vision_range
        selected_indices = [i for i in selected_indices if vision_range[0] <= i < vision_range[1]]
        selected_indices = [i - vision_range[0] for i in selected_indices]
        print(f"\n选中的token数量: {len(selected_indices)}")
        draw_selected_tokens(
            image,
            selected_indices,
            token_patch_size,
            save_path=os.path.join(save_path, f"selected_tokens_{layer_idx}.png"),
        )
    return coordinates, response


def batch_visualize(
    tester,
    data_list: list,
    output_dir: str,
    show: bool = False,
):
    """
    批量可视化

    Args:
        tester
        data_list: 数据列表，每个元素是包含"img_path"和"text"的dict
        output_dir: 输出目录
        show: 是否显示可视化结果
    """
    os.makedirs(output_dir, exist_ok=True)

    total = len(data_list)
    print(f"开始批量处理，共 {total} 个图像\n")

    results = []

    for idx, item in enumerate(data_list):
        img_path = item.get("img_path")
        instruction = item.get("text")

        if not img_path or not instruction:
            print(f"[{idx + 1}/{total}] 跳过：缺少img_path或text")
            continue

        if not os.path.exists(img_path):
            print(f"[{idx + 1}/{total}] 跳过：图像不存在 {img_path}")
            continue

        try:
            # 生成保存路径
            image_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{idx:03d}_{image_name}_visualize")
            # 检查tester类型
            if isinstance(tester, Qwen3VLVisionSelectorTester):
                coordinates, response = visualize_visionselector_tokens(
                    tester=tester,
                    save_path=save_path,
                    show=show,
                    sample=item,
                )
            elif isinstance(tester, Qwen3VLSparseTester):
                coordinates, response = visualize_sparse_tokens(
                    tester=tester,
                    save_path=save_path,
                    show=show,
                    sample=item,
                )
            else:
                raise TypeError("tester 必须是 Qwen3VLVisionSelectorTester 或 Qwen3VLSparseTester 类型")

            result = {
                "index": idx,
                "img_path": img_path,
                "instruction": instruction,
                "pred": coordinates,
                "bbox": item.get("bbox"),
                "response": response,
            }
            results.append(result)

            # print(f"  完成: {save_path}_selected_tokens.png")
            print(f"  完成: {save_path}_visualize\n")

        except Exception as e:
            print(f"  错误: {str(e)}\n")
            result = {
                "index": idx,
                "img_path": img_path,
                "instruction": instruction,
                "bbox": item.get("bbox"),
                "response": response,
                "error": str(e),
            }
            results.append(result)
        with open(os.path.join(save_path, "visualize_results.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
    print("\n批量处理完成！")
    print(f"成功: {sum(1 for r in results if 'error' not in r)}/{total}")
    print(f"失败: {sum(1 for r in results if 'error' in r)}/{total}")


def main():
    parser = argparse.ArgumentParser(description="批量可视化Qwen3VLVisionSelector的visual token")
    parser.add_argument(
        "--model_type", type=str, required=True, help="模型类型 (qwen3vl_vision_selector 或 qwen3vl_sparse)"
    )
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--json_path", type=str, required=True, help="JSON文件路径")
    parser.add_argument("--output", type=str, default="test/screenspot_v2/visualization/output/", help="输出目录")
    parser.add_argument("--budgets", type=float, default=0.5, help="token budget比例")
    parser.add_argument("--show", action="store_true", help="显示可视化结果")

    args = parser.parse_args()

    # 加载JSON数据
    print(f"正在加载JSON文件: {args.json_path}")
    with open(args.json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    print(f"加载了 {len(data_list)} 条数据")

    if not data_list:
        print("错误: JSON文件为空")
        return
    # 加载模型
    print(f"\n正在加载模型: {args.model_path}")
    if "qwen3vl_vision_selector" in args.model_type.lower():
        tester = Qwen3VLVisionSelectorTester(args.model_path, budgets=args.budgets)
    elif "qwen3vl_sparse" in args.model_type.lower():
        tester = Qwen3VLSparseTester(args.model_path, budgets=args.budgets)
    else:
        raise ValueError("模型类型必须包含 'qwen3vl_vision_selector' 或 'qwen3vl_sparse'")
    print("模型加载完成\n")

    output_dir = os.path.join(args.output, os.path.basename(args.model_path))
    # 批量可视化
    batch_visualize(
        tester=tester,
        data_list=data_list,
        output_dir=output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
