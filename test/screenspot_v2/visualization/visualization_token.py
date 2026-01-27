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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tester.qwen3vl_visionselector_tester import Qwen3VLVisionSelectorTester
from tester.qwen3vl_sparse_tester import Qwen3VLSparseTester
from util import align_size_to_patch


def draw_original_image(
    image: Image.Image,
    ax: Optional[plt.Axes] = None,,
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
    if hasattr(text_model, "selected_vision_idx_list"):
        selected_idx_list = text_model.selected_idx_list
    draw_bbox_and_pred(
        image=image,
        bbox=sample["bbox"],
        pred=coordinates,
        save_path=os.path.join(save_path, "bbox_pred.png"),
    )
    for layer_idx, selected_vision_idx in selected_idx_list.items():
        selected_indices = selected_vision_idx.tolist()
        vision_range = tester.vision_range
        selected_indices = [
            i for i in selected_indices
            if vision_range[0] <= i < vision_range[1]
        ]
        print(f"\n选中的token数量: {len(selected_indices)}")
        draw_selected_tokens(
            image,
            selected_indices,
            token_patch_size,
            save_path=os.path.join(save_path, f"selected_tokens_{layer_idx}.png"),
        )


def main():
    parser = argparse.ArgumentParser(description="可视化Qwen3VLVisionSelector的visual token")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--image_path", type=str, required=True, help="图像路径")
    parser.add_argument("--instruction", type=str, default="Click on the center of the screen", help="指令文本")
    parser.add_argument("--output", type=str, default="./output", help="输出目录")
    parser.add_argument("--budgets", type=float, default=0.5, help="token budget比例")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 加载模型
    print(f"正在加载模型: {args.model_path}")
    tester = Qwen3VLVisionSelectorTester(args.model_path, budgets=args.budgets)
    print("模型加载完成")

    # 生成保存路径
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(args.output, f"{image_name}_visualize")
    sample = {
        "img_path": args.image_path,
        "text": args.instruction,
        "bbox": [0.64, 0.10, 0.92, 0.18],
    }
    # 可视化
    print("\n开始可视化...")
    print(f"指令: {args.instruction}")
    coordinates, response = visualize_visionselector_tokens(
        tester=tester,
        save_path=save_path,
        show=False,
        sample=sample,
    )

    print("\n可视化完成！")
    print("输出文件:")
    print(f"- {save_path}: 显示token分数热力图")


if __name__ == "__main__":
    main()
