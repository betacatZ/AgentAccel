import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import argparse
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tester.qwen3vl_visionselector_tester import Qwen3VLVisionSelectorTester


def resize_to_patch_multiple(image: Image.Image, patch_size: int = 16) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    将图像调整到patch_size的倍数大小（向上取整）

    Args:
        image: 原始图像
        patch_size: patch大小

    Returns:
        调整后的图像和新的图像大小
    """
    width, height = image.size

    # 计算调整后的大小，向上取整到patch_size的倍数
    new_width = ((width + patch_size - 1) // patch_size) * patch_size
    new_height = ((height + patch_size - 1) // patch_size) * patch_size

    # 如果已经是倍数，不做调整
    if new_width == width and new_height == height:
        return image, (width, height)

    # 调整图像大小
    resized_image = image.resize((new_width, new_height))
    return resized_image, (new_width, new_height)


def visualize_token_scores(
    image: Image.Image,
    token_scores: torch.Tensor,
    selected_indices: Optional[List[int]] = None,
    patch_size: int = 16,
    image_size: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    可视化token的分数热力图和选择结果

    Args:
        image: 原始图像
        token_scores: 所有token的分数
        selected_indices: 选择的token索引列表（可选）
        patch_size: 视觉token的patch大小（默认为16）
        image_size: 图像的原始大小，如果为None则使用图像的实际大小
        save_path: 保存可视化结果的路径
        show: 是否显示可视化结果

    Returns:
        热力图和叠加后的图像
    """
    # 调整图像大小
    if image_size is not None:
        # 如果指定了image_size，先调整到指定大小，再调整到patch_size的倍数
        image = image.resize(image_size)

    # 将图像调整到patch_size的倍数
    image, (width, height) = resize_to_patch_multiple(image, patch_size)

    # 将图像转换为RGBA模式以支持透明度
    image = image.convert("RGBA")

    # 计算图像的patch数量
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    total_tokens = num_patches_h * num_patches_w

    # 创建图像副本用于matplotlib显示
    image_copy = image.copy()

    # 将token分数转换为热力图
    scores = token_scores.float().cpu().numpy()
    heatmap = scores.reshape(num_patches_h, num_patches_w)

    # 调整热力图大小以匹配图像
    heatmap_resized = np.repeat(np.repeat(heatmap, patch_size, axis=0), patch_size, axis=1)

    # 如果有选择的token，将未选中的token设置为半透明
    if selected_indices:
        # 创建一个透明覆盖层
        overlay_copy = Image.new("RGBA", image_copy.size, (0, 0, 0, 0))
        draw_copy = ImageDraw.Draw(overlay_copy)
        selected_set = set(selected_indices)

        # 绘制未被选中的token（半透明灰色覆盖）
        for idx in range(total_tokens):
            if idx not in selected_set:
                row = idx // num_patches_w
                col = idx % num_patches_w
                x1 = col * patch_size
                y1 = row * patch_size
                x2 = (col + 1) * patch_size
                y2 = (row + 1) * patch_size
                draw_copy.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 255), outline=None)

        # 将覆盖层叠加到image_copy上
        image_copy = Image.alpha_composite(image_copy, overlay_copy)

        # 被选中的token保持原样显示（不绘制红色边框）

    # 创建一个包含三个子图的布局
    plt.figure(figsize=(20, 8))

    # 1. 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image")

    # 2. Selector Token（选中的token高亮显示）
    plt.subplot(1, 3, 2)
    plt.imshow(image_copy)
    plt.axis("off")
    plt.title("Selected Tokens")

    # 3. 热力图
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(heatmap_resized, cmap="jet", alpha=0.5)
    plt.colorbar(label="Token Importance Score", shrink=0.8)
    plt.axis("off")
    plt.title("Token Importance Heatmap")

    # 调整子图间距
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    # 显示图像
    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument("--config", "-c", type=str, default=None, help="path to config.yaml")
    parser.add_argument("--output", type=str, default="./output")
    args = parser.parse_args()
    return args


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def visualize_visionselector_tokens(
    tester: Qwen3VLVisionSelectorTester,
    image: Image.Image,
    instruction: str,
    save_path: str,
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
    """
    # 获取模型的visual模块
    visual_model = tester.model.model.visual

    # 获取patch_size
    patch_size = getattr(visual_model, "patch_size", 16)
    spatial_merge_size = getattr(visual_model, "spatial_merge_size", 2)
    token_patch_size = patch_size * spatial_merge_size

    # 调整图像大小到patch_size的倍数
    image, (width, height) = resize_to_patch_multiple(image, token_patch_size)
    print(f"调整后的图像大小: {width}x{height}")

    # 调用模型生成点击坐标（这会触发visual token选择）
    coordinates, response = tester.generate_click_coordinate(instruction, image)

    print(f"\n生成的响应: {response}")
    print(f"点击坐标: {coordinates}")

    # 检查是否有保存的token选择信息
    if hasattr(visual_model, "last_selected_indices"):
        selected_indices = visual_model.last_selected_indices.cpu().tolist()
        print(f"\n选中的token数量: {len(selected_indices)}")

        # 计算总token数量
        num_patches_h = height // token_patch_size
        num_patches_w = width // token_patch_size
        total_tokens = num_patches_h * num_patches_w
        print(f"总token数量: {total_tokens}")
        print(f"选择比例: {len(selected_indices) / total_tokens * 100:.2f}%")

    # 检查是否有保存的token分数
    if hasattr(visual_model, "learned_scores"):
        token_scores = visual_model.learned_scores
        print(f"\nToken分数形状: {token_scores.shape}")

        # 可视化token分数热力图
        print("正在可视化token分数热力图...")
        visualize_token_scores(
            image=image,
            token_scores=token_scores,
            selected_indices=selected_indices if hasattr(visual_model, "last_selected_indices") else None,
            patch_size=token_patch_size,
            save_path=f"{save_path}_token_heatmap.png" if save_path else None,
            show=show,
        )

    return coordinates, response


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

    # 加载图像
    print(f"\n正在加载图像: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    print(f"原始图像大小: {image.size}")

    # 生成保存路径
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(args.output, f"{image_name}_visualize")

    # 可视化
    print("\n开始可视化...")
    print(f"指令: {args.instruction}")
    coordinates, response = visualize_visionselector_tokens(
        tester=tester,
        image=image,
        instruction=args.instruction,
        save_path=save_path,
        show=False,
    )

    print("\n可视化完成！")
    print("输出文件:")
    print(f"- {save_path}_selected_tokens.png: 显示选择的token区域")
    print(f"- {save_path}_token_heatmap.png: 显示token分数热力图")


if __name__ == "__main__":
    main()
    # 示例用法
    # import sys
    # from PIL import Image
    # import torch

    # if len(sys.argv) < 2:
    #     print("Usage: python draw_token.py <image_path> [selected_indices]")
    #     sys.exit(1)

    # # 加载图像
    # image_path = sys.argv[1]
    # image = Image.open(image_path)

    # # 获取图像的实际大小
    # original_width, original_height = image.size
    # print(f"原始图像大小: {original_width}x{original_height}")

    # # 使用默认的patch_size 16
    # patch_size = 16
    # # 计算调整后的图像大小（向上取整到patch_size的倍数）
    # adjusted_width = ((original_width + patch_size - 1) // patch_size) * patch_size
    # adjusted_height = ((original_height + patch_size - 1) // patch_size) * patch_size
    # print(f"调整后的图像大小 (patch_size={patch_size}的倍数): {adjusted_width}x{adjusted_height}")

    # # 示例token索引（实际使用时应从模型获取）
    # if len(sys.argv) > 2:
    #     selected_indices = list(map(int, sys.argv[2].split(",")))
    # else:
    #     # 随机选择一半的token作为示例
    #     total_tokens = (adjusted_height // patch_size) * (adjusted_width // patch_size)
    #     num_selected = total_tokens // 2  # 选择一半的token
    #     selected_indices = np.random.choice(total_tokens, num_selected, replace=False).tolist()
    #     selected_indices.sort()  # 保持索引顺序
    #     print(f"随机选择的token数量: {len(selected_indices)} (总token数量: {total_tokens})")
    #     print(f"选择的token索引: {selected_indices[:10]}...")

    # # 示例token分数（实际使用时应从模型获取）
    # total_tokens = (adjusted_height // patch_size) * (adjusted_width // patch_size)
    # token_scores = torch.randn(total_tokens)
    # token_scores = torch.softmax(token_scores, dim=0)

    # # 可视化token分数热力图
    # print("\n--- 可视化token分数热力图 ---")
    # visualize_token_scores(
    #     image=image,
    #     token_scores=token_scores,
    #     selected_indices=selected_indices,
    #     patch_size=patch_size,
    #     save_path="token_heatmap.png",
    # )

    # print("\n可视化完成！")
    # print("输出文件:")
    # print("- selected_tokens.png: 显示选择的token区域")
    # print("- token_heatmap.png: 显示token分数热力图")
