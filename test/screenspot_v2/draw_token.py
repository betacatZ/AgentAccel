import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


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


def visualize_selected_tokens(
    image: Image.Image,
    selected_indices: List[int],
    patch_size: int = 16,
    image_size: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Image.Image:
    """
    可视化visionselector裁剪的token对应的图像区域

    Args:
        image: 原始图像
        selected_indices: 选择的token索引列表
        patch_size: 视觉token的patch大小（默认为16）
        image_size: 图像的原始大小，如果为None则使用图像的实际大小
        save_path: 保存可视化结果的路径
        show: 是否显示可视化结果

    Returns:
        可视化后的图像
    """
    # 调整图像大小
    if image_size is not None:
        # 如果指定了image_size，先调整到指定大小，再调整到patch_size的倍数
        image = image.resize(image_size)

    # 将图像调整到patch_size的倍数
    image, (width, height) = resize_to_patch_multiple(image, patch_size)

    # 将图像转换为RGBA模式以支持透明度
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)

    # 计算图像的patch数量
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    total_tokens = num_patches_h * num_patches_w

    # 创建集合以便快速查找选中的索引
    selected_set = set(selected_indices)

    # 绘制未被选中的token（半透明灰色覆盖）
    for idx in range(total_tokens):
        if idx not in selected_set:
            # 计算patch的行和列
            row = idx // num_patches_w
            col = idx % num_patches_w

            # 计算patch在图像上的坐标
            x1 = col * patch_size
            y1 = row * patch_size
            x2 = (col + 1) * patch_size
            y2 = (row + 1) * patch_size

            # 绘制半透明灰色矩形覆盖未选中区域
            draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 128), outline=None)

    # 被选中的token保持原样显示（不绘制红色边框）

    # 保存图像
    if save_path:
        image.save(save_path)

    # 显示图像
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Selected {len(selected_indices)} tokens out of {num_patches_h * num_patches_w}")
        plt.show()

    return image


def visualize_token_scores(
    image: Image.Image,
    token_scores: torch.Tensor,
    selected_indices: Optional[List[int]] = None,
    patch_size: int = 16,
    image_size: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[Image.Image, Image.Image]:
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
    scores = token_scores.cpu().numpy()
    heatmap = scores.reshape(num_patches_h, num_patches_w)

    # 调整热力图大小以匹配图像
    heatmap_resized = np.repeat(np.repeat(heatmap, patch_size, axis=0), patch_size, axis=1)

    # 如果有选择的token，将未选中的token设置为半透明
    if selected_indices:
        draw_copy = ImageDraw.Draw(image_copy)
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
                draw_copy.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 128), outline=None)

        # 被选中的token保持原样显示（不绘制红色边框）

    # 绘制热力图
    plt.figure(figsize=(20, 10))

    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image_copy)
    plt.axis("off")
    plt.title("Original Image")

    # 热力图
    plt.subplot(1, 2, 2)
    plt.imshow(image_copy)
    plt.imshow(heatmap_resized, cmap="jet", alpha=0.5)
    plt.colorbar(label="Token Importance Score")
    plt.axis("off")
    plt.title("Token Importance Heatmap")

    # 同样处理返回的图像
    if selected_indices:
        draw = ImageDraw.Draw(image)
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
                draw.rectangle([x1, y1, x2, y2], fill=(128, 128, 128, 128), outline=None)

        # 被选中的token保持原样显示（不绘制红色边框）

    # 保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    # 显示图像
    if show:
        plt.show()

    # 创建热力图叠加的图像
    heatmap_image = Image.fromarray((heatmap_resized * 255).astype(np.uint8))
    heatmap_image = heatmap_image.convert("RGB")
    overlay_image = Image.blend(image, heatmap_image, alpha=0.5)

    return heatmap_image, overlay_image


def get_token_patch_coordinates(
    token_idx: int, patch_size: int = 16, image_width: Optional[int] = None, image_height: Optional[int] = None
) -> Tuple[int, int, int, int]:
    """
    将token索引转换为图像上的patch坐标

    Args:
        token_idx: token索引
        patch_size: 视觉token的patch大小
        image_width: 图像宽度（如果为None，将根据token_idx和patch_size推断）
        image_height: 图像高度（如果为None，将根据token_idx和patch_size推断）

    Returns:
        patch的坐标 (x1, y1, x2, y2)
    """
    if image_width is None or image_height is None:
        # 如果没有提供图像大小，假设一个合理的默认值
        # 注意：这是一个退避方案，实际使用时应提供图像大小
        num_patches_w = 32  # 假设图像宽度为32个patch
    else:
        num_patches_w = image_width // patch_size

    row = token_idx // num_patches_w
    col = token_idx % num_patches_w

    x1 = col * patch_size
    y1 = row * patch_size
    x2 = (col + 1) * patch_size
    y2 = (row + 1) * patch_size

    return x1, y1, x2, y2


if __name__ == "__main__":
    # 示例用法
    import sys
    from PIL import Image
    import torch

    if len(sys.argv) < 2:
        print("Usage: python draw_token.py <image_path> [selected_indices]")
        sys.exit(1)

    # 加载图像
    image_path = sys.argv[1]
    image = Image.open(image_path)

    # 获取图像的实际大小
    original_width, original_height = image.size
    print(f"原始图像大小: {original_width}x{original_height}")

    # 使用默认的patch_size 16
    patch_size = 16
    # 计算调整后的图像大小（向上取整到patch_size的倍数）
    adjusted_width = ((original_width + patch_size - 1) // patch_size) * patch_size
    adjusted_height = ((original_height + patch_size - 1) // patch_size) * patch_size
    print(f"调整后的图像大小 (patch_size={patch_size}的倍数): {adjusted_width}x{adjusted_height}")

    # 示例token索引（实际使用时应从模型获取）
    if len(sys.argv) > 2:
        selected_indices = list(map(int, sys.argv[2].split(",")))
    else:
        # 随机选择一半的token作为示例
        total_tokens = (adjusted_height // patch_size) * (adjusted_width // patch_size)
        num_selected = total_tokens // 2  # 选择一半的token
        selected_indices = np.random.choice(total_tokens, num_selected, replace=False).tolist()
        selected_indices.sort()  # 保持索引顺序
        print(f"随机选择的token数量: {len(selected_indices)} (总token数量: {total_tokens})")
        print(f"选择的token索引: {selected_indices[:10]}...")

    # 示例token分数（实际使用时应从模型获取）
    total_tokens = (adjusted_height // patch_size) * (adjusted_width // patch_size)
    token_scores = torch.randn(total_tokens)
    token_scores = torch.softmax(token_scores, dim=0)

    # 可视化选择的token
    print("\n--- 可视化选择的token ---")
    visualize_selected_tokens(
        image=image, selected_indices=selected_indices, patch_size=patch_size, save_path="selected_tokens.png"
    )

    # 可视化token分数热力图
    print("\n--- 可视化token分数热力图 ---")
    visualize_token_scores(
        image=image,
        token_scores=token_scores,
        selected_indices=selected_indices,
        patch_size=patch_size,
        save_path="token_heatmap.png",
    )

    print("\n可视化完成！")
    print("输出文件:")
    print("- selected_tokens.png: 显示选择的token区域")
    print("- token_heatmap.png: 显示token分数热力图")
