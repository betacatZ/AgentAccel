import json
import os
import argparse
from PIL import Image
from visualization_token import visualize_visionselector_tokens
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tester.qwen3vl_visionselector_tester import Qwen3VLVisionSelectorTester


def batch_visualize(
    tester: Qwen3VLVisionSelectorTester,
    data_list: list,
    output_dir: str,
    show: bool = False,
):
    """
    批量可视化

    Args:
        tester: Qwen3VLVisionSelectorTester实例
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

            # 可视化
            coordinates, response = visualize_visionselector_tokens(
                tester=tester,
                save_path=save_path,
                show=show,
                sample=item,
            )

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
    tester = Qwen3VLVisionSelectorTester(args.model_path, budgets=args.budgets)
    print("模型加载完成\n")

    # 批量可视化
    batch_visualize(
        tester=tester,
        data_list=data_list,
        output_dir=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
