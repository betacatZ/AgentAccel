import argparse
import json
import logging
import os
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm
import yaml

from tester import Qwen3VLTester


def parse_args():
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument("--config", "-c", type=str, default=None, help="path to config.yaml")
    parser.add_argument("--output", type=str, default="test/screenspot_v2/output/")
    args = parser.parse_args()
    return args


def _normalize_bbox(bbox: List[float], img_size: Tuple[int, int]) -> List[float]:
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    W, H = img_size
    return [x / W, y / H, x2 / W, y2 / H]


def evaluate(tester, dataset_path: str, task: str) -> Tuple[Dict[str, float], List[Dict]]:
    tasks_result: Dict[str, float] = {}
    results: List[Dict] = []

    dataset_file = os.path.join(dataset_path, f"screenspot_{task}_v2.json")
    with open(dataset_file, "r") as f:
        screenspot_data = json.load(f)

    num_action = 0
    corr_action = 0
    text_correct: List[int] = []
    icon_correct: List[int] = []
    num_wrong_format = 0

    for j, item in tqdm(enumerate(screenspot_data), total=len(screenspot_data)):
        num_action += 1

        filename = item["img_filename"]
        img_path = os.path.join(dataset_path, "screenspotv2_image", filename)
        if not os.path.exists(img_path):
            logging.info("img not found: %s", img_path)
            num_wrong_format += 1
            if item["data_type"] == "text":
                text_correct.append(0)
            else:
                icon_correct.append(0)
            continue

        image = Image.open(img_path)
        instruction = item["instruction"]
        bbox_norm = _normalize_bbox(item["bbox"], image.size)

        # click_point, response = tester.generate_click_coordinate(instruction, image)
        click_point, response = tester.generate_click_coordinate(instruction, image)
        correct = False
        if click_point is None:
            num_wrong_format += 1
            if item["data_type"] == "text":
                text_correct.append(0)
            else:
                icon_correct.append(0)
            tqdm.write("Step: %s wrong format" % str(j))
        else:
            x1, y1, x2, y2 = bbox_norm
            correct = x1 <= click_point[0] <= x2 and y1 <= click_point[1] <= y2
            if correct:
                corr_action += 1
                if item["data_type"] == "text":
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
                tqdm.write("match %.6f" % (corr_action / max(num_action, 1)))
            else:
                if item["data_type"] == "text":
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                tqdm.write("unmatch %.6f" % (corr_action / max(num_action, 1)))

        results.append(
            {
                "img_path": img_path,
                "text": instruction,
                "bbox": bbox_norm,
                "pred": click_point,
                "respose": response,
                "type": item["data_type"],
                "source": item["data_source"],
                "correct": correct,
            }
        )
    action_acc = corr_action / max(num_action, 1)
    logging.info("Action Acc: %.6f", action_acc)
    logging.info("Total num: %d", num_action)
    logging.info("Wrong format num: %d", num_wrong_format)
    text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0.0
    icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0.0
    logging.info("Text Acc: %.6f", text_acc)
    logging.info("Icon Acc: %.6f", icon_acc)

    tasks_result["action_acc"] = action_acc
    tasks_result["text_acc"] = text_acc
    tasks_result["icon_acc"] = icon_acc
    tasks_result["total_num"] = num_action
    tasks_result["wrong_format_num"] = num_wrong_format
    return tasks_result, results



def run(tester, dataset_path: str, task: str, output: str):
    tasks = ["mobile", "desktop", "web"] if task == "all" else [task]
    for t in tasks:
        tasks_result, results = evaluate(tester, dataset_path, t)
        output_json = os.path.join(output, f"screenspot_v2_{t}_result.json")
        output_detail_json = os.path.join(output, f"screenspot_v2_{t}_detail.json")
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(tasks_result, f, ensure_ascii=False, indent=4)
        with open(output_detail_json, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    cfg = load_config(args.config)

    for key in ["model", "model_path", "dataset_path", "task"]:
        assert cfg.get(key) is not None, f"{key} must be specified in config"

    if cfg["model"].lower() == "qwen3vl":
        tester = Qwen3VLTester(cfg["model_path"])
    elif cfg["model"].lower() == "mobimind":
        from tester.MobiMind_tester import MobiMindTester

        tester = MobiMindTester(cfg["model_path"])
    elif cfg["model"].lower() == "qwen3vl_vision_selector":
        from tester.qwen3vl_visionselector_tester import Qwen3VLVisionSelectorTester

        tester = Qwen3VLVisionSelectorTester(cfg["model_path"], budgets=cfg["budgets"])
    else:
        raise ValueError(f"Unknown model: {cfg['model']}")
    exp_name = cfg.get("exp_name", "default_exp")
    output_path = os.path.join(args.output, exp_name)
    run(tester, cfg["dataset_path"], cfg["task"], output_path)

    with open(os.path.join(output_path, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)



if __name__ == "__main__":
    main()
