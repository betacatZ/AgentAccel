from PIL import Image, ImageDraw
import os


def draw_bbox_and_pred(sample, save_path=None):
    """
    Draw bbox (red) and pred point (blue) on image.

    Args:
        sample (dict): one ScreenSpot v2 record
        save_path (str | None): if set, save visualized image
    """
    img = Image.open(sample["img_path"]).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    # ---------- bbox ----------
    x1, y1, x2, y2 = sample["bbox"]
    x1, y1 = x1 * W, y1 * H
    x2, y2 = x2 * W, y2 * H
    draw.rectangle([x1, y1, x2, y2], outline="blue", width=4)

    # ---------- pred point ----------
    px, py = sample["pred"]
    px, py = px * W, py * H
    r = 14
    draw.ellipse([px - r, py - r, px + r, py + r], fill="red", outline="red")

    if save_path is not None:
        img.save(save_path)

    return img


def main():
    # ---------- 示例 sample ----------
    sample = {
        "img_path": "/data8/zhangdeming/data/OS-Copilot/ScreenSpot-v2/screenspotv2_image/mobile_b550b4ed-79ce-446a-a1c3-a164b314bfe8.png",
        "text": "check out jony j's album",
        "bbox": [0.0452991452991453, 0.2713270142180095, 0.9487179487179487, 0.5501579778830964],
        "pred": [0.899, 0.935],
        "type": "icon",
        "source": "ios",
        "correct": False,
    }

    # ---------- 调用函数 ----------
    output_path = "test/screenspot_v2/visualization/output/"
    save_path = os.path.basename(sample["img_path"]).replace(".png", "_ground.png")
    img = draw_bbox_and_pred(sample, save_path=os.path.join(output_path, save_path))


if __name__ == "__main__":
    main()
