import cv2
import numpy as np
from PIL import Image
import os


def load_image(filepath):
    try:
        raw_data = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(raw_data, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            return binary_img
        return None
    except Exception as e:
        print(f"加载图像出错: {e}")
        return None


def save_image(img, filepath):
    try:
        cv2.imwrite(filepath, img)
        return True
    except Exception as e:
        print(f"保存图像出错: {e}")
        return False


def compare_images(img1, img2):
    if img1.shape != img2.shape:
        # 调整大小使其一致
        h, w = min(img1.shape[0], img2.shape[0]), min(
            img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]

    diff = cv2.absdiff(img1, img2)
    changed_pixels = np.sum(diff > 0)
    total_pixels = img1.size
    change_percentage = (changed_pixels / total_pixels) * 100

    diff_enhanced = np.zeros_like(diff)
    diff_enhanced[diff > 0] = 255

    return {
        'diff': diff_enhanced,
        'changed_pixels': changed_pixels,
        'total_pixels': total_pixels,
        'change_percentage': change_percentage
    }


def create_comparison_view(images, labels=None):
    if not images:
        return None

    target_h = max(img.shape[0] for img in images)
    target_w = max(img.shape[1] for img in images)

    resized_images = []
    for img in images:
        if img.shape[:2] != (target_h, target_w):
            img = cv2.resize(img, (target_w, target_h),
                             interpolation=cv2.INTER_NEAREST)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        resized_images.append(img)

    separator = np.ones((target_h, 10, 3), dtype=np.uint8) * 255
    result = resized_images[0]
    for i in range(1, len(resized_images)):
        result = np.hstack([result, separator, resized_images[i]])

    return result
