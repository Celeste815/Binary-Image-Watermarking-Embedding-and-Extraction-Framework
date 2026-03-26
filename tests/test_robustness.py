from tests.test_algorithm import generate_test_image
from utils.watermark_utils import text_to_bits, bits_to_text, calculate_ber, calculate_nc
from utils.image_utils import load_image, save_image, compare_images
from core.extraction import WatermarkExtractor
from core.embedding import WatermarkEmbedder
import numpy as np
import cv2
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_noise(img, noise_level=0.05):
    # 添加椒盐噪声
    noisy = img.copy()
    h, w = noisy.shape
    num_noise = int(h * w * noise_level)

    rows = np.random.randint(0, h, num_noise)
    cols = np.random.randint(0, w, num_noise)

    for i in range(num_noise):
        noisy[rows[i], cols[i]] = 255 - noisy[rows[i], cols[i]]

    return noisy


def rotate_image(img, angle=1.0):
    # 旋转图像
    h, w = img.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h),
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return rotated


def scale_image(img, scale_factor=0.9):
    # 缩放图像
    h, w = img.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    if scale_factor < 1:
        result = np.ones((h, w), dtype=np.uint8) * 255
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        result[start_h:start_h+new_h, start_w:start_w+new_w] = scaled
        return result
    else:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return scaled[start_h:start_h+h, start_w:start_w+w]


def translate_image(img, tx=5, ty=5):
    # 平移图像
    h, w = img.shape
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, translation_matrix, (w, h),
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return translated


def compress_image(img, quality=50):
    # 模拟JPEG压缩
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(decoded, 127, 255, cv2.THRESH_BINARY)
    return binary


def crop_image(img, crop_ratio=0.05):
    # 裁剪图像边缘
    h, w = img.shape
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    cropped = img[crop_h:h-crop_h, crop_w:w-crop_w]

    # 填充回原尺寸
    result = np.ones((h, w), dtype=np.uint8) * 255
    result[crop_h:h-crop_h, crop_w:w-crop_w] = cropped
    return result


def test_noise_robustness():
    # 测试对噪声的鲁棒性
    print("\n" + "="*60)
    print("鲁棒性测试1: 噪声攻击")
    print("="*60)

    test_img = generate_test_image((256, 256), 'text')
    watermark_text = "ROBUSTNESS_TEST"
    watermark_bits = text_to_bits(watermark_text)

    embedder = WatermarkEmbedder(block_size=8)
    extractor = WatermarkExtractor(block_size=8)
    key = 2251538

    watermarked, stats, blocks, _ = embedder.embed(
        test_img, watermark_bits, key=key, shuffling_enabled=True, return_detailed=True
    )
    print(f"原始嵌入容量: {stats['capacity']} bits")

    # 测试不同噪声水平
    noise_levels = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]
    results = []

    for noise in noise_levels:
        attacked = add_noise(watermarked, noise)
        extracted_bits, _ = extractor.extract(
            attacked, key=key, shuffling_enabled=True,
            embedded_blocks=blocks, expected_length=len(watermark_bits)
        )
        accuracy, ber, nc = extractor.verify(extracted_bits, watermark_bits)
        print(
            f"噪声水平 {noise:.2f}: 准确率={accuracy:.2f}%, BER={ber:.4f}, NC={nc:.4f}")
        results.append({
            'noise_level': noise,
            'accuracy': accuracy,
            'ber': ber,
            'nc': nc
        })
    return results


def test_rotation_robustness():
    # 测试对旋转的鲁棒性
    print("\n" + "="*60)
    print("鲁棒性测试2: 旋转攻击")
    print("="*60)

    test_img = generate_test_image((256, 256), 'checkerboard')
    watermark_text = "ROTATION_TEST"
    watermark_bits = text_to_bits(watermark_text)

    embedder = WatermarkEmbedder(block_size=8)
    extractor = WatermarkExtractor(block_size=8)
    key = 2251538

    watermarked, stats, blocks, _ = embedder.embed(
        test_img, watermark_bits, key=key, shuffling_enabled=True, return_detailed=True
    )

    # 测试不同旋转角度
    angles = [0.5, 1.0, 2.0, 3.0, 5.0]
    results = []

    for angle in angles:
        attacked = rotate_image(watermarked, angle)
        extracted_bits, _ = extractor.extract(
            attacked, key=key, shuffling_enabled=True,
            embedded_blocks=blocks, expected_length=len(watermark_bits)
        )
        accuracy, ber, nc = extractor.verify(extracted_bits, watermark_bits)
        print(f"旋转角度 {angle}°: 准确率={accuracy:.2f}%, BER={ber:.4f}, NC={nc:.4f}")
        results.append({
            'angle': angle,
            'accuracy': accuracy,
            'ber': ber,
            'nc': nc
        })
    return results


def test_scaling_robustness():
    # 测试对缩放的鲁棒性
    print("\n" + "="*60)
    print("鲁棒性测试3: 缩放攻击")
    print("="*60)

    test_img = generate_test_image((256, 256), 'text')
    watermark_text = "SCALING_TEST"
    watermark_bits = text_to_bits(watermark_text)

    embedder = WatermarkEmbedder(block_size=8)
    extractor = WatermarkExtractor(block_size=8)
    key = 2251538

    watermarked, stats, blocks, _ = embedder.embed(
        test_img, watermark_bits, key=key, shuffling_enabled=True, return_detailed=True
    )

    # 测试不同缩放因子
    scales = [0.8, 0.9, 0.95, 1.05, 1.1, 1.2]
    results = []

    for scale in scales:
        attacked = scale_image(watermarked, scale)
        extracted_bits, _ = extractor.extract(
            attacked, original_shape=test_img.shape if scale != 1.0 else None,
            key=key, shuffling_enabled=True,
            embedded_blocks=blocks, expected_length=len(watermark_bits)
        )
        accuracy, ber, nc = extractor.verify(extracted_bits, watermark_bits)
        print(
            f"缩放因子 {scale:.2f}: 准确率={accuracy:.2f}%, BER={ber:.4f}, NC={nc:.4f}")
        results.append({
            'scale': scale,
            'accuracy': accuracy,
            'ber': ber,
            'nc': nc
        })
    return results


def test_translation_robustness():
    # 测试对平移的鲁棒性
    print("\n" + "="*60)
    print("鲁棒性测试4: 平移攻击")
    print("="*60)

    test_img = generate_test_image((256, 256), 'random')
    watermark_text = "TRANSLATION_TEST"
    watermark_bits = text_to_bits(watermark_text)

    embedder = WatermarkEmbedder(block_size=8)
    extractor = WatermarkExtractor(block_size=8)
    key = 2251538

    watermarked, stats, blocks, _ = embedder.embed(
        test_img, watermark_bits, key=key, shuffling_enabled=True, return_detailed=True
    )

    # 测试不同平移量
    translations = [(2, 2), (5, 5), (10, 10), (20, 20)]
    results = []

    for tx, ty in translations:
        attacked = translate_image(watermarked, tx, ty)
        extracted_bits, _ = extractor.extract(
            attacked, original_shape=test_img.shape,
            key=key, shuffling_enabled=True,
            embedded_blocks=blocks, expected_length=len(watermark_bits)
        )
        accuracy, ber, nc = extractor.verify(extracted_bits, watermark_bits)
        print(f"平移 ({tx}, {ty}): 准确率={accuracy:.2f}%, BER={ber:.4f}, NC={nc:.4f}")
        results.append({
            'translation': (tx, ty),
            'accuracy': accuracy,
            'ber': ber,
            'nc': nc
        })
    return results


def test_jpeg_robustness():
    # 测试对JPEG压缩的鲁棒性
    print("\n" + "="*60)
    print("鲁棒性测试5: JPEG压缩攻击")
    print("="*60)

    test_img = generate_test_image((256, 256), 'text')
    watermark_text = "JPEG_TEST"
    watermark_bits = text_to_bits(watermark_text)

    embedder = WatermarkEmbedder(block_size=8)
    extractor = WatermarkExtractor(block_size=8)
    key = 2251538

    watermarked, stats, blocks, _ = embedder.embed(
        test_img, watermark_bits, key=key, shuffling_enabled=True, return_detailed=True
    )

    # 测试不同压缩质量
    qualities = [90, 75, 50, 30, 20, 10]
    results = []

    for quality in qualities:
        attacked = compress_image(watermarked, quality)
        extracted_bits, _ = extractor.extract(
            attacked, key=key, shuffling_enabled=True,
            embedded_blocks=blocks, expected_length=len(watermark_bits)
        )
        accuracy, ber, nc = extractor.verify(extracted_bits, watermark_bits)
        print(
            f"JPEG质量 {quality}: 准确率={accuracy:.2f}%, BER={ber:.4f}, NC={nc:.4f}")
        results.append({
            'quality': quality,
            'accuracy': accuracy,
            'ber': ber,
            'nc': nc
        })
    return results


def test_cropping_robustness():
    # 测试对裁剪的鲁棒性
    print("\n" + "="*60)
    print("鲁棒性测试6: 裁剪攻击")
    print("="*60)

    test_img = generate_test_image((256, 256), 'checkerboard')
    watermark_text = "CROPPING_TEST"
    watermark_bits = text_to_bits(watermark_text)

    embedder = WatermarkEmbedder(block_size=8)
    extractor = WatermarkExtractor(block_size=8)
    key = 2251538

    watermarked, stats, blocks, _ = embedder.embed(
        test_img, watermark_bits, key=key, shuffling_enabled=True, return_detailed=True
    )

    # 测试不同裁剪比例
    crop_ratios = [0.02, 0.05, 0.08, 0.10, 0.12]
    results = []

    for ratio in crop_ratios:
        attacked = crop_image(watermarked, ratio)
        extracted_bits, _ = extractor.extract(
            attacked, original_shape=test_img.shape,
            key=key, shuffling_enabled=True,
            embedded_blocks=blocks, expected_length=len(watermark_bits)
        )
        accuracy, ber, nc = extractor.verify(extracted_bits, watermark_bits)
        print(
            f"裁剪比例 {ratio:.2f}: 准确率={accuracy:.2f}%, BER={ber:.4f}, NC={nc:.4f}")
        results.append({
            'crop_ratio': ratio,
            'accuracy': accuracy,
            'ber': ber,
            'nc': nc
        })
    return results


def test_combined_attacks():
    # 测试组合攻击
    print("\n" + "="*60)
    print("鲁棒性测试7: 组合攻击")
    print("="*60)

    test_img = generate_test_image((256, 256), 'text')
    watermark_text = "COMBINED_TEST"
    watermark_bits = text_to_bits(watermark_text)

    embedder = WatermarkEmbedder(block_size=8)
    extractor = WatermarkExtractor(block_size=8)
    key = 2251538

    watermarked, stats, blocks, _ = embedder.embed(
        test_img, watermark_bits, key=key, shuffling_enabled=True, return_detailed=True
    )

    attack_combinations = [
        {"name": "轻度组合", "attacks": [
            ("噪声", lambda x: add_noise(x, 0.02)),
            ("缩放", lambda x: scale_image(x, 0.98)),
            ("平移", lambda x: translate_image(x, 2, 2))
        ]},
        {"name": "中度组合", "attacks": [
            ("噪声", lambda x: add_noise(x, 0.05)),
            ("旋转", lambda x: rotate_image(x, 1.0)),
            ("缩放", lambda x: scale_image(x, 0.95)),
            ("平移", lambda x: translate_image(x, 5, 5))
        ]},
        {"name": "重度组合", "attacks": [
            ("噪声", lambda x: add_noise(x, 0.08)),
            ("旋转", lambda x: rotate_image(x, 2.0)),
            ("缩放", lambda x: scale_image(x, 0.9)),
            ("平移", lambda x: translate_image(x, 8, 8)),
            ("JPEG", lambda x: compress_image(x, 50))
        ]}
    ]
    results = []

    for combo in attack_combinations:
        print(f"\n--- {combo['name']} ---")
        attacked = watermarked.copy()

        for attack_name, attack_func in combo['attacks']:
            attacked = attack_func(attacked)
            print(f"  应用攻击: {attack_name}")

        extracted_bits, _ = extractor.extract(
            attacked, original_shape=test_img.shape,
            key=key, shuffling_enabled=True,
            embedded_blocks=blocks, expected_length=len(watermark_bits)
        )
        accuracy, ber, nc = extractor.verify(extracted_bits, watermark_bits)
        print(f"提取准确率: {accuracy:.2f}%, BER={ber:.4f}, NC={nc:.4f}")
        results.append({
            'combination': combo['name'],
            'accuracy': accuracy,
            'ber': ber,
            'nc': nc
        })
    return results


def run_all_robustness_tests():
    print("\n" + "="*60)
    print("二值图像水印系统——鲁棒性测试")
    print("="*60)

    all_results = {}
    try:
        all_results['noise'] = test_noise_robustness()
    except Exception as e:
        print(f"噪声测试失败: {e}")
        all_results['noise'] = {'error': str(e)}

    try:
        all_results['rotation'] = test_rotation_robustness()
    except Exception as e:
        print(f"旋转测试失败: {e}")
        all_results['rotation'] = {'error': str(e)}

    try:
        all_results['scaling'] = test_scaling_robustness()
    except Exception as e:
        print(f"缩放测试失败: {e}")
        all_results['scaling'] = {'error': str(e)}

    try:
        all_results['translation'] = test_translation_robustness()
    except Exception as e:
        print(f"平移测试失败: {e}")
        all_results['translation'] = {'error': str(e)}

    try:
        all_results['jpeg'] = test_jpeg_robustness()
    except Exception as e:
        print(f"JPEG测试失败: {e}")
        all_results['jpeg'] = {'error': str(e)}

    try:
        all_results['cropping'] = test_cropping_robustness()
    except Exception as e:
        print(f"裁剪测试失败: {e}")
        all_results['cropping'] = {'error': str(e)}

    try:
        all_results['combined'] = test_combined_attacks()
    except Exception as e:
        print(f"组合攻击测试失败: {e}")
        all_results['combined'] = {'error': str(e)}

    print("\n" + "="*60)
    print("鲁棒性测试完成!")
    print("="*60)

    return all_results


def generate_robustness_report(results, output_path="robustness_report.txt"):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("二值图像水印系统——鲁棒性测试报告\n")
        f.write("="*80 + "\n\n")

        for test_name, test_results in results.items():
            f.write(f"\n【{test_name.upper()} 测试】\n")
            f.write("-"*40 + "\n")

            if isinstance(test_results, dict) and 'error' in test_results:
                f.write(f"测试失败: {test_results['error']}\n")
            elif isinstance(test_results, list):
                for item in test_results:
                    f.write(str(item) + "\n")
            else:
                f.write(str(test_results) + "\n")

        f.write("\n" + "="*80 + "\n")
        f.write("报告生成时间: " + str(np.datetime64('now')) + "\n")

    print(f"\n鲁棒性测试报告已保存到: {output_path}")


if __name__ == "__main__":
    results = run_all_robustness_tests()
    generate_robustness_report(results)
