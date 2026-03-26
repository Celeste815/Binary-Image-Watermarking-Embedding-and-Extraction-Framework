from pathlib import Path
import time
import sys
import os
import cv2
import numpy as np
from core.embedding import WatermarkEmbedder
from core.extraction import WatermarkExtractor
from core.flippability import FlippabilityCalculator
from core.shuffling import Shuffler
from utils.image_utils import load_image, save_image, compare_images
from utils.watermark_utils import text_to_bits, bits_to_text, calculate_ber, calculate_nc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_test_image(size=(256, 256), pattern='random'):
    h, w = size

    if pattern == 'random':
        img = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255

    elif pattern == 'checkerboard':
        img = np.zeros((h, w), dtype=np.uint8)
        block_size = 16
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                if (i // block_size + j // block_size) % 2 == 0:
                    img[i:min(i+block_size, h), j:min(j+block_size, w)] = 255

    elif pattern == 'text':
        img = np.ones((h, w), dtype=np.uint8) * 255
        line_height = 20
        for i in range(line_height, h, line_height * 2):
            img[i:i+2, :] = 0

    elif pattern == 'qrcode':
        img = np.ones((h, w), dtype=np.uint8) * 255
        for x, y in [(20, 20), (20, w-50), (h-50, 20)]:
            img[y:y+7, x:x+7] = 0
            img[y+2:y+5, x+2:x+5] = 255
            img[y+1:y+6, x+1:x+6] = 0

    else:
        img = np.ones((h, w), dtype=np.uint8) * 255
        img[50:150, 50:150] = 0
        img[100:200, 100:200] = 255

    return img


def test_flippability_calculation():
    """测试可翻转性计算"""
    print("\n" + "="*60)
    print("测试1: 可翻转性计算")
    print("="*60)

    calculator = FlippabilityCalculator()
    test_patterns = [
        # 均匀区域
        np.ones((3, 3), dtype=np.uint8) * 255,
        np.zeros((3, 3), dtype=np.uint8),

        # 孤立点
        np.array([
            [255, 255, 255],
            [255, 0, 255],
            [255, 255, 255]
        ], dtype=np.uint8),

        # 边缘
        np.array([
            [255, 255, 255],
            [0, 0, 0],
            [255, 255, 255]
        ], dtype=np.uint8),

        # 角点
        np.array([
            [0, 255, 255],
            [255, 255, 255],
            [255, 255, 255]
        ], dtype=np.uint8),

        # 复杂纹理
        np.array([
            [0, 255, 0],
            [255, 0, 255],
            [0, 255, 0]
        ], dtype=np.uint8)
    ]

    results = []

    for i, pattern in enumerate(test_patterns):
        score = calculator.get_score(pattern)
        print(f"\n模式 {i+1}:")
        print(pattern)
        print(f"可翻转性分数: {score:.3f}")
        results.append(score)

    # 生成测试图像并计算整体可翻转性图
    test_img = generate_test_image((128, 128), 'text')
    flip_map = calculator.compute_map(test_img)

    flippable, ratio = calculator.get_flippable_pixels()
    print(f"\n测试图像可翻转性统计:")
    print(f"  平均分数: {np.mean(flip_map):.3f}")
    print(f"  可翻转像素: {flippable} ({ratio*100:.1f}%)")

    return results


def test_embedding_extraction():
    """测试水印嵌入和提取"""
    print("\n" + "="*60)
    print("测试2: 水印嵌入与提取")
    print("="*60)

    test_img = generate_test_image((256, 256), 'checkerboard')
    watermark_text = "TEST123"
    watermark_bits = text_to_bits(watermark_text)

    print(f"原始水印: {watermark_text}")
    print(f"水印比特: {watermark_bits}")

    embedder = WatermarkEmbedder(block_size=8, min_flip_score=0.3)
    extractor = WatermarkExtractor(block_size=8)

    test_cases = [
        {"shuffling": False, "key": 12345, "desc": "无混洗"},
        {"shuffling": True, "key": 12345, "desc": "有混洗"},
        {"shuffling": True, "key": 67890, "desc": "不同密钥"}
    ]

    results = []

    for case in test_cases:
        print(f"\n--- 测试: {case['desc']} ---")
        watermarked, stats, blocks, flipped = embedder.embed(
            test_img,
            watermark_bits,
            key=case['key'],
            shuffling_enabled=case['shuffling'],
            return_detailed=True
        )
        print(f"嵌入统计: {stats}")

        extracted_bits, ext_stats = extractor.extract(
            watermarked,
            key=case['key'],
            shuffling_enabled=case['shuffling'],
            embedded_blocks=blocks,
            expected_length=len(watermark_bits)
        )

        accuracy, ber, nc = extractor.verify(extracted_bits, watermark_bits)
        extracted_text = bits_to_text(extracted_bits)

        print(f"提取水印: {extracted_text}")
        print(f"准确率: {accuracy:.2f}%, BER: {ber:.4f}, NC: {nc:.4f}")

        results.append({
            'case': case['desc'],
            'accuracy': accuracy,
            'ber': ber,
            'nc': nc,
            'extracted': extracted_text
        })

    return results


def test_multiple_watermarks():
    """测试多种水印"""
    print("\n" + "="*60)
    print("测试3: 多种水印测试")
    print("="*60)
    test_img = generate_test_image((256, 256), 'random')
    watermarks = [
        "A",
        "Hello",
        "TEST123456",
        "中文水印",
        "!@#$%^&*()"
    ]

    embedder = WatermarkEmbedder(block_size=8)
    extractor = WatermarkExtractor(block_size=8)
    key = 2251538
    results = []

    for wm in watermarks:
        print(f"\n测试水印: '{wm}'")
        bits = text_to_bits(wm)
        print(f"比特数: {len(bits)}")

        watermarked, stats, blocks, _ = embedder.embed(
            test_img, bits, key=key, shuffling_enabled=True, return_detailed=True
        )
        print(f"嵌入容量: {stats['capacity']} bits")

        extracted_bits, _ = extractor.extract(
            watermarked, key=key, shuffling_enabled=True,
            embedded_blocks=blocks, expected_length=len(bits)
        )

        accuracy, ber, nc = extractor.verify(extracted_bits, bits)
        extracted_text = bits_to_text(extracted_bits)

        print(f"提取结果: {extracted_text}")
        print(f"准确率: {accuracy:.2f}%")
        results.append({
            'watermark': wm,
            'accuracy': accuracy,
            'ber': ber,
            'nc': nc
        })
    return results


def test_capacity_analysis():
    """测试嵌入容量分析"""
    print("\n" + "="*60)
    print("测试4: 嵌入容量分析")
    print("="*60)

    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    patterns = ['random', 'checkerboard', 'text']
    embedder = WatermarkEmbedder(block_size=8)
    results = {}

    for pattern in patterns:
        print(f"\n图案类型: {pattern}")
        results[pattern] = {}
        for size in sizes:
            img = generate_test_image(size, pattern)
            capacity = embedder.get_embedding_capacity(img)
            test_bits = np.ones(capacity, dtype=int)  # 全1水印
            watermarked, stats, _, _ = embedder.embed(
                img, test_bits, key=12345, shuffling_enabled=True, return_detailed=True
            )

            actual_capacity = stats['capacity']
            print(
                f"  尺寸 {size[0]}x{size[1]}: 估算容量={capacity}, 实际容量={actual_capacity}")
            results[pattern][str(size)] = {
                'estimated': capacity,
                'actual': actual_capacity,
                'density': actual_capacity / (size[0] * size[1]) * 1000
            }
    return results


def test_shuffling_consistency():
    """测试混洗一致性"""
    print("\n" + "="*60)
    print("测试5: 混洗一致性测试")
    print("="*60)

    from core.shuffling import Shuffler
    test_img = generate_test_image((64, 64), 'random')
    shuffler = Shuffler()

    # 测试1: 同一密钥应产生相同混洗结果
    key = 12345
    shuffled1 = shuffler.shuffle_with_key(test_img, key)
    shuffled2 = shuffler.shuffle_with_key(test_img, key)
    same = np.array_equal(shuffled1, shuffled2)
    print(f"同一密钥多次混洗结果相同: {same}")

    # 测试2: 不同密钥应产生不同结果
    shuffled3 = shuffler.shuffle_with_key(test_img, 54321)
    different = not np.array_equal(shuffled1, shuffled3)
    print(f"不同密钥混洗结果不同: {different}")

    # 测试3: 反向混洗应恢复原图
    forward_map, inverse_map = shuffler.generate_maps(test_img.shape, key)
    shuffled = shuffler.shuffle(test_img, forward_map)
    restored = shuffler.unshuffle(shuffled, inverse_map)
    restored_correct = np.array_equal(test_img, restored)
    print(f"反向混洗恢复原图: {restored_correct}")

    return {
        'same_key_consistency': same,
        'different_key_uniqueness': different,
        'reversibility': restored_correct
    }


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("二值图像水印系统——完整测试")
    print("="*60)
    all_results = {}

    try:
        # 测试1: 可翻转性计算
        all_results['flippability'] = test_flippability_calculation()
    except Exception as e:
        print(f"可翻转性测试失败: {e}")
        all_results['flippability'] = {'error': str(e)}

    try:
        # 测试2: 嵌入提取
        all_results['embedding_extraction'] = test_embedding_extraction()
    except Exception as e:
        print(f"嵌入提取测试失败: {e}")
        all_results['embedding_extraction'] = {'error': str(e)}

    try:
        # 测试3: 多种水印
        all_results['multiple_watermarks'] = test_multiple_watermarks()
    except Exception as e:
        print(f"多种水印测试失败: {e}")
        all_results['multiple_watermarks'] = {'error': str(e)}

    try:
        # 测试4: 容量分析
        all_results['capacity_analysis'] = test_capacity_analysis()
    except Exception as e:
        print(f"容量分析测试失败: {e}")
        all_results['capacity_analysis'] = {'error': str(e)}

    try:
        # 测试5: 混洗一致性
        all_results['shuffling_consistency'] = test_shuffling_consistency()
    except Exception as e:
        print(f"混洗一致性测试失败: {e}")
        all_results['shuffling_consistency'] = {'error': str(e)}

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    return all_results


def generate_algorithm_report(results, output_path="algorithm_report.txt"):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("二值图像水印系统——算法测试报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        # 测试1: 可翻转性计算
        f.write("【测试1: 可翻转性计算】\n")
        f.write("-"*60 + "\n")
        if 'flippability' in results:
            if isinstance(results['flippability'], dict) and 'error' in results['flippability']:
                f.write(f"测试失败: {results['flippability']['error']}\n")
            elif isinstance(results['flippability'], list):
                for i, score in enumerate(results['flippability']):
                    f.write(f"  模式 {i+1} 可翻转性分数: {score:.3f}\n")
                f.write(f"\n  平均分数: {np.mean(results['flippability']):.3f}\n")
                f.write(f"  标准差: {np.std(results['flippability']):.3f}\n")
        f.write("\n")

        # 测试2: 水印嵌入与提取
        f.write("【测试2: 水印嵌入与提取】\n")
        f.write("-"*60 + "\n")
        if 'embedding_extraction' in results:
            if isinstance(results['embedding_extraction'], dict) and 'error' in results['embedding_extraction']:
                f.write(f"测试失败: {results['embedding_extraction']['error']}\n")
            elif isinstance(results['embedding_extraction'], list):
                for item in results['embedding_extraction']:
                    f.write(f"\n  测试场景: {item['case']}\n")
                    f.write(f"    准确率: {item['accuracy']:.2f}%\n")
                    f.write(f"    BER: {item['ber']:.4f}\n")
                    f.write(f"    NC: {item['nc']:.4f}\n")
                    f.write(f"    提取结果: {item['extracted']}\n")
        f.write("\n")

        # 测试3: 多种水印测试
        f.write("【测试3: 多种水印测试】\n")
        f.write("-"*60 + "\n")
        if 'multiple_watermarks' in results:
            if isinstance(results['multiple_watermarks'], dict) and 'error' in results['multiple_watermarks']:
                f.write(f"测试失败: {results['multiple_watermarks']['error']}\n")
            elif isinstance(results['multiple_watermarks'], list):
                f.write("\n  水印类型测试结果:\n")
                for item in results['multiple_watermarks']:
                    f.write(f"\n    水印内容: {item['watermark']}\n")
                    f.write(f"    准确率: {item['accuracy']:.2f}%\n")
                    f.write(f"    BER: {item['ber']:.4f}\n")
                    f.write(f"    NC: {item['nc']:.4f}\n")
        f.write("\n")

        # 测试4: 嵌入容量分析
        f.write("【测试4: 嵌入容量分析】\n")
        f.write("-"*60 + "\n")
        if 'capacity_analysis' in results:
            if isinstance(results['capacity_analysis'], dict) and 'error' in results['capacity_analysis']:
                f.write(f"测试失败: {results['capacity_analysis']['error']}\n")
            else:
                for pattern, sizes in results['capacity_analysis'].items():
                    f.write(f"\n  图案类型: {pattern}\n")
                    f.write("  " + "-"*40 + "\n")
                    for size, data in sizes.items():
                        f.write(f"    尺寸 {size}:\n")
                        f.write(f"      估算容量: {data['estimated']} bits\n")
                        f.write(f"      实际容量: {data['actual']} bits\n")
                        f.write(
                            f"      嵌入密度: {data['density']:.2f} bits/千像素\n")
        f.write("\n")

        # 测试5: 混洗一致性测试
        f.write("【测试5: 混洗一致性测试】\n")
        f.write("-"*60 + "\n")
        if 'shuffling_consistency' in results:
            if isinstance(results['shuffling_consistency'], dict) and 'error' in results['shuffling_consistency']:
                f.write(f"测试失败: {results['shuffling_consistency']['error']}\n")
            else:
                consistency = results['shuffling_consistency']
                f.write(
                    f"\n  同一密钥多次混洗结果相同: {consistency.get('same_key_consistency', 'N/A')}\n")
                f.write(
                    f"  不同密钥混洗结果不同: {consistency.get('different_key_uniqueness', 'N/A')}\n")
                f.write(
                    f"  反向混洗恢复原图: {consistency.get('reversibility', 'N/A')}\n")
        f.write("\n")

        # 汇总统计
        f.write("="*80 + "\n")
        f.write("【测试汇总统计】\n")
        f.write("="*80 + "\n")

        total_tests = 0
        passed_tests = 0

        for test_name, test_result in results.items():
            total_tests += 1
            if not (isinstance(test_result, dict) and 'error' in test_result):
                passed_tests += 1
                status = "✓ 通过"
            else:
                status = "✗ 失败"
            f.write(f"{status} {test_name}\n")

        f.write(
            f"\n通过率: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)\n")

        f.write("\n" + "="*80 + "\n")
        f.write("测试报告结束\n")
        f.write("="*80 + "\n")

    print(f"\n算法测试报告已保存到: {output_path}")


def run_all_tests_with_report():
    results = run_all_tests()
    generate_algorithm_report(results)
    return results


if __name__ == "__main__":
    results = run_all_tests_with_report()
