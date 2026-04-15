import numpy as np
import time
from .shuffling import Shuffler


class WatermarkExtractor:
    def __init__(self, block_size=8):
        self.block_size = block_size
        self.shuffler = Shuffler()

    def extract(self, watermarked_img, original_shape=None, key=None,
                shuffling_enabled=True, embedded_blocks=None,
                expected_length=None):
        start_time = time.time()
        test_img = watermarked_img.copy()
        h, w = test_img.shape

        if original_shape is not None and original_shape != (h, w):
            # 假设图像中心是原始图像
            oh, ow = original_shape
            start_h = (h - oh) // 2
            start_w = (w - ow) // 2
            if start_h >= 0 and start_w >= 0:
                test_img = test_img[start_h:start_h+oh, start_w:start_w+ow]
                h, w = test_img.shape

        # 如果启用了混洗，应用相同的混洗
        if shuffling_enabled and key is not None:
            forward_map, _ = self.shuffler.generate_maps((h, w), key)
            shuffled_test = self.shuffler.shuffle(test_img, forward_map)
        else:
            shuffled_test = test_img

        # 提取比特
        extracted_bits = []
        if embedded_blocks is not None and len(embedded_blocks) > 0:
            # 使用嵌入时记录的块位置
            for i, j in embedded_blocks:
                if i + self.block_size <= h and j + self.block_size <= w:
                    block = shuffled_test[i:i +
                                          self.block_size, j:j+self.block_size]
                    parity = (np.sum(block == 0)) % 2
                    extracted_bits.append(parity)
        else:
            max_blocks_h = h // self.block_size
            max_blocks_w = w // self.block_size

            for i in range(0, max_blocks_h * self.block_size, self.block_size):
                for j in range(0, max_blocks_w * self.block_size, self.block_size):
                    if expected_length and len(extracted_bits) >= expected_length:
                        break

                    if i + self.block_size <= h and j + self.block_size <= w:
                        block = shuffled_test[i:i +
                                              self.block_size, j:j+self.block_size]
                        parity = (np.sum(block == 0)) % 2
                        extracted_bits.append(parity)

        # 限制提取的比特数
        if expected_length is not None:
            extracted_bits = extracted_bits[:expected_length]

        duration = time.time() - start_time

        stats = {
            'extracted_length': len(extracted_bits),
            'time': duration
        }

        return np.array(extracted_bits, dtype=int), stats

    def verify(self, extracted_bits, original_bits):
        from utils.watermark_utils import calculate_ber, calculate_nc
        min_len = min(len(extracted_bits), len(original_bits))
        if min_len == 0:
            return 0.0, 1.0, 0.0

        correct = np.sum(extracted_bits[:min_len] == original_bits[:min_len])
        accuracy = correct / min_len * 100

        ber = calculate_ber(extracted_bits, original_bits)
        nc = calculate_nc(extracted_bits, original_bits)

        return accuracy, ber, nc
