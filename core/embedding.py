import numpy as np
import time
from .flippability import FlippabilityCalculator
from .shuffling import Shuffler


class WatermarkEmbedder:
    def __init__(self, block_size=8, min_flip_score=0.3, use_randomization=True):
        self.block_size = block_size
        self.min_flip_score = min_flip_score
        self.use_randomization = use_randomization

        self.flippability_calc = FlippabilityCalculator()
        self.shuffler = Shuffler()

        # 嵌入记录
        self.embedded_blocks = []
        self.flipped_pixels = []
        self.embedding_stats = {}

    def embed(self, img, watermark_bits, key=None, shuffling_enabled=True,
              flip_map=None, return_detailed=False):
        start_time = time.time()
        working_img = img.copy()
        h, w = working_img.shape

        # 步骤1：混洗映射
        if shuffling_enabled and key is not None:
            forward_map, inverse_map = self.shuffler.generate_maps((h, w), key)
            shuffled_img = self.shuffler.shuffle(working_img, forward_map)
            self.forward_map = forward_map
            self.inverse_map = inverse_map
        else:
            shuffled_img = working_img
            self.forward_map = None
            self.inverse_map = None

        # 步骤2：计算可翻转性图
        if flip_map is None:
            flip_map = self.flippability_calc.compute_map(shuffled_img)

        # 启用混洗，需对flip_map进行相同的混洗
        if shuffling_enabled and self.forward_map is not None:
            flip_map_uint8 = (flip_map * 255).astype(np.uint8)
            shuffled_flip_map_uint8 = self.shuffler.shuffle(
                flip_map_uint8, self.forward_map)
            shuffled_flip_map = shuffled_flip_map_uint8.astype(
                np.float32) / 255.0
        else:
            shuffled_flip_map = flip_map

        # 步骤3：分块嵌入
        bits_embedded = 0
        total_blocks = 0
        flipped_count = 0
        skipped_blocks = 0

        self.embedded_blocks = []
        self.flipped_pixels = []

        # 计算最大块数
        max_blocks_h = h // self.block_size
        max_blocks_w = w // self.block_size

        # 按顺序遍历所有可能的块
        for i in range(0, max_blocks_h * self.block_size, self.block_size):
            for j in range(0, max_blocks_w * self.block_size, self.block_size):
                if bits_embedded >= len(watermark_bits):
                    break

                total_blocks += 1

                # 提取当前块
                block = shuffled_img[i:i+self.block_size, j:j+self.block_size]
                flip_scores = shuffled_flip_map[i:i +
                                                self.block_size, j:j+self.block_size]

                # 检查块是否完整
                if block.shape[0] != self.block_size or block.shape[1] != self.block_size:
                    continue

                # 检查是否有足够多的高分像素
                high_score_pixels = np.sum(flip_scores >= self.min_flip_score)
                if high_score_pixels == 0:
                    skipped_blocks += 1
                    continue

                # 记录使用的块位置
                self.embedded_blocks.append((i, j))

                # 获取目标奇偶性
                target_bit = watermark_bits[bits_embedded]

                # 嵌入
                new_block, flipped = self._embed_in_block(
                    block, target_bit, flip_scores)

                if flipped:
                    flipped_count += 1
                    # 记录翻转的像素位置（全局坐标）
                    flipped_positions = np.where(block != new_block)
                    for fi, fj in zip(flipped_positions[0], flipped_positions[1]):
                        self.flipped_pixels.append((i + fi, j + fj))

                # 更新图像
                shuffled_img[i:i+self.block_size,
                             j:j+self.block_size] = new_block
                bits_embedded += 1

        # 步骤4：反向混洗
        if shuffling_enabled and self.inverse_map is not None:
            watermarked_img = self.shuffler.unshuffle(
                shuffled_img, self.inverse_map)
        else:
            watermarked_img = shuffled_img

        # 计算统计信息
        duration = time.time() - start_time
        self.embedding_stats = {
            'capacity': bits_embedded,
            'total_blocks': total_blocks,
            'used_blocks': bits_embedded,
            'skipped_blocks': skipped_blocks,
            'flipped_pixels': flipped_count,
            'embedding_rate': bits_embedded / (h * w) * 1000,
            'time': duration
        }

        if return_detailed:
            return watermarked_img, self.embedding_stats, self.embedded_blocks, self.flipped_pixels
        else:
            return watermarked_img

    def _embed_in_block(self, block, target_bit, flip_scores):
        # 计算当前块的奇偶性
        current_parity = (np.sum(block == 0)) % 2

        if current_parity == target_bit:
            return block, False

        # 寻找最佳翻转点
        best_score = -1
        best_pos = None

        h, w = block.shape
        for bi in range(h):
            for bj in range(w):
                score = flip_scores[bi, bj]

                if bi == 0 or bi == h-1 or bj == 0 or bj == w-1:
                    min_boundary_score = 0.5
                    if score < min_boundary_score:
                        continue

                # 只考虑分数足够高的像素
                if score > best_score and score >= self.min_flip_score:
                    best_score = score
                    best_pos = (bi, bj)

        if best_pos is not None:
            block[best_pos] = 255 - block[best_pos]  # 黑到白
            return block, True
        else:
            return block, False

    def get_embedding_capacity(self, img, min_score=0.3):
        flip_map = self.flippability_calc.compute_map(img)
        h, w = img.shape

        # 计算每个块的可嵌入性
        max_blocks_h = h // self.block_size
        max_blocks_w = w // self.block_size

        embeddable_blocks = 0

        for i in range(0, max_blocks_h * self.block_size, self.block_size):
            for j in range(0, max_blocks_w * self.block_size, self.block_size):
                if i + self.block_size <= h and j + self.block_size <= w:
                    flip_scores = flip_map[i:i +
                                           self.block_size, j:j+self.block_size]
                    high_score_pixels = np.sum(flip_scores >= min_score)

                    if high_score_pixels > 0:
                        embeddable_blocks += 1

        return embeddable_blocks
