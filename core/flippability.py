import numpy as np
from scipy import ndimage


class FlippabilityCalculator:
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.T1 = 2  # 平滑度阈值 T1 = 2
        self.w1 = 0.25  # 规则1减分权重 (水平/垂直方向转换数为0)
        self.w2 = 0.125  # 规则2减分权重 (最小转换数 < T1)
        self.w3 = 0.125  # 规则3/4权重 (转换数不变加分/增加减分)
        self.w4 = 0.125  # 规则5权重 (连通性改变减分)
        self.SB = 0.5    # 基础分数
        self.lut = self._build_lookup_table()

    def _build_lookup_table(self):
        lut = {}
        total_patterns = 512

        with open('flippability_patterns.txt', 'w', encoding='utf-8') as f:
            f.write("3x3模式可翻转性分数表\n")
            f.write("=" * 60 + "\n")
            f.write("索引\t二进制模式\t\t分数\n")
            f.write("-" * 60 + "\n")
            for pattern in range(total_patterns):
                bits = [(pattern >> i) & 1 for i in range(9)]
                block_3x3 = np.array(bits).reshape((3, 3)) * 255

                score = self._compute_score(block_3x3)
                lut[pattern] = score

                binary_str = ''.join(str(b) for b in bits)
                formatted_binary = f"{binary_str[0:3]} {binary_str[3:6]} {binary_str[6:9]}"

                f.write(f"{pattern:3d}\t{formatted_binary}\t\t{score:.3f}\n")

        scores = list(lut.values())
        zero_scores = sum(1 for s in scores if s == 0)
        high_scores = sum(1 for s in scores if s >= 0.5)

        """ print(f"查找表构建完成:")
        print(f"  总模式数: {total_patterns}")
        print(
            f"  得0分模式数: {zero_scores} ({zero_scores/total_patterns*100:.1f}%)")
        print(
            f"  得高分模式数(>=0.5): {high_scores} ({high_scores/total_patterns*100:.1f}%)")
        print(f"  详细结果已保存到 'flippability_patterns.txt'") """

        return lut

    # 计算3x3块的四个方向转换次数
    def _compute_transitions(self, block):
        # 水平方向转换次数 T_H
        T_H = 0
        T_H += (block[0, 0] != block[0, 1]) + (block[0, 1] != block[0, 2])
        T_H += (block[1, 0] != block[1, 1]) + (block[1, 1] != block[1, 2])
        T_H += (block[2, 0] != block[2, 1]) + (block[2, 1] != block[2, 2])

        # 垂直方向转换次数 T_V
        T_V = 0
        T_V += (block[0, 0] != block[1, 0]) + (block[1, 0] != block[2, 0])
        T_V += (block[0, 1] != block[1, 1]) + (block[1, 1] != block[2, 1])
        T_V += (block[0, 2] != block[1, 2]) + (block[1, 2] != block[2, 2])

        # 主对角线方向转换次数 T_D1
        T_D1 = 0
        T_D1 += (block[0, 0] != block[1, 1])
        T_D1 += (block[1, 1] != block[2, 2])

        # 副对角线方向转换次数 T_D2
        T_D2 = 0
        T_D2 += (block[0, 2] != block[1, 1])
        T_D2 += (block[1, 1] != block[2, 0])

        return T_H, T_V, T_D1, T_D2

    def _compute_connectivity(self, block):
        """
        计算3x3块的连通分量数量
        Args:
            block: 3x3二值块 (0/1)
        Returns:
            (black_clusters, white_clusters) 黑白像素的连通分量数
        """
        structure = np.ones((3, 3))  # 8连通性

        # 黑色像素聚类数（值为1）
        _, black_clusters = ndimage.label(block, structure=structure)

        # 白色像素聚类数（值为0）
        _, white_clusters = ndimage.label(1 - block, structure=structure)

        return black_clusters, white_clusters

    def _compute_score(self, block_3x3):
        block = (block_3x3 > 127).astype(int)
        center = block[1, 1]

        # ========== Step 1: 快速排除 ==========
        # 条件 (a): 均匀块（全黑或全白）→ 不可翻转
        if len(np.unique(block)) == 1:
            return 0.0

        # 条件 (b): 孤立点（中心与所有邻居相反）→ 不可翻转
        neighbors = block.flatten()
        neighbors = np.delete(neighbors, 4)
        if np.all(neighbors == 1 - center):
            return 0.0

        # ========== Step 2: 计算特征 ==========
        # 1. 计算局部熵（表示纹理复杂度）
        p_black = np.sum(block) / 9
        p_white = 1 - p_black
        if p_black > 0 and p_black < 1:
            entropy = -p_black * np.log2(p_black) - p_white * np.log2(p_white)
        else:
            entropy = 0

        # 2. 计算边缘强度（梯度）
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        grad_x = np.abs(np.sum(block * sobel_x))
        grad_y = np.abs(np.sum(block * sobel_y))
        edge_strength = (grad_x + grad_y) / 36.0  # 归一化

        # 3. 计算转换数（颜色变化次数）
        T_H, T_V, T_D1, T_D2 = self._compute_transitions(block)
        total_trans = T_H + T_V + T_D1 + T_D2

        # 4. 计算中心像素与周围的相关性
        neighbor_mean = np.mean(neighbors)
        center_diff = abs(center - neighbor_mean)

        # 5. 翻转稳定性
        flipped_block = block.copy()
        flipped_block[1, 1] = 1 - center
        T_H_f, T_V_f, T_D1_f, T_D2_f = self._compute_transitions(flipped_block)
        total_new = T_H_f + T_V_f + T_D1_f + T_D2_f
        stability = 1 - abs(total_new - total_trans) / max(total_trans, 1)

        # ========== Step 3: 综合评分 ==========
        # 综合评分
        score = (0.3 * entropy + 0.2 * edge_strength + 0.2 * (total_trans / 14.0) +
                 0.15 * (1 - center_diff) + 0.15 * stability
                 )

        # 连通性惩罚
        black_clusters, white_clusters = self._compute_connectivity(block)
        black_clusters_f, white_clusters_f = self._compute_connectivity(
            flipped_block)
        if black_clusters != black_clusters_f or white_clusters != white_clusters_f:
            score *= 0.8

        return max(0.0, min(1.0, score))

    # 使用查找表获取可翻转性分数
    def get_score(self, block_3x3):
        block_bin = (block_3x3 > 127).astype(int)
        pattern = 0
        for i in range(3):
            for j in range(3):
                pattern |= (block_bin[i, j] << (i*3 + j))

        return self.lut.get(pattern, 0.0)

    # 计算整个图像的可翻转性图
    def compute_map(self, img):
        h, w = img.shape
        flip_map = np.zeros((h, w), dtype=np.float32)
        self.score_distribution = {}

        # 为每个像素计算可翻转性分数（排除边界）
        for i in range(1, h-1):
            for j in range(1, w-1):
                block_3x3 = img[i-1:i+2, j-1:j+2]
                score = self.get_score(block_3x3)
                flip_map[i, j] = score

                rounded_score = round(score, 1)
                self.score_distribution[rounded_score] = self.score_distribution.get(
                    rounded_score, 0) + 1

        # 边界像素设为0（不可翻转）
        flip_map[0, :] = 0
        flip_map[h-1, :] = 0
        flip_map[:, 0] = 0
        flip_map[:, w-1] = 0

        self.flip_map = flip_map
        return flip_map

    def get_flippable_pixels(self, flip_map=None, min_score=None):
        """
        获取可翻转像素数量和比例
        Args:
            flip_map: 可翻转性图
            min_score: 最小分数阈值
        Returns:
            (数量, 比例)
        """
        if flip_map is None:
            flip_map = self.flip_map if hasattr(self, 'flip_map') else None
            if flip_map is None:
                return 0, 0.0

        if min_score is None:
            min_score = self.threshold

        h, w = flip_map.shape
        total_pixels = (h-2) * (w-2) if h > 2 and w > 2 else h * w

        flippable = np.sum(flip_map >= min_score)
        ratio = flippable / total_pixels if total_pixels > 0 else 0

        return flippable, ratio


def test_with_paper_patterns():
    """测试"""
    calc = FlippabilityCalculator()
    pattern1 = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]) * 255

    patterns = [pattern1]
    descriptions = ["纹理"]

    for i, (pattern, desc) in enumerate(zip(patterns, descriptions)):
        score = calc.get_score(pattern)
        print(f"\n模式{i+1} ({desc}) 分数: {score:.3f}")
        print("模式:")
        for row in pattern:
            print(" ".join(["●" if p > 127 else "○" for p in row]))


if __name__ == "__main__":
    test_with_paper_patterns()
