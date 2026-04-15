# Binary-Image-Watermarking-Embedding-and-Extraction-Framework
此项目是本人自主独立开发研究的二值图像水印嵌入与提取系统。面向二值图像在数字化办公中广泛应用所带来的安全挑战，针对其像素冗余度低、视觉失真敏感的技术难点，开展水印嵌入与提取算法的设计与分析研究。

## 核心技术
在技术层面，构建了多维可翻转性评分模型，通过分析像素的平滑度、连通性与孤立点特征，筛选视觉影响最小的像素作为修改对象；采用块级统计特征调制策略，以块内黑像素数的奇偶性承载水印信息，实现盲提取；引入基于密钥的像素混洗机制，解决可翻转像素分布不均问题并增强安全性。

同时对水印文本进行AES加密处理，使得实验安全性大幅提高。

系统采用Python与OpenCV实现模块化架构，支持扫描文本、签名、二维码等多种二值图像，实现了高隐蔽性水印嵌入、盲提取、基础鲁棒性保障与密钥安全管理，并对嵌入容量与计算复杂度进行了理论分析。

## 创新点
本项目的核心创新在于对可翻转性评分模型的深度重构与优化。相较于传统方法依赖平滑度与连通性的二元判别，本研究创新性地构建了多维特征融合的连续评分机制，将局部熵、边缘强度、转换数、中心像素相关性、翻转稳定性五类特征纳入统一框架，实现对像素视觉敏感度的精细化表达；同时引入连通性惩罚机制，通过评估翻转前后黑白像素8连通分量的变化，确保操作不破坏笔画完整性与拓扑结构。

此外，系统采用查找表预计算策略对全部512种3×3邻域模式进行离线评分，将在线计算转化为常数时间查表操作，显著提升运行效率。这一改进型模型在继承经典方法理论清晰性的基础上，通过多维特征融合与工程优化，显著增强了算法的通用性、隐蔽性与实时性，为二值图像内容安全认证提供了兼具理论深度与工程可行性的解决方案。

## 核心代码
对可翻转性评估算法进行了重构与优化，创新性的采用了连续评分机制

        def _compute_score(self, block_3x3):
        block = (block_3x3 > 127).astype(int)
        center = block[1, 1]

        # ========== Step 1: 快速排除 ==========
        if len(np.unique(block)) == 1:
            return 0.0

        neighbors = block.flatten()
        neighbors = np.delete(neighbors, 4)
        if np.all(neighbors == 1 - center):
            return 0.0

        if self._has_straight_line(block):
            return 0.0

        # ========== Step 2: 计算特征 ==========
        # 1. 计算局部熵
        p_black = np.sum(block) / 9
        p_white = 1 - p_black
        if p_black > 0 and p_black < 1:
            entropy = -p_black * np.log2(p_black) - p_white * np.log2(p_white)
        else:
            entropy = 0

        # 2. 计算边缘强度
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        grad_x = np.abs(np.sum(block * sobel_x))
        grad_y = np.abs(np.sum(block * sobel_y))
        edge_strength = (grad_x + grad_y) / 36.0  # 归一化

        # 3. 计算转换数
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
        score = (0.3 * entropy + 0.15 * edge_strength + 0.15 * (total_trans / 14.0) +
                 0.2 * (1 - center_diff) + 0.2 * stability
                 )

        # 连通性惩罚
        black_clusters, white_clusters = self._compute_connectivity(block)
        black_clusters_f, white_clusters_f = self._compute_connectivity(
            flipped_block)
        if black_clusters != black_clusters_f or white_clusters != white_clusters_f:
            score *= 0.8

        return max(0.0, min(1.0, score))

## 成果展示

<img width="1615" height="894" alt="image" src="https://github.com/user-attachments/assets/b6c02176-55e3-4645-ac44-6195fb1c91f3" />

可翻转性热图展示：

<img width="1214" height="642" alt="image" src="https://github.com/user-attachments/assets/33e7b9eb-91a4-41ff-9029-364107127e2c" />

提取验证：

<img width="1207" height="663" alt="image" src="https://github.com/user-attachments/assets/0b382c82-7744-48d5-899c-3c22b2790e2e" />

<img width="1206" height="679" alt="image" src="https://github.com/user-attachments/assets/b0d20b52-bbf3-4e2a-b0d6-7985ece77f44" />

差异对比：

<img width="1225" height="489" alt="image" src="https://github.com/user-attachments/assets/0c327958-dcd8-4ccb-b561-6f0096da8d3a" />

攻击测试：

<img width="1406" height="867" alt="image" src="https://github.com/user-attachments/assets/9e1a56d5-e249-4252-b388-0ae4b7e3d2fa" />

algorithm_report.txt为系统算法测试结果详细内容
