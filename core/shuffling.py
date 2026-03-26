import numpy as np
import random


class Shuffler:
    def __init__(self, key=None):
        """
        初始化混洗器

        Args:
            key: 随机种子密钥
        """
        self.key = key
        self.forward_map = None
        self.inverse_map = None

    # 生成混洗映射
    def generate_maps(self, img_shape, key=None):
        if key is not None:
            self.key = key
        elif self.key is None:
            self.key = 2251538  # 默认密钥

        h, w = img_shape
        total_pixels = h * w

        # 生成随机排列
        np.random.seed(self.key)
        indices = np.arange(total_pixels)
        np.random.shuffle(indices)

        self.forward_map = indices.reshape((h, w))
        self.inverse_map = np.zeros_like(self.forward_map)
        for i in range(h):
            for j in range(w):
                idx = self.forward_map[i, j]
                self.inverse_map[idx // w, idx % w] = i * w + j

        return self.forward_map, self.inverse_map

    def shuffle(self, img, map=None):
        if map is None:
            if self.forward_map is None:
                raise ValueError("请先生成混洗映射")
            map = self.forward_map

        h, w = img.shape
        flat_img = img.flatten()
        shuffled_flat = flat_img[map.flatten()]
        return shuffled_flat.reshape((h, w))

    def unshuffle(self, img, map=None):
        if map is None:
            if self.inverse_map is None:
                raise ValueError("请先生成反向混洗映射")
            map = self.inverse_map

        h, w = img.shape
        flat_img = img.flatten()
        unshuffled_flat = flat_img[map.flatten()]
        return unshuffled_flat.reshape((h, w))

    def shuffle_with_key(self, img, key):
        forward_map, _ = self.generate_maps(img.shape, key)
        return self.shuffle(img, forward_map)
