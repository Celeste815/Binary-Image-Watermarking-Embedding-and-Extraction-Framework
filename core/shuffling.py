import numpy as np


class Shuffler:
    def __init__(self, key=None):
        self.key = key
        self.forward_map = None
        self.inverse_map = None

    def generate_maps(self, img_shape, key=None):
        if key is not None:
            self.key = key
        elif self.key is None:
            self.key = 2251538

        h, w = img_shape
        total_pixels = h * w

        np.random.seed(self.key)
        indices = np.arange(total_pixels)
        shuffled_indices = indices.copy()
        np.random.shuffle(shuffled_indices)

        self.forward_map = shuffled_indices.reshape((h, w))
        self.inverse_map = np.zeros((h, w), dtype=np.int64)
        for i in range(h):
            for j in range(w):
                orig_idx = i * w + j
                new_idx = self.forward_map[i, j]
                new_i = new_idx // w
                new_j = new_idx % w
                self.inverse_map[new_i, new_j] = orig_idx

        return self.forward_map, self.inverse_map

    def shuffle(self, img, map=None):
        if map is None:
            if self.forward_map is None:
                raise ValueError("请先生成混洗映射")
            map = self.forward_map

        h, w = img.shape
        flat_img = img.flatten()

        shuffled_flat = np.zeros_like(flat_img)
        for orig_idx, new_idx in enumerate(map.flatten()):
            shuffled_flat[new_idx] = flat_img[orig_idx]

        return shuffled_flat.reshape((h, w))

    def unshuffle(self, img, map=None):
        if map is None:
            if self.inverse_map is None:
                raise ValueError("请先生成反向混洗映射")
            map = self.inverse_map

        h, w = img.shape
        flat_img = img.flatten()

        unshuffled_flat = np.zeros_like(flat_img)
        for new_idx, orig_idx in enumerate(map.flatten()):
            unshuffled_flat[orig_idx] = flat_img[new_idx]

        return unshuffled_flat.reshape((h, w))

    def shuffle_with_key(self, img, key):
        forward_map, _ = self.generate_maps(img.shape, key)
        return self.shuffle(img, forward_map)

    def unshuffle_with_key(self, img, key):
        _, inverse_map = self.generate_maps(img.shape, key)
        return self.unshuffle(img, inverse_map)
