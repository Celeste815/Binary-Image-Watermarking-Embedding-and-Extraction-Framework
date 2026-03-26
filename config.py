"配置文件-存储所有全局参数和默认设置"


class Config:
    # 算法参数
    BLOCK_SIZE = 8  # 默认块大小
    FLIPPABILITY_THRESHOLD = 0.1  # 可翻转性阈值
    MIN_FLIP_SCORE = 0.3  # 最小翻转分数

    ALPHA = 3
    # MIN_FLIP_SCORE = 0.1

    # 混洗参数
    DEFAULT_KEY = 2251538  # 默认密钥

    # 水印参数
    DEFAULT_WATERMARK = "2251538LY"

    # 显示参数
    DISPLAY_SIZE = (400, 400)  # 显示尺寸

    # 文件路径
    OUTPUT_DIR = "./output/"
    TEST_IMAGE_DIR = "./test_images/"

    @classmethod
    def to_dict(cls):
        """将配置转换为字典"""
        return {key: value for key, value in cls.__dict__.items()
                if not key.startswith('__') and not callable(value)}
