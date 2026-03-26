import numpy as np
import hashlib


def text_to_bits(text, encoding='ascii', bits_per_char=7):
    bits = []
    for char in text:
        if encoding == 'ascii':
            ascii_val = ord(char)
            bits.extend([int(b)
                        for b in format(ascii_val, f'0{bits_per_char}b')])
        elif encoding == 'utf-8':
            byte_data = char.encode('utf-8')
            for byte in byte_data:
                bits.extend([int(b) for b in format(byte, '08b')])

    return np.array(bits, dtype=int)


def bits_to_text(bits, encoding='ascii', bits_per_char=7):
    text = ""
    if encoding == 'ascii':
        for i in range(0, len(bits), bits_per_char):
            if i + bits_per_char <= len(bits):
                byte_bits = bits[i:i+bits_per_char]
                ascii_val = int(''.join(map(str, byte_bits)), 2)
                if 32 <= ascii_val <= 126:
                    text += chr(ascii_val)
                else:
                    text += '?'
    elif encoding == 'utf-8':
        bytes_list = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte_bits = bits[i:i+8]
                byte_val = int(''.join(map(str, byte_bits)), 2)
                bytes_list.append(byte_val)

        try:
            text = bytes(bytes_list).decode('utf-8', errors='replace')
        except:
            text = "解码错误"

    return text


def calculate_ber(bits1, bits2):
    """
    计算比特错误率
    Args:
        bits1: 第一个比特序列
        bits2: 第二个比特序列
    Returns:
        比特错误率
    """
    min_len = min(len(bits1), len(bits2))
    if min_len == 0:
        return 1.0

    errors = np.sum(bits1[:min_len] != bits2[:min_len])
    return errors / min_len


def calculate_nc(bits1, bits2):
    """
    计算归一化相关系数
    Args:
        bits1: 第一个比特序列
        bits2: 第二个比特序列
    Returns:
        归一化相关系数
    """
    min_len = min(len(bits1), len(bits2))
    if min_len == 0:
        return 0.0

    # 转换为-1/1表示
    b1 = 2 * bits1[:min_len] - 1
    b2 = 2 * bits2[:min_len] - 1

    numerator = np.sum(b1 * b2)
    denominator = np.sqrt(np.sum(b1**2) * np.sum(b2**2))

    if denominator == 0:
        return 0.0

    return numerator / denominator
