from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import hashlib
import base64


class AESCrypto:
    def __init__(self, key: int):
        key_bytes = str(key).encode('utf-8')
        self.aes_key = hashlib.sha256(key_bytes).digest()
        self.iv = hashlib.md5(key_bytes).digest()

    def encrypt(self, plaintext: str) -> bytes:
        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.iv)
        # 对明文进行填充并加密
        padded_data = pad(plaintext.encode('utf-8'), AES.block_size)
        encrypted = cipher.encrypt(padded_data)
        return encrypted

    def decrypt(self, ciphertext: bytes) -> str:
        try:
            cipher = AES.new(self.aes_key, AES.MODE_CBC, self.iv)
            decrypted_padded = cipher.decrypt(ciphertext)
            decrypted = unpad(decrypted_padded, AES.block_size)
            return decrypted.decode('utf-8')
        except ValueError:
            return "解密失败：密钥错误"

    def encrypt_to_bits(self, plaintext: str) -> list:
        encrypted_bytes = self.encrypt(plaintext)
        # 将加密后的字节转换为比特流
        bits = []
        for byte in encrypted_bytes:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    def decrypt_from_bits(self, bits: list) -> str:
        # 将比特流转换为字节
        bytes_list = []
        for i in range(0, len(bits) - (len(bits) % 8), 8):
            byte_val = 0
            for bit in bits[i:i+8]:
                byte_val = (byte_val << 1) | bit
            bytes_list.append(byte_val)

        # 解密
        ciphertext = bytes(bytes_list)
        return self.decrypt(ciphertext)
