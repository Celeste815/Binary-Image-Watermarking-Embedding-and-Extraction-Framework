import tkinter as tk
from tkinter import ttk, messagebox
import re
import numpy as np
import cv2

from core.extraction import WatermarkExtractor
from utils.watermark_utils import bits_to_text
from gui.panels import ImagePanel


def add_noise(img, noise_level=0.05):
    noisy = img.copy()
    h, w = noisy.shape
    num_noise = int(h * w * noise_level)
    rows = np.random.randint(0, h, num_noise)
    cols = np.random.randint(0, w, num_noise)
    for i in range(num_noise):
        noisy[rows[i], cols[i]] = 255 - noisy[rows[i], cols[i]]
    return noisy


def rotate_image(img, angle=1.0):
    h, w = img.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def scale_image(img, scale_factor=0.9):
    h, w = img.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    if scale_factor < 1:
        result = np.ones((h, w), dtype=np.uint8) * 255
        start_h, start_w = (h - new_h) // 2, (w - new_w) // 2
        result[start_h:start_h+new_h, start_w:start_w+new_w] = scaled
        return result
    else:
        start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
        return scaled[start_h:start_h+h, start_w:start_w+w]


def translate_image(img, tx=5, ty=5):
    h, w = img.shape
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, translation_matrix, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def crop_image(img, crop_ratio=0.05):
    h, w = img.shape
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    crop_h = min(crop_h, h // 2 - 1)
    crop_w = min(crop_w, w // 2 - 1)
    if crop_h <= 0 or crop_w <= 0:
        return img.copy()
    cropped = img[crop_h:h-crop_h, crop_w:w-crop_w]
    result = np.ones((h, w), dtype=np.uint8) * 255
    result[crop_h:h-crop_h, crop_w:w-crop_w] = cropped
    return result


ATTACK_CONFIGS = {
    "裁剪": [0.02, 0.05, 0.08, 0.12, 0.15],
    "旋转": [0.5, 1.0, 2.0, 3.0, 5.0],
    "缩放": [0.8, 0.9, 0.95, 1.05, 1.1],
    "平移": [(2, 2), (5, 5), (8, 8), (11, 11), (15, 15)],
    "椒盐噪声": [0.01, 0.03, 0.05, 0.08, 0.10]
}


def apply_attack(img, attack_name, param):
    if attack_name == "裁剪":
        return crop_image(img, float(param))
    elif attack_name == "旋转":
        return rotate_image(img, float(param))
    elif attack_name == "缩放":
        return scale_image(img, float(param))
    elif attack_name == "平移":
        if isinstance(param, tuple):
            tx, ty = param
        else:
            tx = ty = int(param)
        return translate_image(img, tx, ty)
    elif attack_name == "椒盐噪声":
        return add_noise(img, float(param))
    else:
        raise ValueError(f"未知的攻击类型: {attack_name}")


class AttackWindow:
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main_window = main_window
        self.window = tk.Toplevel(parent)
        self.window.title("二值图像水印系统——鲁棒性攻击测试")
        self.window.geometry("1130x670")

        self.config = main_window.config
        self.shuffling_var = main_window.shuffling_var
        self.status_bar = main_window.status_bar

        self.test_results_cache = {}
        self.test_original_img = None
        self.test_watermarked_img = None
        self.test_embedded_blocks = None
        self.test_watermark_bits = None

        self.param_hints = {
            "裁剪": "建议范围: 0.02-0.12 (比例)",
            "旋转": "建议范围: 0.5-5 (角度)",
            "缩放": "建议范围: 0.8-1.2 (比例)",
            "平移": "建议范围: 2-15 (像素)",
            "椒盐噪声": "建议范围: 0.01-0.15"
        }

        self._create_ui()
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.refresh_current_image(show_info=False)

    def _on_closing(self):
        self.window.destroy()

    def _create_ui(self):
        main_pane = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        left_pane = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(left_pane, weight=1)

        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=2)

        # 攻击控制面板
        control_container = ttk.LabelFrame(
            left_pane, text="攻击测试控制", padding=10)
        left_pane.add(control_container, weight=1)
        control_frame = ttk.Frame(control_container)
        control_frame.pack(fill=tk.X)

        # 刷新按钮
        refresh_btn = ttk.Button(
            control_frame, text="刷新图像", command=self.refresh_current_image, width=10)
        refresh_btn.grid(row=0, column=0, sticky=tk.W, pady=4)
        self.current_img_label = ttk.Label(
            control_frame, text="当前图像: 未加载", foreground="blue")
        self.current_img_label.grid(
            row=0, column=1, columnspan=2, sticky=tk.W, padx=5)

        # 水印文本
        ttk.Label(control_frame, text="水印文本:").grid(
            row=1, column=0, sticky=tk.W, pady=4)
        self.test_watermark_var = tk.StringVar(value="")
        ttk.Entry(control_frame, textvariable=self.test_watermark_var, state='readonly').grid(
            row=1, column=1, columnspan=2, sticky=tk.EW, padx=5)

        # 攻击类型
        ttk.Label(control_frame, text="攻击类型:").grid(
            row=2, column=0, sticky=tk.W, pady=4)
        self.attack_type = tk.StringVar(value="裁剪")
        attack_combo = ttk.Combobox(control_frame, textvariable=self.attack_type,
                                    values=list(self.param_hints.keys()), state="readonly")
        attack_combo.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5)
        attack_combo.bind("<<ComboboxSelected>>", self._update_param_hint)

        # 攻击参数
        ttk.Label(control_frame, text="参数:").grid(
            row=3, column=0, sticky=tk.W, pady=4)
        self.attack_param = tk.StringVar(value="0.10")
        ttk.Entry(control_frame, textvariable=self.attack_param).grid(
            row=3, column=1, sticky=tk.EW, padx=5)

        # 参数提示
        self.param_note_label = ttk.Label(
            control_frame, text="", font=('Arial', 8), foreground='gray')
        self.param_note_label.grid(
            row=4, column=1, columnspan=2, sticky=tk.W, padx=5, pady=(0, 10))

        # 按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=5)
        ttk.Button(btn_frame, text="运行单个测试", command=self.run_single_attack_test).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(btn_frame, text="运行全部测试", command=self.run_all_attacks).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        control_frame.columnconfigure(1, weight=1)

        # 测试结果列表
        list_frame = ttk.LabelFrame(left_pane, text="测试结果列表", padding=10)
        left_pane.add(list_frame, weight=2)

        columns = ("攻击类型", "参数", "准确率", "BER", "NC")
        self.result_tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", height=10)
        for col, width in zip(columns, [80, 50, 60, 80, 80]):
            self.result_tree.heading(col, text=col, anchor=tk.W)
            self.result_tree.column(col, width=width, anchor=tk.W)

        scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_tree.bind("<<TreeviewSelect>>", self.on_result_select)

        # 图像对比区
        img_frame = ttk.LabelFrame(right_frame, text="图像对比", padding=10)
        img_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 10), pady=0)
        images_container = ttk.Frame(img_frame)
        images_container.pack(fill=tk.BOTH, expand=True)

        watermarked_frame = ttk.LabelFrame(
            images_container, text="原始含水印图像", padding=5)
        watermarked_frame.pack(side=tk.LEFT, fill=tk.BOTH,
                               expand=True, padx=5, pady=5)
        self.test_watermarked_panel = ImagePanel(
            watermarked_frame, width=300, height=300)
        self.test_watermarked_panel.pack(
            fill=tk.BOTH, expand=True, padx=5, pady=5)

        attacked_frame = ttk.LabelFrame(
            images_container, text="攻击后图像", padding=5)
        attacked_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                            expand=True, padx=5, pady=5)
        self.attacked_panel = ImagePanel(attacked_frame, width=300, height=300)
        self.attacked_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 提取结果展示区
        result_text_frame = ttk.LabelFrame(
            right_frame, text="提取结果详情", padding=10)
        result_text_frame.pack(fill=tk.X, padx=(0, 10), pady=10)
        self.test_result_text = tk.Text(
            result_text_frame, height=10, wrap=tk.WORD, font=('Consolas', 10), relief=tk.FLAT)
        text_scrollbar = ttk.Scrollbar(
            result_text_frame, orient=tk.VERTICAL, command=self.test_result_text.yview)
        self.test_result_text.configure(yscrollcommand=text_scrollbar.set)
        self.test_result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 底部状态栏
        status_frame = ttk.Frame(self.window)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))
        self.test_status_label = ttk.Label(
            status_frame, text="就绪", relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.test_status_label.pack(fill=tk.X)

        self._update_param_hint()

    def _update_param_hint(self, event=None):
        hint_text = self.param_hints.get(self.attack_type.get(), "无可用提示")
        self.param_note_label.config(text=hint_text)

    def refresh_current_image(self, show_info=True):
        if self.main_window.watermarked_img is not None:
            self.test_watermarked_img = self.main_window.watermarked_img.copy()
            self.test_original_img = self.main_window.original_img.copy(
            ) if self.main_window.original_img is not None else None
            self.test_embedded_blocks = self.main_window.embedded_blocks
            self.test_watermark_bits = self.main_window.watermark_bits
            self.test_watermark_var.set(self.main_window.watermark_var.get(
            ) if hasattr(self.main_window, 'watermark_var') else "")
            self.test_watermarked_panel.display_image(
                self.test_watermarked_img)
            self.attacked_panel.clear_image()
            self.test_result_text.delete(1.0, tk.END)
            self.current_img_label.config(
                text=f"已加载 {self.test_watermarked_img.shape[1]}x{self.test_watermarked_img.shape[0]}")
            if show_info:
                messagebox.showinfo("成功", "已从主窗口加载当前图像数据")
        else:
            if show_info:
                messagebox.showwarning("警告", "主窗口中没有含水印图像，请先嵌入水印")
            self.current_img_label.config(text="当前图像: 未加载", foreground="red")

    def _prepare_test_data(self):
        if self.test_watermarked_img is None:
            self.refresh_current_image(show_info=False)
            if self.test_watermarked_img is None:
                messagebox.showerror("错误", "主窗口中没有含水印图像，请先嵌入水印")
                return False
        if self.main_window.embedding_key is None:
            messagebox.showerror("错误", "未找到嵌入密钥，请先在主窗口嵌入水印")
            return False
        return True

    def _add_to_result_list(self, attack_name, param, accuracy, ber, nc, attacked_img, extracted_text, key):
        if isinstance(param, tuple):
            param_display = f"({param[0]},{param[1]})"
        else:
            param_display = f"{param}"

        item_id = self.result_tree.insert("", tk.END, values=(
            attack_name, param_display, f"{accuracy:.2f}%", f"{ber:.6f}", f"{nc:.6f}"))

        self.test_results_cache[f"{attack_name}_{param}"] = {
            'attacked_img': attacked_img, 'accuracy': accuracy, 'ber': ber, 'nc': nc,
            'extracted_text': extracted_text, 'key': key, 'param_display': param_display
        }
        return item_id

    def _display_detailed_result(self, attack_name, param, accuracy, ber, nc, extracted_text, key):
        self.test_result_text.delete(1.0, tk.END)
        self.test_result_text.insert(1.0,
                                     f"攻击类型:     {attack_name}\n"
                                     f"攻击参数:     {param}\n\n"
                                     f"提取密钥:     {key}\n"
                                     f"原始水印:     {self.test_watermark_var.get()}\n"
                                     f"提取水印:     {extracted_text}\n\n"
                                     f"准确率 (ACC): {accuracy:.2f}%\n"
                                     f"比特错误率 (BER): {ber:.6f}\n"
                                     f"归一化相关系数 (NC): {nc:.6f}\n")

    def run_single_attack_test(self):
        if not self._prepare_test_data():
            return
        key = self.main_window.embedding_key
        if key is None:
            messagebox.showerror("错误", "未找到嵌入密钥")
            return

        attack_name = self.attack_type.get()
        param_str = self.attack_param.get()
        try:
            param = float(param_str)
        except ValueError:
            param = param_str

        self.test_status_label.config(text=f"正在执行攻击: {attack_name}...")
        self.window.update()

        attacked_img = apply_attack(
            self.test_watermarked_img, attack_name, param)
        if attacked_img is None:
            messagebox.showerror("错误", f"攻击类型 {attack_name} 参数无效")
            self.test_status_label.config(text="就绪")
            return

        self.attacked_panel.display_image(attacked_img)
        extractor = WatermarkExtractor(block_size=self.config.BLOCK_SIZE)
        extracted_bits, _ = extractor.extract(
            attacked_img, original_shape=self.test_original_img.shape if self.test_original_img is not None else None,
            key=key, shuffling_enabled=self.shuffling_var.get(),
            embedded_blocks=self.test_embedded_blocks,
            expected_length=len(self.test_watermark_bits) if self.test_watermark_bits is not None else None)

        if self.test_watermark_bits is not None:
            accuracy, ber, nc = extractor.verify(
                extracted_bits, self.test_watermark_bits)
            extracted_text = bits_to_text(extracted_bits)
            self._display_detailed_result(
                attack_name, param, accuracy, ber, nc, extracted_text, key)
            item_id = self._add_to_result_list(
                attack_name, param, accuracy, ber, nc, attacked_img, extracted_text, key)
            self.result_tree.selection_set(item_id)
            self.test_status_label.config(
                text=f"攻击完成: {attack_name} - 准确率 {accuracy:.2f}%")
        else:
            messagebox.showerror("错误", "没有水印信息可供验证")
            self.test_status_label.config(text="就绪")

    def run_all_attacks(self):
        if not self._prepare_test_data():
            return
        key = self.main_window.embedding_key
        if key is None:
            messagebox.showerror("错误", "未找到嵌入密钥")
            return

        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        self.test_results_cache.clear()

        self.window.config(cursor="watch")
        self.window.update()

        extractor = WatermarkExtractor(block_size=self.config.BLOCK_SIZE)
        total_tests = sum(len(v) for v in ATTACK_CONFIGS.values())
        current_test = 0

        for attack_name, params in ATTACK_CONFIGS.items():
            for param in params:
                try:
                    current_test += 1
                    self.test_status_label.config(
                        text=f"测试中 ({current_test}/{total_tests}): {attack_name} {param}...")
                    self.window.update()

                    attacked_img = apply_attack(
                        self.test_watermarked_img, attack_name, param)
                    if attacked_img is None:
                        continue

                    extracted_bits, _ = extractor.extract(
                        attacked_img,
                        original_shape=self.test_original_img.shape if self.test_original_img is not None else None,
                        key=key, shuffling_enabled=self.shuffling_var.get(),
                        embedded_blocks=self.test_embedded_blocks,
                        expected_length=len(self.test_watermark_bits) if self.test_watermark_bits is not None else None)

                    if self.test_watermark_bits is not None:
                        accuracy, ber, nc = extractor.verify(
                            extracted_bits, self.test_watermark_bits)
                        extracted_text = bits_to_text(extracted_bits)
                        self._add_to_result_list(
                            attack_name, param, accuracy, ber, nc, attacked_img, extracted_text, key)
                except Exception as e:
                    print(f"测试失败 {attack_name}_{param}: {e}")

        self.window.config(cursor="")
        self.test_status_label.config(text=f"全部攻击测试完成，共运行 {current_test} 项")
        messagebox.showinfo(
            "测试完成", f"所有攻击测试已完成！\n共运行 {current_test} 项测试，结果已显示在列表中。")

    def on_result_select(self, event):
        selection = self.result_tree.selection()
        if not selection:
            return
        item = self.result_tree.item(selection[0], 'values')
        if len(item) < 2:
            return

        attack_name, param_str = item[0], item[1]

        # 解析参数
        try:
            if attack_name == "平移":
                match = re.search(r'\((\d+),(\d+)\)', param_str)
                if match:
                    param = (int(match.group(1)), int(match.group(2)))
                else:
                    param = param_str
            else:
                param = float(param_str)
        except ValueError:
            param = param_str

        cache_key = f"{attack_name}_{param}"
        if cache_key in self.test_results_cache:
            cached = self.test_results_cache[cache_key]
            self.attacked_panel.display_image(cached['attacked_img'])
            self._display_detailed_result(
                attack_name, cached.get(
                    'param_display', param_str), cached['accuracy'],
                cached['ber'], cached['nc'], cached.get('extracted_text', 'N/A'), cached.get('key', '未记录'))
