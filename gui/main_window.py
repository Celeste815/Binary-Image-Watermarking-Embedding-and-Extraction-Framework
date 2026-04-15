import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from config import Config
from core.embedding import WatermarkEmbedder
from core.extraction import WatermarkExtractor
from core.flippability import FlippabilityCalculator
from utils.image_utils import load_image, save_image, compare_images, create_comparison_view
from utils.watermark_utils import text_to_bits, bits_to_text, encrypted_text_to_bits, bits_to_encrypted_text
from gui.panels import ImagePanel
from gui.dialogs import StatsDialog, AboutDialog


class ImageComparisonWindow:
    def __init__(self, parent, comparison_image, titles, diff_info):
        self.window = tk.Toplevel(parent)
        self.window.title("图像对比——差异分析")

        comparison_rgb = cv2.cvtColor(comparison_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(comparison_rgb)
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        max_width = int(screen_width * 0.8)
        max_height = int(screen_height * 0.8)

        if pil_image.width > max_width or pil_image.height > max_height:
            pil_image.thumbnail((max_width, max_height),
                                Image.Resampling.LANCZOS)

        self.canvas = tk.Canvas(self.window)
        scrollbar_y = ttk.Scrollbar(
            self.window, orient="vertical", command=self.canvas.yview)
        scrollbar_x = ttk.Scrollbar(
            self.window, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=scrollbar_y.set,
                              xscrollcommand=scrollbar_x.set)

        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.inner_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")

        self.photo = ImageTk.PhotoImage(pil_image)
        image_label = ttk.Label(self.inner_frame, image=self.photo)
        image_label.pack(padx=10, pady=10)

        title_text = "|".join(titles)
        title_label = ttk.Label(
            self.inner_frame,
            text=f"对比视图: {title_text}",
            font=('Arial', 15, 'bold')
        )
        title_label.pack(pady=(0, 5))

        info_frame = ttk.LabelFrame(self.inner_frame, text="差异统计", padding=10)
        info_frame.pack(pady=5, padx=10, fill=tk.X)
        info_text = f"""
        修改像素数:{diff_info['changed_pixels']:,}像素
        修改比例:{diff_info['change_percentage']:.4f}%
        图像尺寸:{diff_info.get('width', 'N/A')}x{diff_info.get('height', 'N/A')}
        """
        info_label = ttk.Label(
            info_frame,
            text=info_text,
            font=('Consolas', 13),
            justify=tk.LEFT
        )
        info_label.pack()

        btn_frame = ttk.Frame(self.inner_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="关闭", command=self.window.destroy).pack(
            side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存对比图",
                   command=lambda: self._save_comparison(comparison_rgb)).pack(side=tk.LEFT, padx=5)

        self.inner_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        window_width = min(pil_image.width + 50, max_width)
        window_height = min(pil_image.height + 200, max_height)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self._bind_mousewheel()

    def _bind_mousewheel(self):
        def on_mousewheel(event):
            if event.delta:
                self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else:
                if event.num == 4:
                    self.canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.canvas.yview_scroll(1, "units")

        self.canvas.bind_all("<MouseWheel>", on_mousewheel)
        self.canvas.bind_all("<Button-4>", on_mousewheel)
        self.canvas.bind_all("<Button-5>", on_mousewheel)
        self.window.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _on_closing(self):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
        self.window.destroy()

    def _save_comparison(self, image_array):
        from PIL import Image
        import os
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("BMP files", "*.bmp")
            ],
            title="保存对比图像"
        )
        if path:
            try:
                pil_image = Image.fromarray(image_array)
                pil_image.save(path)
                messagebox.showinfo("成功", f"对比图像已保存到:\n{path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败:\n{str(e)}")


class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("二值图像水印嵌入提取系统")
        self.root.geometry("1300x700")

        self.config = Config()
        self.embedder = WatermarkEmbedder(
            block_size=self.config.BLOCK_SIZE, min_flip_score=self.config.MIN_FLIP_SCORE)
        self.extractor = WatermarkExtractor(block_size=self.config.BLOCK_SIZE)
        self.flippability_calc = FlippabilityCalculator(
            threshold=self.config.FLIPPABILITY_THRESHOLD)

        self.original_img = None
        self.watermarked_img = None
        self.watermark_bits = None
        self.flip_map = None
        self.embedded_blocks = []
        self.flipped_pixels = []
        self.embedding_key = None
        self.encryption_enabled = tk.BooleanVar(value=True)  # 默认启用加密
        self._setup_ui()

    def _setup_ui(self):
        self._create_menu()

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_frame = ttk.Frame(main_frame, width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        self._create_control_panel(left_frame)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._create_status_bar()
        self._create_display_panel(right_frame)

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(
            label="打开图像", command=self.load_image, accelerator="Ctrl+O")
        file_menu.add_command(
            label="保存结果", command=self.save_results, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(
            label="退出", command=self.root.quit, accelerator="Ctrl+Q")

        # 编辑菜单
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="编辑", menu=edit_menu)
        edit_menu.add_command(label="重置", command=self.reset)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于", command=self.show_about)

        self.root.bind('<Control-o>', lambda e: self.load_image())
        self.root.bind('<Control-s>', lambda e: self.save_results())
        self.root.bind('<Control-q>', lambda e: self.root.quit())

    def _create_control_panel(self, parent):
        title = ttk.Label(parent, text="控制面板", font=('Arial', 14, 'bold'))
        title.pack(pady=10)

        btn_frame = ttk.LabelFrame(parent, text="操作", padding=10)
        btn_frame.pack(fill=tk.X, pady=5)

        buttons = [
            ("1. 读取图片", self.load_image),
            ("2. 计算可翻转性", self.compute_flippability),
            ("3. 嵌入水印", self.embed_watermark),
            ("4. 显示差异", self.show_difference),
            ("5. 提取验证", self.extract_watermark),
            ("6. 保存结果", self.save_results),
            ("7. 算法测试", self.run_tests1),
            ("8. 鲁棒性测试", self.run_tests2)
        ]
        for text, command in buttons:
            btn = ttk.Button(btn_frame, text=text, command=command, width=20)
            btn.pack(pady=2)

        param_frame = ttk.LabelFrame(parent, text="参数设置", padding=10)
        param_frame.pack(fill=tk.X, pady=5)

        ttk.Label(param_frame, text="水印文本:").grid(
            row=1, column=0, sticky=tk.W, pady=2)
        self.watermark_var = tk.StringVar(value=self.config.DEFAULT_WATERMARK)
        ttk.Entry(param_frame, textvariable=self.watermark_var,
                  width=15).grid(row=1, column=1, pady=2, padx=5)

        # 加密选项
        ttk.Checkbutton(param_frame, text="启用AES加密",
                        variable=self.encryption_enabled).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.shuffling_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(param_frame, text="启用混洗", variable=self.shuffling_var).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        info_frame = ttk.LabelFrame(parent, text="信息", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.info_text = tk.Text(info_frame, height=10, width=35, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(self.info_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.info_text.yview)

    def _create_display_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="图像显示")
        display_frame = ttk.Frame(main_tab)
        display_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(display_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(left_frame, text="原始图像", font=('Arial', 12, 'bold')).pack()
        self.original_panel = ImagePanel(left_frame, width=450, height=450)
        self.original_panel.pack(pady=5)

        right_frame = ttk.Frame(display_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        ttk.Label(right_frame, text="含水印图像", font=('Arial', 12, 'bold')).pack()
        self.watermarked_panel = ImagePanel(right_frame, width=450, height=450)
        self.watermarked_panel.pack(pady=5)

        flip_tab = ttk.Frame(notebook)
        notebook.add(flip_tab, text="可翻转性分析")
        ttk.Label(flip_tab, text="可翻转性热图", font=(
            'Arial', 12, 'bold')).pack(pady=5)
        self.flip_panel = ImagePanel(flip_tab, width=600, height=400)
        self.flip_panel.pack(pady=5)

        stats_tab = ttk.Frame(notebook)
        notebook.add(stats_tab, text="统计信息")
        self.stats_text = tk.Text(stats_tab, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_status_bar(self):
        self.status_bar = ttk.Label(
            self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ==================== 功能方法 ====================
    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.bmp *.tif *.tiff")])

        if path:
            img = load_image(path)
            if img is not None:
                self.original_img = img
                self.original_panel.display_image(img)

                self.watermarked_img = None
                self.flip_map = None

                self._log_info(f"图像加载成功: {img.shape[1]}x{img.shape[0]}")
                self.status_bar.config(text=f"已加载: {Path(path).name}")
            else:
                messagebox.showerror("错误", "无法加载图像")

    def compute_flippability(self):
        if self.original_img is None:
            messagebox.showerror("错误", "请先加载图像")
            return

        self.flip_map = self.flippability_calc.compute_map(
            self.original_img)  # 获取分数
        flip_map_display = (self.flip_map * 255).astype(np.uint8)  # 转化为热图
        heatmap = cv2.applyColorMap(flip_map_display, cv2.COLORMAP_JET)
        self.flip_panel.display_image(heatmap)

        flippable, ratio = self.flippability_calc.get_flippable_pixels()
        self._log_info(f"可翻转性图计算完成")
        self._log_info(f"平均分数: {np.mean(self.flip_map):.3f}")
        self._log_info(f"可翻转像素: {flippable} ({ratio*100:.1f}%)")
        self.status_bar.config(text="可翻转性图计算完成")

    def embed_watermark(self):
        if self.original_img is None:
            messagebox.showerror("错误", "请先加载图像")
            return

        self.embedder.block_size = 8
        self.extractor.block_size = 8
        watermark_text = self.watermark_var.get()

        # 输入密钥
        from tkinter import simpledialog
        key_str = simpledialog.askstring(
            "输入密钥",
            "请输入嵌入密钥（整数）:",
            parent=self.root
        )
        try:
            key = int(key_str)
            self.embedding_key = key
        except ValueError:
            messagebox.showerror("错误", "密钥必须是整数！")
            return

        # 根据是否启用加密，选择转换方式
        if self.encryption_enabled.get():
            self._log_info(f"使用AES加密，密钥: {key}")
            self.watermark_bits = encrypted_text_to_bits(watermark_text, key)
            self._log_info(f"加密后比特流长度: {len(self.watermark_bits)}")
        else:
            self._log_info("未启用加密，直接嵌入")
            self.watermark_bits = text_to_bits(watermark_text)
            self._log_info(f"原始比特流长度: {len(self.watermark_bits)}")

        self.watermarked_img, stats, blocks, flipped = self.embedder.embed(
            self.original_img,
            self.watermark_bits,
            key=key,
            shuffling_enabled=self.shuffling_var.get(),
            flip_map=self.flip_map,
            return_detailed=True
        )

        self.embedded_blocks = blocks
        self.flipped_pixels = flipped
        self.watermarked_panel.display_image(self.watermarked_img)

        self.stats_text.delete(1.0, tk.END)
        text = "嵌入统计:\n\n"
        for key, value in stats.items():
            if isinstance(value, float):
                text += f"{key}: {value:.4f}\n"
            else:
                text += f"{key}: {value}\n"
        text += f"使用的密钥: {self.embedding_key}\n"
        text += f"AES加密: {'启用' if self.encryption_enabled.get() else '禁用'}\n"
        self.stats_text.insert(1.0, text)
        self.status_bar.config(text=f"水印嵌入完成——嵌入容量: {stats['capacity']} bits")
        self._log_info(f"水印嵌入完成，使用密钥: {self.embedding_key}")

    def extract_watermark(self):
        if self.watermarked_img is None:
            messagebox.showerror("错误", "请先嵌入水印")
            return

        from tkinter import simpledialog
        key_str = simpledialog.askstring(
            "密钥验证",
            "请输入提取密钥:",
            parent=self.root
        )
        try:
            key = int(key_str)
        except ValueError:
            messagebox.showerror("错误", "密钥必须是整数！")
            return

        test_img = self.watermarked_img
        original_shape = None

        extracted_bits, stats = self.extractor.extract(
            test_img,
            original_shape=original_shape,
            key=key,
            shuffling_enabled=self.shuffling_var.get(),
            embedded_blocks=self.embedded_blocks,
            expected_length=len(
                self.watermark_bits) if self.watermark_bits is not None else None
        )

        if self.watermark_bits is not None:
            accuracy, ber, nc = self.extractor.verify(
                extracted_bits, self.watermark_bits)

            # 根据加密设置进行解密
            if self.encryption_enabled.get():
                try:
                    extracted_text = bits_to_encrypted_text(
                        extracted_bits, key)
                except Exception as e:
                    extracted_text = f"[解密失败: {str(e)}]"
            else:
                extracted_text = bits_to_text(extracted_bits)

            # 原始水印文本
            original_text = self.watermark_var.get()

            result = f"提取结果:\n\n"
            result += f"原始水印: {original_text}\n"
            result += f"提取水印: {extracted_text}\n"
            result += f"使用的密钥: {key}\n"
            if hasattr(self, 'embedding_key') and self.embedding_key is not None:
                result += f"嵌入时的密钥: {self.embedding_key}\n"
                if key == self.embedding_key:
                    result += "✓ 密钥匹配\n\n"
                else:
                    result += "✗ 密钥不匹配（水印可能无法正确解密）\n\n"
            result += f"准确率: {accuracy:.2f}%\n"
            result += f"比特错误率: {ber:.4f}\n"
            result += f"相关系数: {nc:.4f}\n"
            result += f"AES加密: {'启用' if self.encryption_enabled.get() else '禁用'}\n"
            result += f"提取耗时: {stats['time']:.3f}秒"

            # 如果解密后的文本与原始文本匹配，显示成功信息
            if self.encryption_enabled.get() and extracted_text == original_text:
                result += "\n\n水印解密 内容完全匹配！"
            elif self.encryption_enabled.get():
                result += f"\n\n水印解密 内容部分匹配！"

            messagebox.showinfo("提取验证", result)
            self._log_info(
                f"提取验证——密钥: {key}, 准确率: {accuracy:.2f}%, BER: {ber:.4f}")
        else:
            if self.encryption_enabled.get():
                try:
                    extracted_text = bits_to_encrypted_text(
                        extracted_bits, key)
                except Exception as e:
                    extracted_text = f"[解密失败: {str(e)}]"
            else:
                extracted_text = bits_to_text(extracted_bits)

            result = f"提取的水印: {extracted_text}\n\n使用的密钥: {key}"
            if hasattr(self, 'embedding_key') and self.embedding_key is not None:
                result += f"\n嵌入时的密钥: {self.embedding_key}"
            messagebox.showinfo("提取结果", result)

    def show_difference(self):
        if self.watermarked_img is None or self.original_img is None:
            messagebox.showerror("错误", "需要原始图像和水印图像")
            return

        diff_info = compare_images(self.original_img, self.watermarked_img)
        diff_info['width'] = self.original_img.shape[1]
        diff_info['height'] = self.original_img.shape[0]

        # 展示差异处
        diff_mask = (self.original_img != self.watermarked_img)
        kernel = np.ones((3, 3), np.uint8)
        diff_mask_dilated = cv2.dilate(
            diff_mask.astype(np.uint8), kernel, iterations=1)

        pure_diff = np.zeros(
            (self.original_img.shape[0], self.original_img.shape[1], 3), dtype=np.uint8)
        pure_diff[diff_mask_dilated.astype(bool)] = [0, 0, 255]  # 红色

        changed_pixels = diff_info['changed_pixels']
        total_pixels = self.original_img.shape[0] * self.original_img.shape[1]
        change_percentage = (changed_pixels / total_pixels) * 100

        diff_info['change_percentage'] = change_percentage
        comparison = create_comparison_view(
            [self.original_img, self.watermarked_img, pure_diff]
        )
        ImageComparisonWindow(
            self.root,
            comparison,
            ['原始图像', '含水印图像', '差异图像'],
            diff_info
        )
        self.status_bar.config(
            text=f"差异分析: {diff_info['changed_pixels']:,} 像素修改 ({diff_info['change_percentage']:.6f}%)"
        )
        self._log_info(f"差异分析完成——修改像素: {diff_info['changed_pixels']}, "
                       f"比例: {diff_info['change_percentage']:.4f}%")

    def save_results(self):
        if self.watermarked_img is None:
            messagebox.showerror("错误", "没有可保存的结果")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("BMP files", "*.bmp")]
        )
        if not path:  # 用户取消了保存
            return

        try:
            # 保存含水印图像
            save_image(self.watermarked_img, path)
            self.status_bar.config(text=f"结果已保存到: {Path(path).name}")
            self._log_info(f"图像已保存: {path}")
            messagebox.showinfo("成功", f"图像已保存到:\n{path}")

        except Exception as e:
            messagebox.showerror("错误", f"保存失败:\n{str(e)}")
            self._log_info(f"保存失败: {str(e)}")

    def reset(self):
        self.original_img = None
        self.watermarked_img = None
        self.watermark_bits = None
        self.flip_map = None
        self.embedded_blocks = []
        self.flipped_pixels = []
        self.embedding_key = None
        self.original_panel.clear()
        self.watermarked_panel.clear()
        self.flip_panel.clear()
        self.info_text.delete(1.0, tk.END)
        self.stats_text.delete(1.0, tk.END)

        self.status_bar.config(text="已重置")

    def run_tests1(self):
        from test_algorithm import run_all_tests
        results = run_all_tests()
        stats_dialog = StatsDialog(self.root, "测试结果", results)

    def run_tests2(self):
        # UI展示
        from gui.attack import AttackWindow
        attack_window = AttackWindow(self.root, self)

    def show_about(self):
        # 显示关于对话框
        AboutDialog(self.root)

    def _log_info(self, message):
        # 记录信息到信息文本框
        self.info_text.insert(tk.END, message + "\n")
        self.info_text.see(tk.END)
