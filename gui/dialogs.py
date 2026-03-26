import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import webbrowser
from datetime import datetime


class StatsDialog(tk.Toplevel):
    def __init__(self, parent, title, stats_data):
        super().__init__(parent)

        self.title(title)
        self.geometry("500x400")
        self.resizable(True, True)

        self.transient(parent)
        self.grab_set()
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        self._create_widgets(stats_data)
        self.wait_window()

    def _create_widgets(self, stats_data):
        """创建控件"""
        # 主框架
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(
            main_frame,
            text="测试结果统计",
            font=('Arial', 12, 'bold')
        )
        title_label.pack(pady=(0, 10))

        # 创建文本框显示统计信息
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # 文本框和滚动条
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(
            text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 插入统计信息
        self._insert_stats(text_widget, stats_data)

        # 按钮框架
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        # 关闭按钮
        ttk.Button(btn_frame, text="关闭",
                   command=self.destroy).pack(side=tk.RIGHT)

        # 导出按钮
        ttk.Button(btn_frame, text="导出为文本",
                   command=lambda: self._export_stats(stats_data)).pack(side=tk.RIGHT, padx=(0, 10))

    def _insert_stats(self, text_widget, stats_data):
        """插入统计信息到文本框"""
        text_widget.insert(tk.END, "=" * 60 + "\n")
        text_widget.insert(tk.END, "二值图像水印系统测试报告\n")
        text_widget.insert(tk.END, "=" * 60 + "\n\n")

        if isinstance(stats_data, dict):
            for key, value in stats_data.items():
                text_widget.insert(tk.END, f"【{key}】\n")
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, float):
                            text_widget.insert(
                                tk.END, f"  {subkey}: {subvalue:.4f}\n")
                        else:
                            text_widget.insert(
                                tk.END, f"  {subkey}: {subvalue}\n")
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        text_widget.insert(tk.END, f"  测试{i+1}: {item}\n")
                else:
                    if isinstance(value, float):
                        text_widget.insert(tk.END, f"  {key}: {value:.4f}\n")
                    else:
                        text_widget.insert(tk.END, f"  {key}: {value}\n")
                text_widget.insert(tk.END, "\n")
        else:
            text_widget.insert(tk.END, str(stats_data))

        text_widget.config(state=tk.DISABLED)

    def _export_stats(self, stats_data):
        """导出统计信息到文件"""
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"),
                       ("JSON文件", "*.json"), ("所有文件", "*.*")]
        )

        if path:
            try:
                if path.endswith('.json'):
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(stats_data, f, indent=2, ensure_ascii=False)
                else:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write("=" * 60 + "\n")
                        f.write(f"二值图像水印系统测试报告\n")
                        f.write(
                            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(str(stats_data))

                messagebox.showinfo("导出成功", f"统计信息已导出到:\n{path}")
            except Exception as e:
                messagebox.showerror("导出失败", f"导出时出错: {str(e)}")


class AboutDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("关于")
        self.geometry("500x300")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (400 // 2)
        y = (self.winfo_screenheight() // 2) - (300 // 2)
        self.geometry(f'+{x}+{y}')
        self._create_widgets()
        self.wait_window()

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(
            main_frame,
            text="二值图像水印嵌入与提取系统",
            font=('Arial', 16, 'bold')
        )
        title_label.pack()

        desc_text = (
            "主要功能：\n""• 可翻转性分析\n""• 混洗\n"
            "• 水印嵌入\n""• 水印提取\n"
            "• 性能测试""• 攻击测试"
        )
        desc_label = ttk.Label(main_frame, text=desc_text, justify=tk.CENTER)
        desc_label.pack(pady=10)

        author_label = ttk.Label(
            main_frame,
            text="版权所有 © 2026",
            font=('Arial', 8),
            foreground='gray'
        )
        author_label.pack(pady=(10, 0))

        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(btn_frame, text="使用说明文档",
                   command=self._open_documentation).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭",
                   command=self.destroy).pack(side=tk.RIGHT, padx=5)

    def _open_documentation(self):
        messagebox.showinfo("提示", "已在系统列表中——“二值图像水印嵌入与提取系统使用方法.pdf”")
