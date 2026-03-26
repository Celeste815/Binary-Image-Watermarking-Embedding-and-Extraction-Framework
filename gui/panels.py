import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


class ImagePanel(ttk.Frame):
    def __init__(self, parent, width=400, height=400, bg='gray'):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.bg = bg
        self.image = None
        self.photo = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._create_widgets()
        self._bind_events()

    def _create_widgets(self):
        """创建控件"""
        # 画布
        self.canvas = tk.Canvas(
            self, width=self.width, height=self.height,
            bg=self.bg, highlightthickness=0
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 信息标签
        self.info_label = ttk.Label(self, text="", font=('Arial', 8))
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始化画布内容
        self._draw_placeholder()

    def _bind_events(self):
        """绑定事件"""
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_move)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

    def _draw_placeholder(self):
        """绘制占位符"""
        self.canvas.create_rectangle(
            10, 10, self.width-10, self.height-10,
            outline='#ccc', width=1, dash=(5, 5)
        )
        self.canvas.create_text(
            self.width//2, self.height//2,
            text="无图像", fill='#999', font=('Arial', 14)
        )

    def display_image(self, img):
        if img is None:
            self.clear()
            return

        self.image = img
        h, w = self.image.shape[:2]
        zoom_x = self.width / w
        zoom_y = self.height / h
        self.zoom_level = min(1.0, min(zoom_x, zoom_y))
        self.pan_x = 0
        self.pan_y = 0
        self._update_display()

    def _update_display(self):
        if self.image is None:
            return

        self.canvas.delete("all")
        if len(self.image.shape) == 3:
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(img_rgb)
        if self.zoom_level != 1.0:
            new_size = (
                int(pil_img.width * self.zoom_level),
                int(pil_img.height * self.zoom_level)
            )
            pil_img = pil_img.resize(new_size, Image.NEAREST)

        self.photo = ImageTk.PhotoImage(pil_img)
        x = self.width//2 - pil_img.width//2 + self.pan_x
        y = self.height//2 - pil_img.height//2 + self.pan_y
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)

        h, w = self.image.shape[:2]
        self.info_label.config(
            text=f"{w}x{h} | 缩放: {self.zoom_level:.2f}x"
        )

    def clear(self):
        """清空显示"""
        self.image = None
        self.photo = None
        self.canvas.delete("all")
        self._draw_placeholder()
        self.info_label.config(text="")

    def _on_mouse_down(self, event):
        """鼠标按下事件"""
        self.last_x = event.x
        self.last_y = event.y

    def _on_mouse_move(self, event):
        """鼠标移动事件"""
        if self.image is None:
            return

        dx = event.x - self.last_x
        dy = event.y - self.last_y

        self.pan_x += dx
        self.pan_y += dy

        self.last_x = event.x
        self.last_y = event.y

        self._update_display()

    def _on_mouse_wheel(self, event):
        """鼠标滚轮事件"""
        if self.image is None:
            return

        # 缩放
        if event.delta > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level *= 0.9

        # 限制缩放范围
        self.zoom_level = max(0.1, min(10.0, self.zoom_level))

        self._update_display()

    def _on_mouse_motion(self, event):
        """鼠标移动事件 - 显示像素值"""
        if self.image is None:
            return

        # 计算图像坐标
        pil_width = self.image.shape[1] * self.zoom_level
        pil_height = self.image.shape[0] * self.zoom_level

        img_x = (event.x - self.pan_x -
                 (self.width - pil_width)//2) / self.zoom_level
        img_y = (event.y - self.pan_y -
                 (self.height - pil_height)//2) / self.zoom_level

        if 0 <= img_x < self.image.shape[1] and 0 <= img_y < self.image.shape[0]:
            x, y = int(img_x), int(img_y)
            value = self.image[y, x]

            if len(self.image.shape) == 2:
                self.info_label.config(
                    text=f"({x}, {y}) = {value} | 缩放: {self.zoom_level:.2f}x"
                )
            else:
                self.info_label.config(
                    text=f"({x}, {y}) = {value} | 缩放: {self.zoom_level:.2f}x"
                )

    def _on_mouse_leave(self, event):
        """鼠标离开事件"""
        if self.image is not None:
            h, w = self.image.shape[:2]
            self.info_label.config(
                text=f"{w}x{h} | 缩放: {self.zoom_level:.2f}x")
