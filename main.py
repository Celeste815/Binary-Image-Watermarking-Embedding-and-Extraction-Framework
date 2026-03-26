"二值图像水印系统"
from gui.main_window import MainWindow
import tkinter as tk
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    root = tk.Tk()
    try:
        root.iconbitmap(default='icon.ico')
    except:
        pass

    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
