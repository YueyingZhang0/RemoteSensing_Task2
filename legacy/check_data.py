import numpy as np
from PIL import Image
from pathlib import Path

# 请确保这里的路径是你电脑上的真实路径
fov_path = Path(r"C:\Users\zhang\Desktop\vessel\patches\fov")
mask_path = Path(r"C:\Users\zhang\Desktop\vessel\patches\masks")

def analyze(folder, label):
    first_file = next(folder.iterdir())
    img = Image.open(first_file).convert("L")
    arr = np.asarray(img)
    print(f"--- {label} 分析 ({first_file.name}) ---")
    print(f"最大像素值: {arr.max()}")
    print(f"最小像素值: {arr.min()}")
    print(f"平均像素值: {arr.mean():.2f}")
    print(f"大于127的像素占比: {(arr > 127).mean():.4f}")
    print(f"大于0的像素占比: {(arr > 0).mean():.4f}\n")

analyze(fov_path, "FOV 文件夹")
analyze(mask_path, "Mask 文件夹")