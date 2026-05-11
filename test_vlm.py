"""诊断脚本: 测试本地 VLM 推理 (使用修复后的 local_vlm.py)"""
import sys
import traceback
from pathlib import Path

print("=== VLM 推理诊断 (修复版) ===\n")

# 1. 加载模型
print("[1] 加载模型...")
from local_vlm import LocalVLM
vlm = LocalVLM("F:/qwen3_5")
vlm.load()
print("    OK\n")

# 2. 找一张测试图片
img_dir = Path("F:/datasetpic")
test_images = sorted([f for f in img_dir.iterdir()
    if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and f.is_file()])
test_img = test_images[0]
print(f"[2] 测试图片: {test_img.name}\n")

# 3. 运行完整推理
print("[3] 运行 label_image (完整推理)...")
try:
    labels = vlm.label_image(str(test_img))
    print(f"    OK!")
    print(f"    标签结果:")
    for k, v in labels.items():
        print(f"      {k}: {v}")
    print("\n=== 诊断完成，全部通过 ===")
except Exception as e:
    print(f"    失败: {e}")
    traceback.print_exc()
