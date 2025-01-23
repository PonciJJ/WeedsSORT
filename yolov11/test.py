from ultralytics import YOLO
import os

# 加载模型
model = YOLO("runs/detect/train5/weights/best.pt")

# 推理数据集
results = model("/home/c515/lmx/ultralytics-8.3.35/temp_datasets/temp2/test/images")

# 固定保存路径
save_dir = "refer/refer7"
os.makedirs(save_dir, exist_ok=True)  

# 保存推理结果
for i, result in enumerate(results):
    save_path = os.path.join(save_dir, f"result_{i}.jpg") 
    result.save(save_path)

