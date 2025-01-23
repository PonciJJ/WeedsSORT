import os
import json
import cv2
import shutil
import random
import yaml

def convert_via_to_yolo_and_split(image_dir, json_dir, output_dir, target_size=640, split_ratios=(0.85, 0.1, 0.05)):
    """
    将VGG JSON格式的标注数据（以 .jpg.json 命名）转换为YOLO格式，统一图像大小，划分为训练、验证和测试集。
    
    Args:
        image_dir (str): 原始图像文件夹路径。
        json_dir (str): VGG JSON标注文件夹路径。
        output_dir (str): 输出数据集根目录。
        target_size (int): 图像缩放目标宽或高的尺寸。
        split_ratios (tuple): 数据划分比例，默认为(0.7, 0.2, 0.1)。
    """
    # 创建输出文件夹结构
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # 获取所有图像和对应的JSON文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith('.jpg.json')]

    # 确保图像和JSON文件匹配
    paired_files = [f for f in image_files if f + ".json" in json_files]
    random.shuffle(paired_files)  # 打乱顺序

    if len(paired_files) == 0:
        raise ValueError("未找到匹配的图像和JSON文件，请检查文件夹内容是否正确。")
    
    # 数据划分
    total_files = len(paired_files)
    train_count = int(total_files * split_ratios[0])
    valid_count = int(total_files * split_ratios[1])
    
    train_files = paired_files[:train_count]
    valid_files = paired_files[train_count:train_count + valid_count]
    test_files = paired_files[train_count + valid_count:]
    
    # 处理每个数据集
    for split, files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        for file in files:
            # 图像路径和JSON路径
            image_path = os.path.join(image_dir, file)
            json_path = os.path.join(json_dir, file + ".json")
            
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping {file}: Unable to read image.")
                continue
            
            h, w = image.shape[:2]
            
            # 计算缩放比例
            if w > h:
                scale = target_size / w
                new_w, new_h = target_size, int(h * scale)
            else:
                scale = target_size / h
                new_w, new_h = int(w * scale), target_size
            
            # 缩放图像
            resized_image = cv2.resize(image, (new_w, new_h))
            dest_image_path = os.path.join(output_dir, split, 'images', file)
            cv2.imwrite(dest_image_path, resized_image)
            
            # 转换JSON到YOLO格式
            with open(json_path, 'r') as f:
                via_data = json.load(f)
            
            for key, value in via_data.items():
                regions = value.get("regions", [])
                yolo_annotations = []
                
                for region in regions:
                    shape = region["shape_attributes"]
                    x, y, width, height = shape["x"], shape["y"], shape["width"], shape["height"]
                    
                    # 转换为 YOLO 格式 (class_id, x_center, y_center, width, height)
                    x_center = (x + width / 2) / w
                    y_center = (y + height / 2) / h
                    norm_width = width / w
                    norm_height = height / h
                    
                    # 按缩放比例调整
                    x_center *= new_w / target_size
                    y_center *= new_h / target_size
                    norm_width *= new_w / target_size
                    norm_height *= new_h / target_size
                    
                    # 默认类别为0（可根据实际类别映射调整）
                    class_id = 0
                    yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                
                # 保存为 YOLO 格式
                dest_label_path = os.path.join(output_dir, split, 'labels', os.path.splitext(file)[0] + ".txt")
                with open(dest_label_path, 'w') as f:
                    f.write("\n".join(yolo_annotations))
    
    # 创建 data.yaml 文件
    yaml_data = {
        'train': os.path.abspath(os.path.join(output_dir, 'train/images')),
        'val': os.path.abspath(os.path.join(output_dir, 'valid/images')),
        'test': os.path.abspath(os.path.join(output_dir, 'test/images')),
        'nc': 1,  # 类别数量，根据需要更新
        'names': ['class0']  # 类别名称，根据需要更新
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_data, f)
    
    print(f"Dataset processed and saved in {output_dir} with train, valid, and test splits.")

# 使用示例
image_dir = "/home/c515/lmx/ultralytics-8.3.35/my_datasets/data9/data_org/ds/img"  # 原始图像文件夹路径
json_dir = "/home/c515/lmx/ultralytics-8.3.35/my_datasets/data9/data_org/ds/ann"  # VGG JSON标注文件夹路径（.jpg.json格式）
output_dir = "/home/c515/lmx/ultralytics-8.3.35/my_datasets/data9/data_processed"  # 输出数据集路径
target_size = 640  # 图像缩放目标尺寸
split_ratios = (0.85, 0.1, 0.05)  # 训练、验证、测试集比例

convert_via_to_yolo_and_split(image_dir, json_dir, output_dir, target_size, split_ratios)
