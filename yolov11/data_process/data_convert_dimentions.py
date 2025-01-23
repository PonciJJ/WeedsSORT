import os
import cv2
import shutil
import random
import yaml

def scale_images_and_split(image_dir, label_dir, output_dir, target_size=640, split_ratios=(0.7, 0.2, 0.1)):
    """
    将图像缩小到指定尺寸，保持标注文件不变，并按比例划分为训练集、验证集和测试集。
    
    Args:
        image_dir (str): 原始图像文件夹路径。
        label_dir (str): YOLO标注文件夹路径。
        output_dir (str): 输出数据集根目录。
        target_size (int): 缩放目标宽或高的尺寸。
        split_ratios (tuple): 数据划分比例，默认为(0.7, 0.2, 0.1)。
    """
    # 创建输出文件夹结构
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # 获取所有图像文件和对应的标注文件
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
    
    # 确保图像和标注一一对应
    paired_files = [f for f in image_files if os.path.splitext(f)[0] + ".txt" in label_files]
    random.shuffle(paired_files)  # 打乱顺序
    
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
            # 图像路径和标注路径
            image_path = os.path.join(image_dir, file)
            label_path = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")
            
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
            
            # 复制标注文件
            dest_label_path = os.path.join(output_dir, split, 'labels', os.path.splitext(file)[0] + ".txt")
            shutil.copy(label_path, dest_label_path)
    
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
image_dir = "/home/c515/lmx/ultralytics-8.3.35/my_datasets/data10/CottonWeedDet12/data_org/weedImages"  # 原始图像文件夹路径
label_dir = "/home/c515/lmx/ultralytics-8.3.35/my_datasets/data10/CottonWeedDet12/data_org/annotation_YOLO_txt"  # YOLO标注文件夹路径
output_dir = "/home/c515/lmx/ultralytics-8.3.35/my_datasets/data10/CottonWeedDet12/data_processed"  # 输出数据集路径
target_size = 640  # 图像缩放目标尺寸
split_ratios = (0.85, 0.1, 0.05)  # 训练、验证、测试集比例

scale_images_and_split(image_dir, label_dir, output_dir, target_size, split_ratios)
