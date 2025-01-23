import os
import json
import shutil
import random
import yaml

def coco_to_yolo_format(coco_json_path, images_dir, output_dir, classes_txt):
    # 加载 COCO JSON 文件
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 加载类别文件
    with open(classes_txt, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # 创建目标文件夹结构
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # 获取图片信息
    images = {img["id"]: img for img in coco_data["images"]}

    # 随机分割数据集
    all_image_files = [img["file_name"] for img in coco_data["images"]]
    random.shuffle(all_image_files)
    train_split = int(0.8 * len(all_image_files))  # 80% 训练
    valid_split = int(0.9 * len(all_image_files))  # 10% 验证
    dataset_split = {}
    for i, img_name in enumerate(all_image_files):
        if i < train_split:
            dataset_split[img_name] = "train"
        elif i < valid_split:
            dataset_split[img_name] = "valid"
        else:
            dataset_split[img_name] = "test"

    # 遍历标注
    for ann in coco_data["annotations"]:
        image_info = images[ann["image_id"]]
        img_width = image_info["width"]
        img_height = image_info["height"]
        img_name = image_info["file_name"]
        img_path = os.path.join(images_dir, img_name)

        # 检查图片属于哪个数据集分组
        split = dataset_split.get(img_name, "train")

        # 计算 YOLO 坐标
        x_min, y_min, box_width, box_height = ann["bbox"]
        x_center = x_min + box_width / 2
        y_center = y_min + box_height / 2

        x_center /= img_width
        y_center /= img_height
        box_width /= img_width
        box_height /= img_height

        # 直接使用 category_id
        category_id = ann["category_id"]

        # 保存到 YOLO 标签文件
        txt_file_path = os.path.join(output_dir, split, 'labels', os.path.splitext(img_name)[0] + ".txt")
        with open(txt_file_path, 'a') as f:
            f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

        # 复制图片到目标文件夹
        dest_image_path = os.path.join(output_dir, split, 'images', img_name)
        if not os.path.exists(dest_image_path):  # 避免重复拷贝
            shutil.copy(img_path, dest_image_path)

    print(f"转换完成！数据已整理到 {output_dir}")

def generate_classes_txt(coco_json_path, classes_txt):
    # 从 COCO JSON 自动生成 classes.txt
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    with open(classes_txt, 'w') as f:
        for category in coco_data["categories"]:
            f.write(f"{category['name']}\n")

    print(f"classes.txt 已生成: {classes_txt}")

def generate_yaml(output_dir, classes_txt):
    with open(classes_txt, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    yaml_content = {
        'train': os.path.join(output_dir, 'train', 'images'),
        'val': os.path.join(output_dir, 'valid', 'images'),
        'test': os.path.join(output_dir, 'test', 'images'),
        'nc': len(classes),
        'names': classes
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"data.yaml 已生成: {yaml_path}")

if __name__ == "__main__":
    # 参数设置
    coco_json_path = "/home/c515/lmx/ultralytics-8.2.100/my_datasets/data7/weedcoco.json"  # COCO JSON 文件路径
    images_dir = "/home/c515/lmx/ultralytics-8.2.100/my_datasets/data7/images"               # 图片文件夹路径
    output_dir = "/home/c515/lmx/ultralytics-8.2.100/my_datasets/data7"                # 输出文件夹路径
    classes_txt = "/home/c515/lmx/ultralytics-8.2.100/my_datasets/data7/classes.txt"         # 类别文件，每行一个类别名称

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 自动生成类别文件
    generate_classes_txt(coco_json_path, classes_txt)

    # 执行转换
    coco_to_yolo_format(coco_json_path, images_dir, output_dir, classes_txt)

    # 生成 data.yaml
    generate_yaml(output_dir, classes_txt)
