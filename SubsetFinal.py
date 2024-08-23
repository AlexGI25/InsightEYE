import os
import shutil

# Setează pragurile
max_boxes = 10
min_box_area = 0.01  # adjustments for reducing the volume of the dataset

# paths to directories
train_img_dir = 'D:/FACULTATE/LICENTA/COCO/1Subset/train/images'
train_lbl_dir = 'D:/FACULTATE/LICENTA/COCO/1Subset/train/labels'
filtered_img_dir = 'D:/FACULTATE/LICENTA/COCO/4Final/train/images'
filtered_lbl_dir = 'D:/FACULTATE/LICENTA/COCO/4Final/train/labels'


os.makedirs(filtered_img_dir, exist_ok=True)
os.makedirs(filtered_lbl_dir, exist_ok=True)


def filter_images_and_labels(img_dir, lbl_dir, filtered_img_dir, filtered_lbl_dir, max_boxes, min_box_area):
    for lbl_file in os.listdir(lbl_dir):
        lbl_path = os.path.join(lbl_dir, lbl_file)
        with open(lbl_path, 'r') as f:
            labels = f.readlines()

        if len(labels) > max_boxes:
            continue

        valid = True
        for label in labels:
            parts = label.strip().split()
            class_id = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = map(float, parts[1:])

            if bbox_width * bbox_height < min_box_area:
                valid = False
                break

        if valid:
            img_file = lbl_file.replace('.txt', '.jpg')
            src_img_path = os.path.join(img_dir, img_file)
            dst_img_path = os.path.join(filtered_img_dir, img_file)
            dst_lbl_path = os.path.join(filtered_lbl_dir, lbl_file)
            shutil.copy(src_img_path, dst_img_path)
            shutil.copy(lbl_path, dst_lbl_path)


# apply filtering
filter_images_and_labels(train_img_dir, train_lbl_dir, filtered_img_dir, filtered_lbl_dir, max_boxes, min_box_area)
