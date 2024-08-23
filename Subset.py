import os
import shutil
import json

# defining categories of interest along with the unique .json code
categories_of_interest = {
    'cup': 6461804,
    'bottle': 6461802,
    'book': 6461836,
    'knife': 6461806,
    'bowl': 6461808
}

#path to the original coco dataset
data_dir = 'D:/FACULTATE/LICENTA/COCO/Full'
train_img_dir = os.path.join(data_dir, 'train2017/img')
train_ann_dir = os.path.join(data_dir, 'train2017/ann')
val_img_dir = os.path.join(data_dir, 'val2017/img')
val_ann_dir = os.path.join(data_dir, 'val2017/ann')

# declaring directory the subset will be created
output_dir = 'D:/FACULTATE/LICENTA/COCO/Subset'
os.makedirs(os.path.join(output_dir, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val/labels'), exist_ok=True)


# transforming .json labels into .txt format
def extract_data(img_dir, ann_dir, data_split):
    output_img_dir = os.path.join(output_dir, f'{data_split}/images')
    output_lbl_dir = os.path.join(output_dir, f'{data_split}/labels')

    for ann_file in os.listdir(ann_dir):
        ann_path = os.path.join(ann_dir, ann_file)
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)


        if 'objects' not in ann_data or 'size' not in ann_data or 'width' not in ann_data['size'] or 'height' not in \
                ann_data['size']:
            print(f"Fișier JSON invalid: {ann_file}")
            continue


        img_file = ann_file.replace('.json', '.txt')

        # verifying if the annotations contain the classes of interest
        relevant_annotations = []
        for obj in ann_data['objects']:
            if obj['classId'] in categories_of_interest.values():
                relevant_annotations.append(obj)

        if not relevant_annotations:
            continue


        src_img_path = os.path.join(img_dir, img_file)
        dst_img_path = os.path.join(output_img_dir, img_file)
        shutil.copy(src_img_path, dst_img_path)

        # saving the labels in .txt format
        label_file = os.path.join(output_lbl_dir, img_file.replace('.jpg', '.txt'))
        labels = set()

        width = ann_data['size']['width']
        height = ann_data['size']['height']

        for obj in relevant_annotations:
            if obj['geometryType'] == 'rectangle':
                bbox = obj['points']['exterior']
                x_min, y_min = bbox[0]
                x_max, y_max = bbox[1]
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                bbox_width = (x_max - x_min) / width
                bbox_height = (y_max - y_min) / height

            elif obj['geometryType'] == 'polygon':
                # Calcularea bbox-ului pentru poligon
                x_min = min(point[0] for point in obj['points']['exterior'])
                y_min = min(point[1] for point in obj['points']['exterior'])
                x_max = max(point[0] for point in obj['points']['exterior'])
                y_max = max(point[1] for point in obj['points']['exterior'])
                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                bbox_width = (x_max - x_min) / width
                bbox_height = (y_max - y_min) / height

            else:
                print(f"Geometrie necunoscută: {obj['geometryType']} în {ann_file}")
                continue

            class_index = list(categories_of_interest.values()).index(obj['classId'])
            label = f"{class_index} {x_center} {y_center} {bbox_width} {bbox_height}\n"
            labels.add(label)


        with open(label_file, 'w') as f:
            for label in labels:
                f.write(label)

extract_data(train_img_dir, train_ann_dir, 'train')

extract_data(val_img_dir, val_ann_dir, 'val')
