import os
import matplotlib.pyplot as plt

# Categoriile de interes și ID-urile lor
categories_of_interest = {
    'cup': 0,
    'bottle': 1,
    'book': 2,
    'knife': 3,
    'bowl': 4
}

# Calea către directoarele de etichete
train_lbl_dir = 'D:/FACULTATE/LICENTA/COCO/4Final/train/labels'
val_lbl_dir = 'D:/FACULTATE/LICENTA/COCO/4Final/val/labels'

# Funcție pentru a contoriza imaginile care conțin doar un singur obiect de interes
def count_images_with_single_object(lbl_dir):
    counts = {category: 0 for category in categories_of_interest.keys()}

    for lbl_file in os.listdir(lbl_dir):
        lbl_path = os.path.join(lbl_dir, lbl_file)

        with open(lbl_path, 'r') as f:
            labels = f.readlines()

        unique_classes = set()
        for label in labels:
            class_id = int(label.split()[0])
            for category, id in categories_of_interest.items():
                if class_id == id:
                    unique_classes.add(category)
                    break

        if len(unique_classes) == 1:
            category = list(unique_classes)[0]
            counts[category] += 1

    return counts

# Contorizarea imaginilor în setul de antrenament și validare
train_counts = count_images_with_single_object(train_lbl_dir)
val_counts = count_images_with_single_object(val_lbl_dir)

# Afișarea rezultatelor
print("Contorizarea imaginilor în setul de antrenament:")
for category, count in train_counts.items():
    print(f"{category}: {count}")

print("\nContorizarea imaginilor în setul de validare:")
for category, count in val_counts.items():
    print(f"{category}: {count}")




# Crearea graficului pentru distribuția claselor în setul de antrenament
plt.figure(figsize=(10, 6))
plt.bar(train_counts.keys(), train_counts.values(), color='skyblue')
plt.xlabel('Categorii')
plt.ylabel('Numărul de imagini')
plt.title('Distribuția claselor în setul de antrenare')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
