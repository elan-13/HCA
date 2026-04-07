import os
import random
import shutil

import cv2
from PIL import Image

# ==============================
# CONFIG
# ==============================

INPUT_ROOT = r"D:\SEM-8\HCA-PROJECT\SkinDisease"
OUTPUT_DIR = "Dataset"

SELECTED_CLASSES = [
    "Acne",
    "Eczema",
    "Psoriasis",
    "SkinCancer",
    "Vitiligo",
    "Warts",
]

# NEW SPLITS
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

IMG_SIZE = 224
RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==============================
# START
# ==============================

random.seed(RANDOM_SEED)

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Create folders
for split in ["train", "val", "test"]:
    for cls in SELECTED_CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

# ==============================
# PROCESS
# ==============================

for class_name in SELECTED_CLASSES:
    print(f"\nProcessing class: {class_name}")

    all_images = []

    # Merge original train + test
    for split_name in ["train", "test"]:
        class_path = os.path.join(INPUT_ROOT, split_name, class_name)

        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):
            full_path = os.path.join(class_path, file_name)
            _, ext = os.path.splitext(file_name)

            if os.path.isfile(full_path) and ext.lower() in VALID_EXTENSIONS:
                all_images.append(full_path)

    if not all_images:
        print(f"[WARN] No images for {class_name}")
        continue

    random.shuffle(all_images)

    total = len(all_images)

    train_end = int(TRAIN_SPLIT * total)
    val_end = train_end + int(VAL_SPLIT * total)

    train_imgs = all_images[:train_end]
    val_imgs = all_images[train_end:val_end]
    test_imgs = all_images[val_end:]

    print(f"Total: {total}")
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    def process_and_save(img_list, split):
        count = 0
        for idx, img_path in enumerate(img_list):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                out_name = f"{class_name}_{idx:04d}.jpg"
                save_path = os.path.join(OUTPUT_DIR, split, class_name, out_name)

                Image.fromarray(img).save(save_path, quality=92)
                count += 1

            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")

        print(f"Saved {count} in {split}/{class_name}")

    process_and_save(train_imgs, "train")
    process_and_save(val_imgs, "val")
    process_and_save(test_imgs, "test")

print(f"\n✅ Dataset ready at: {OUTPUT_DIR}")