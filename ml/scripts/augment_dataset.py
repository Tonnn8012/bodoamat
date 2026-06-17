"""
scripts/augment_dataset.py
==========================
Script untuk melakukan augmentasi data secara offline pada laptop/PC training.
Sangat berguna untuk memperbanyak jumlah dataset kompetisi yang biasanya terbatas.

Fungsi Augmentasi:
1. Horizontal Flip (Flip kiri-kanan & penyesuaian label YOLO)
2. Brightness & Contrast adjustment (Mensimulasikan perubahan cuaca/cahaya matahari)
3. Gaussian Blur (Mensimulasikan motion blur akibat getaran drone)
4. Salt & Pepper Noise (Mensimulasikan noise sensor kamera murah)
5. Slight Translation (Pergeseran kecil X-Y untuk melatih toleransi centering)

Cara Pakai:
-----------
Jalankan di direktori dataset Anda:
python scripts/augment_dataset.py --input_dir data/dataset --output_dir data/dataset_augmented --multiplier 3
"""

import os
import cv2
import numpy as np
import argparse
import random
import glob
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DatasetAugmentor")


def read_yolo_labels(label_path: str) -> list[list[float]]:
    """Membaca file label YOLO dan mengembalikan list bounding box."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append([cls_id, cx, cy, w, h])
    return boxes


def write_yolo_labels(label_path: str, boxes: list[list[float]]) -> None:
    """Menyimpan list bounding box ke format file label YOLO."""
    with open(label_path, "w") as f:
        for box in boxes:
            f.write(f"{int(box[0])} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")


def augment_flip(image: np.ndarray, boxes: list[list[float]]) -> tuple[np.ndarray, list[list[float]]]:
    """Melakukan flip horizontal pada gambar dan memperbarui koordinat X-center."""
    flipped_image = cv2.flip(image, 1)
    flipped_boxes = []
    for box in boxes:
        cls_id, cx, cy, w, h = box
        # Flip horizontal: cx_baru = 1.0 - cx
        new_cx = 1.0 - cx
        flipped_boxes.append([cls_id, new_cx, cy, w, h])
    return flipped_image, flipped_boxes


def augment_brightness_contrast(image: np.ndarray) -> np.ndarray:
    """Mengubah brightness dan contrast secara acak."""
    alpha = random.uniform(0.7, 1.3)  # Kontras
    beta = random.randint(-30, 30)    # Kecerahan
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def augment_blur(image: np.ndarray) -> np.ndarray:
    """Menambahkan motion blur / gaussian blur acak."""
    kernel_size = random.choice([3, 5])
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def augment_noise(image: np.ndarray) -> np.ndarray:
    """Menambahkan Gaussian/Salt-and-Pepper noise acak pada gambar."""
    noisy = image.copy()
    h, w, c = noisy.shape
    # Salt and pepper noise
    s_vs_p = 0.5
    amount = random.uniform(0.005, 0.02)
    
    # Salt mode (titik putih)
    num_salt = np.ceil(amount * image.size * s_vs_p / c)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 255

    # Pepper mode (titik hitam)
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p) / c)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy[coords[0], coords[1], :] = 0
    return noisy


def augment_translation(
    image: np.ndarray, boxes: list[list[float]], max_shift: float = 0.05
) -> tuple[np.ndarray, list[list[float]]]:
    """Menggeser gambar secara horizontal/vertikal secara acak (max_shift relative)."""
    h, w = image.shape[:2]
    tx_pct = random.uniform(-max_shift, max_shift)
    ty_pct = random.uniform(-max_shift, max_shift)
    
    tx = int(tx_pct * w)
    ty = int(ty_pct * h)
    
    # Buat matriks translasi
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    translated_boxes = []
    for box in boxes:
        cls_id, cx, cy, box_w, box_h = box
        new_cx = cx + tx_pct
        new_cy = cy + ty_pct
        
        # Kliping agar box tetap berada di dalam frame
        if 0 < new_cx < 1.0 and 0 < new_cy < 1.0:
            # Sesuaikan lebar/tinggi jika terpotong batas tepi
            half_w = box_w / 2
            half_h = box_h / 2
            
            x1 = max(0.0, new_cx - half_w)
            y1 = max(0.0, new_cy - half_h)
            x2 = min(1.0, new_cx + half_w)
            y2 = min(1.0, new_cy + half_h)
            
            final_w = x2 - x1
            final_h = y2 - y1
            final_cx = x1 + final_w / 2
            final_cy = y1 + final_h / 2
            
            # Hanya simpan jika area box yang tersisa masih cukup besar (misal min 10% dari asli)
            if (final_w * final_h) > (box_w * box_h * 0.1):
                translated_boxes.append([cls_id, final_cx, final_cy, final_w, final_h])
                
    return translated_image, translated_boxes


def process_dataset(input_dir: str, output_dir: str, multiplier: int) -> None:
    """Melakukan iterasi dan augmentasi pada seluruh dataset."""
    subdirs = ["train", "val"]
    
    for subdir in subdirs:
        img_in_path = os.path.join(input_dir, "images", subdir)
        lbl_in_path = os.path.join(input_dir, "labels", subdir)
        
        img_out_path = os.path.join(output_dir, "images", subdir)
        lbl_out_path = os.path.join(output_dir, "labels", subdir)
        
        os.makedirs(img_out_path, exist_ok=True)
        os.makedirs(lbl_out_path, exist_ok=True)
        
        img_files = glob.glob(os.path.join(img_in_path, "*.[jJ][pP][gG]")) + \
                    glob.glob(os.path.join(img_in_path, "*.[jJ][pP][eE][gG]")) + \
                    glob.glob(os.path.join(img_in_path, "*.[pP][nN][gG]"))
        
        logger.info(f"Membaca {len(img_files)} gambar dari {img_in_path}...")
        
        count = 0
        for img_path in img_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            ext = os.path.splitext(img_path)[1]
            
            # Cari file label yang bersesuaian
            lbl_path = os.path.join(lbl_in_path, base_name + ".txt")
            if not os.path.exists(lbl_path):
                logger.warning(f"Label tidak ditemukan untuk gambar: {img_path}. Dilewati.")
                continue
                
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            boxes = read_yolo_labels(lbl_path)
            
            # 1. Simpan salinan asli ke folder output
            cv2.imwrite(os.path.join(img_out_path, f"{base_name}{ext}"), image)
            write_yolo_labels(os.path.join(lbl_out_path, f"{base_name}.txt"), boxes)
            count += 1
            
            # 2. Buat augmentasi
            for i in range(multiplier - 1):
                aug_img = image.copy()
                aug_boxes = [box.copy() for box in boxes]
                
                # Gunakan probabilitas untuk menerapkan kombinasi transformasi
                
                # Flip horizontal (50% chance)
                if random.random() > 0.5:
                    aug_img, aug_boxes = augment_flip(aug_img, aug_boxes)
                    
                # Translasi kecil (50% chance)
                if random.random() > 0.5:
                    aug_img, aug_boxes = augment_translation(aug_img, aug_boxes)
                    
                # Brightness & contrast (80% chance)
                if random.random() > 0.2:
                    aug_img = augment_brightness_contrast(aug_img)
                    
                # Blur atau Noise (50% chance)
                if random.random() > 0.5:
                    if random.random() > 0.5:
                        aug_img = augment_blur(aug_img)
                    else:
                        aug_img = augment_noise(aug_img)
                
                # Tulis file hasil augmentasi
                aug_name = f"{base_name}_aug{i}"
                cv2.imwrite(os.path.join(img_out_path, f"{aug_name}{ext}"), aug_img)
                write_yolo_labels(os.path.join(lbl_out_path, f"{aug_name}.txt"), aug_boxes)
                count += 1
                
        logger.info(f"Subdir '{subdir}' selesai. Total file (asli + augmentasi): {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Augmentasi Offline YOLO Dataset untuk VTOL KRTI 2026")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder dataset asli (berisi images/ & labels/)")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder output dataset hasil augmentasi")
    parser.add_argument("--multiplier", type=int, default=3, help="Jumlah total kelipatan data per file asli (misal: 3)")
    
    args = parser.parse_args()
    
    logger.info("=== STRAT DATASET AUGMENTATION ===")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Multiplier: {args.multiplier}x")
    
    process_dataset(args.input_dir, args.output_dir, args.multiplier)
    logger.info("=== AUGMENTATION SELESAI ===")


if __name__ == "__main__":
    main()
