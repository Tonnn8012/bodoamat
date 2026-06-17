"""
scripts/train.py
=================
Script training YOLO untuk KRTI VTOL 2026.

CARA PAKAI (tinggal jalankan salah satu):
-----------------------------------------
# Training standar (YOLOv8n, 100 epoch)
python scripts/train.py

# Training dengan epoch lebih banyak
python scripts/train.py --epochs 200

# Resume training dari checkpoint terakhir
python scripts/train.py --resume

# Dry-run (validasi config tanpa training sungguhan)
python scripts/train.py --epochs 1 --dry-run

# Training dengan model yang lebih besar (jika hardware kuat)
python scripts/train.py --model yolov8s

CATATAN PENTING:
- Training dilakukan di LAPTOP/PC dengan GPU, BUKAN di Orange Pi 4.
- Setelah training selesai, model terbaik otomatis disalin ke models/best.pt
- Untuk deploy ke Orange Pi 4, jalankan scripts/export_model.py
"""

from __future__ import annotations
import argparse
import os
import sys
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VTOLTrainer")

# Root directory proyek (ml/)
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data" / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training YOLO untuk KRTI VTOL 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python scripts/train.py                     # Default training
  python scripts/train.py --epochs 200        # 200 epoch
  python scripts/train.py --resume            # Lanjut dari checkpoint
  python scripts/train.py --model yolov8s     # Model lebih besar
  python scripts/train.py --imgsz 640         # Resolusi training 640
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        help="Model YOLO yang digunakan: yolov8n (nano/default), yolov8s (small). "
             "Untuk Orange Pi 4, GUNAKAN yolov8n saja.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Jumlah epoch training (default: 100).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Ukuran batch. -1 = auto-detect berdasarkan VRAM GPU (default: -1).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Resolusi input training. 320 = optimal untuk Orange Pi 4 CPU, "
             "640 = akurasi lebih tinggi tapi lebih lambat saat inferensi.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Lanjutkan training dari checkpoint terakhir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Validasi config tanpa training sungguhan (1 epoch, 1 batch).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience. Training berhenti jika mAP tidak naik "
             "selama N epoch berturut-turut (default: 20).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device training: '' = auto, '0' = GPU 0, 'cpu' = CPU only.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Jumlah worker untuk data loading (default: 4).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Nama experiment. Kosong = auto-generate dari timestamp.",
    )
    return parser.parse_args()


def validate_dataset() -> bool:
    """Memvalidasi bahwa dataset ada dan tidak kosong."""
    images_train = DATA_DIR / "images" / "train"
    images_val = DATA_DIR / "images" / "val"
    labels_train = DATA_DIR / "labels" / "train"
    labels_val = DATA_DIR / "labels" / "val"

    errors = []

    for folder in [images_train, images_val, labels_train, labels_val]:
        if not folder.exists():
            errors.append(f"  ✗ Folder tidak ditemukan: {folder}")
        elif not any(folder.iterdir()):
            errors.append(f"  ⚠ Folder kosong: {folder}")

    if errors:
        logger.warning("Validasi dataset menemukan masalah:")
        for err in errors:
            logger.warning(err)

        # Cek apakah ada gambar sama sekali
        train_images = list((images_train).glob("*.[jJ][pP][gG]")) + \
                       list((images_train).glob("*.[pP][nN][gG]"))
        if not train_images:
            logger.error(
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  DATASET KOSONG!                                        ║\n"
                "║                                                          ║\n"
                "║  Langkah yang harus dilakukan:                           ║\n"
                "║  1. Ambil foto gate, red_box, landing_pad               ║\n"
                "║  2. Labeling dengan Roboflow/CVAT/LabelImg              ║\n"
                "║  3. Taruh gambar di: data/dataset/images/train/          ║\n"
                "║  4. Taruh label di:  data/dataset/labels/train/          ║\n"
                "║  5. Jalankan lagi: python scripts/train.py              ║\n"
                "╚══════════════════════════════════════════════════════════╝"
            )
            return False

    # Cek konsistensi jumlah gambar vs label
    for split in ["train", "val"]:
        img_folder = DATA_DIR / "images" / split
        lbl_folder = DATA_DIR / "labels" / split

        if img_folder.exists() and lbl_folder.exists():
            img_files = set(
                p.stem for p in img_folder.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            )
            lbl_files = set(
                p.stem for p in lbl_folder.iterdir()
                if p.suffix.lower() == ".txt"
            )

            imgs_without_labels = img_files - lbl_files
            labels_without_imgs = lbl_files - img_files

            if imgs_without_labels:
                logger.warning(
                    f"  ⚠ {split}: {len(imgs_without_labels)} gambar tanpa label "
                    f"(contoh: {list(imgs_without_labels)[:3]})"
                )
            if labels_without_imgs:
                logger.warning(
                    f"  ⚠ {split}: {len(labels_without_imgs)} label tanpa gambar "
                    f"(contoh: {list(labels_without_imgs)[:3]})"
                )

            if img_files:
                logger.info(f"  ✓ {split}: {len(img_files)} gambar, {len(lbl_files)} label")

    return True


def validate_configs() -> bool:
    """Memvalidasi bahwa file konfigurasi tersedia."""
    dataset_yaml = CONFIGS_DIR / "dataset.yaml"
    hyp_yaml = CONFIGS_DIR / "hyp_vtol.yaml"
    classes_json = CONFIGS_DIR / "classes.json"

    all_ok = True
    for cfg in [dataset_yaml, hyp_yaml, classes_json]:
        if cfg.exists():
            logger.info(f"  ✓ Config ditemukan: {cfg.name}")
        else:
            logger.error(f"  ✗ Config TIDAK ditemukan: {cfg}")
            all_ok = False

    return all_ok


def print_class_info():
    """Menampilkan informasi kelas dari classes.json."""
    classes_json = CONFIGS_DIR / "classes.json"
    if classes_json.exists():
        with open(classes_json, "r") as f:
            data = json.load(f)
        logger.info("Kelas deteksi yang akan dilatih:")
        for idx, name in data.get("classes", {}).items():
            desc = data.get("descriptions", {}).get(name, "")
            logger.info(f"  [{idx}] {name} — {desc}")


def main():
    args = parse_args()

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║   KRTI VTOL 2026 — YOLO Training Pipeline   ║")
    print("║   Target Deploy: Orange Pi 4 (RK3399 CPU)   ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # 1. Validasi konfigurasi
    logger.info("=== Langkah 1/4: Validasi Konfigurasi ===")
    if not validate_configs():
        logger.error("Konfigurasi tidak lengkap. Periksa folder configs/")
        sys.exit(1)

    print_class_info()

    # 2. Validasi dataset
    logger.info("=== Langkah 2/4: Validasi Dataset ===")
    if not validate_dataset():
        logger.error("Dataset belum siap. Silakan siapkan dataset terlebih dahulu.")
        sys.exit(1)

    # 3. Import Ultralytics (lazy import agar validasi di atas jalan tanpa GPU)
    logger.info("=== Langkah 3/4: Memuat Ultralytics YOLO ===")
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error(
            "Library 'ultralytics' belum terinstall!\n"
            "Jalankan: pip install ultralytics"
        )
        sys.exit(1)

    # 4. Mulai training
    logger.info("=== Langkah 4/4: Memulai Training ===")

    dataset_yaml = str(CONFIGS_DIR / "dataset.yaml")

    # Buat nama experiment
    if args.name:
        exp_name = args.name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"vtol_{args.model}_{args.imgsz}_{timestamp}"

    # Muat model (pretrained dari COCO)
    model_name = args.model
    if not model_name.endswith(".pt"):
        model_name += ".pt"
    
    logger.info(f"Model base       : {model_name}")
    logger.info(f"Resolusi input   : {args.imgsz}×{args.imgsz}")
    logger.info(f"Epoch            : {args.epochs}")
    logger.info(f"Batch size       : {'auto' if args.batch == -1 else args.batch}")
    logger.info(f"Patience         : {args.patience}")
    logger.info(f"Experiment name  : {exp_name}")
    logger.info(f"Dataset config   : {dataset_yaml}")

    if args.resume:
        # Resume dari checkpoint terakhir
        last_checkpoint = ROOT_DIR / "runs" / "detect" / "train" / "weights" / "last.pt"
        if last_checkpoint.exists():
            logger.info(f"Melanjutkan training dari checkpoint: {last_checkpoint}")
            model = YOLO(str(last_checkpoint))
        else:
            logger.warning("Checkpoint tidak ditemukan. Memulai training dari awal.")
            model = YOLO(model_name)
    else:
        model = YOLO(model_name)

    # Konfigurasi training
    train_kwargs = dict(
        data=dataset_yaml,
        epochs=1 if args.dry_run else args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device if args.device else None,
        workers=args.workers,
        name=exp_name,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        save=True,
        save_period=10,         # Simpan checkpoint setiap 10 epoch
        plots=True,             # Generate plot training metrics
        val=True,               # Validasi setiap epoch
        exist_ok=True,
        resume=args.resume,
        # Augmentasi dari hyp_vtol.yaml
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,             # MATIKAN — gate tidak pernah terbalik
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    if args.dry_run:
        logger.info("MODE DRY-RUN: Hanya 1 epoch untuk validasi config...")
        train_kwargs["batch"] = 1

    # Jalankan training!
    results = model.train(**train_kwargs)

    # 5. Salin model terbaik ke models/best.pt
    logger.info("=== Post-Training: Menyalin Model Terbaik ===")
    
    # Cari file best.pt dari hasil training
    runs_dir = ROOT_DIR / "runs" / "detect" / exp_name / "weights"
    best_pt = runs_dir / "best.pt"
    
    if best_pt.exists():
        dest = MODELS_DIR / "best.pt"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_pt), str(dest))
        logger.info(f"✓ Model terbaik disalin ke: {dest}")
        logger.info(f"  Ukuran: {dest.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        logger.warning(f"File best.pt tidak ditemukan di {runs_dir}")
        # Coba cari di lokasi default
        alt_best = ROOT_DIR / "runs" / "detect" / exp_name / "weights" / "best.pt"
        if alt_best.exists():
            dest = MODELS_DIR / "best.pt"
            shutil.copy2(str(alt_best), str(dest))
            logger.info(f"✓ Model terbaik disalin dari lokasi alternatif ke: {dest}")

    # 6. Print ringkasan
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              TRAINING SELESAI!                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Model tersimpan di : models/best.pt                    ║")
    print(f"║  Logs & plots di    : runs/detect/{exp_name}/")
    print("║                                                          ║")
    print("║  LANGKAH SELANJUTNYA:                                    ║")
    print("║  1. Periksa metrics di runs/detect/.../*.png             ║")
    print("║  2. Test model:                                          ║")
    print("║     python scripts/test_detector.py \\                    ║")
    print("║       --source webcam --model models/best.pt             ║")
    print("║  3. Export untuk Orange Pi 4:                             ║")
    print("║     python scripts/export_model.py                       ║")
    print("║  4. Benchmark:                                           ║")
    print("║     python scripts/benchmark.py                          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    main()
