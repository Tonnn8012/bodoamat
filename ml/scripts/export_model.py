"""
scripts/export_model.py
========================
Script untuk mengekspor model PyTorch (.pt) hasil training ke format ONNX.
ONNX format sangat direkomendasikan untuk inferensi di CPU Orange Pi 4 (RK3399)
karena jauh lebih ringan daripada PyTorch standar.

Konfigurasi ekspor dioptimalkan untuk performa:
- Input Resolution: 320x320 (meningkatkan FPS secara signifikan dibandingkan 640x640)
- Static Input: (dynamic=False) untuk optimasi alokasi memori ONNX Runtime di CPU ARM.

Cara Pakai:
-----------
# Ekspor model best.pt ke best.onnx dengan input size 320x320
python scripts/export_model.py --weights models/best.pt --imgsz 320
"""

import os
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModelExporter")


def export_model(weights_path: str, imgsz: int) -> str | None:
    """Mengekspor model PyTorch (.pt) ke ONNX."""
    if not os.path.exists(weights_path):
        logger.error(f"File model tidak ditemukan di: {weights_path}")
        return None
        
    try:
        # pyrefly: ignore [missing-import]
        from ultralytics import YOLO
    except ImportError:
        logger.error("Library 'ultralytics' tidak ditemukan. Silakan jalankan: pip install ultralytics")
        return None
        
    logger.info(f"Memuat model dari '{weights_path}'...")
    model = YOLO(weights_path)
    
    logger.info(f"Mengekspor ke format ONNX dengan resolusi input {imgsz}x{imgsz}...")
    # Ekspor model ke ONNX
    # dynamic=False agar size input dikunci (static) untuk mempercepat alokasi memori di CPU Orange Pi
    # opset=12 kompatibel dengan sebagian besar instalasi onnxruntime
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        opset=12,
        simplify=True  # Sederhanakan grafis jaringan jika onnx-simplifier terpasang
    )
    
    logger.info(f"Model berhasil diekspor! Lokasi file: {onnx_path}")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ekspor Model YOLOv8 ke ONNX untuk Orange Pi 4")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path ke file bobot PyTorch (.pt)")
    parser.add_argument("--imgsz", type=int, default=320, help="Ukuran gambar input (default: 320)")
    
    args = parser.parse_args()
    
    logger.info("=== START MODEL EXPORT ===")
    export_model(args.weights, args.imgsz)
    logger.info("=== EXPORT SELESAI ===")


if __name__ == "__main__":
    main()
