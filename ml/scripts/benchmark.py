"""
scripts/benchmark.py
====================
Script untuk mengukur performa (FPS dan Latency) model YOLOv8 pada CPU Orange Pi 4.
Mendukung benchmarking model PyTorch (.pt) dan ONNX (.onnx).

Cara Pakai:
-----------
# Jalankan benchmark pada model ONNX
python scripts/benchmark.py --model models/best.onnx --runs 100

# Jalankan benchmark pada model PyTorch
python scripts/benchmark.py --model models/best.pt --runs 100
"""

import os
import sys
import time
import argparse
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Benchmark")

# Tambahkan path root agar bisa import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from vtol_detector import VTOLDetector


def run_benchmark(model_path: str, runs: int, use_onnx: bool) -> None:
    """Menjalankan loop inferensi pada dummy frame untuk mengukur FPS dan Latensi."""
    if not os.path.exists(model_path):
        logger.error(f"Model tidak ditemukan di: {model_path}")
        return

    # Buat dummy frame ukuran 640x480 (sama dengan input kamera drone)
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    logger.info(f"Menginisialisasi VTOLDetector untuk benchmark...")
    logger.info(f"Model: {model_path} | ONNX Mode: {use_onnx}")
    
    try:
        detector = VTOLDetector(
            model_path=model_path,
            conf_threshold=0.25,  # Gunakan threshold standar
            use_onnx=use_onnx
        )
    except Exception as e:
        logger.error(f"Gagal menginisialisasi model: {e}")
        return

    if detector.is_dummy:
        logger.error("Detector berjalan menggunakan model dummy. Benchmark dibatalkan.")
        return

    logger.info("Melakukan pemanasan model (warmup) 10 iterasi...")
    for _ in range(10):
        _ = detector.detect(dummy_frame)

    logger.info(f"Memulai loop benchmark sebanyak {runs} iterasi...")
    
    latencies = []
    
    for i in range(runs):
        start_time = time.time()
        _ = detector.detect(dummy_frame)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000.0
        latencies.append(latency_ms)
        
        if (i + 1) % (runs // 5) == 0 or (i + 1) == runs:
            logger.info(f"Progres: {i + 1}/{runs} iterasi selesai.")

    # Hitung statistik
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    min_latency = min(latencies)
    fps = 1000.0 / avg_latency
    
    print("\n" + "="*45)
    print("           HASIL BENCHMARK INFERENSI")
    print("="*45)
    print(f"Model Path       : {model_path}")
    print(f"Jumlah Iterasi   : {runs}")
    print(f"Resolusi Input   : 640x480 (kamera BGR)")
    print(f"Format Model     : {'ONNX Runtime CPU' if use_onnx else 'PyTorch (standard)'}")
    print("-"*45)
    print(f"Rata-rata Latency: {avg_latency:.2f} ms")
    print(f"Minimum Latency  : {min_latency:.2f} ms")
    print(f"Maximum Latency  : {max_latency:.2f} ms")
    print(f"Kecepatan (FPS)  : {fps:.2f} FPS")
    print("="*45)
    
    if use_onnx:
        print("\n[TIPS] Orange Pi 4 (RK3399) CPU-only biasanya mencapai 4-8 FPS pada model 320x320.")
        print("Pastikan model Anda diekspor dengan imgsz=320 untuk kecepatan maksimal!")
    else:
        print("\n[TIPS] Jalankan dengan flag --onnx jika Anda menguji file .onnx hasil ekspor.")
    print("="*45 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Kecepatan Inferensi YOLOv8")
    parser.add_argument("--model", type=str, required=True, help="Path ke file model (.pt atau .onnx)")
    parser.add_argument("--runs", type=int, default=100, help="Jumlah iterasi pengujian (default: 100)")
    parser.add_argument("--onnx", action="store_true", help="Set flag ini jika model bertipe ONNX")
    
    args = parser.parse_args()
    
    # Deteksi otomatis jika ekstensi .onnx
    is_onnx = args.onnx or args.model.lower().endswith(".onnx")
    
    run_benchmark(args.model, args.runs, is_onnx)


if __name__ == "__main__":
    main()
