"""
scripts/convert_to_rknn.py
==========================
Script utilitas untuk mengonversi model YOLO format ONNX ke RKNN (Rockchip NPU).
Script ini harus dijalankan di PC/Laptop x86_64 Linux yang sudah terinstall `rknn-toolkit2`.

Cara Menjalankan:
    python3 convert_to_rknn.py --input ../models/best.onnx --output ../models/best.rknn
"""

import argparse
import os
import sys

try:
    from rknn.api import RKNN
except ImportError:
    print("Error: Library 'rknn-toolkit2' tidak terinstall!")
    print("Harap jalankan script ini di PC/Laptop development (x86_64 Linux) dengan SDK RKNN terpasang.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Konversi ONNX ke RKNN untuk Orange Pi 5 (RK3588)")
    parser.add_argument("--input", type=str, required=True, help="Path ke model input .onnx")
    parser.add_argument("--output", type=str, required=True, help="Path tujuan model output .rknn")
    parser.add_argument("--quantize", action="store_true", help="Aktifkan kuantisasi INT8 (opsional, butuh dataset kalibrasi)")
    parser.add_argument("--dataset", type=str, default=None, help="Path file txt dataset kalibrasi untuk kuantisasi")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Model input tidak ditemukan di '{args.input}'")
        return

    # Inisialisasi RKNN object
    rknn = RKNN(verbose=True)

    print("--> Mengonfigurasi parameter target RK3588...")
    # Target platform: rk3588 (Orange Pi 5)
    # Mean dan Std values disesuaikan dengan preprocessing normalisasi [0.0 - 1.0] milik YOLO
    # YOLO normalisasi menggunakan /255.0. Jadi mean = 0, std = 255.
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )

    print(f"--> Memuat model ONNX: {args.input}")
    ret = rknn.load_onnx(model=args.input)
    if ret != 0:
        print("Gagal memuat model ONNX!")
        return

    print("--> Membangun model (compiling to RKNN)...")
    ret = rknn.build(do_quantization=args.quantize, dataset=args.dataset)
    if ret != 0:
        print("Gagal membangun model RKNN!")
        return

    print(f"--> Mengekspor model RKNN ke: {args.output}")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print("Gagal mengekspor model RKNN!")
        return

    print("====================================================")
    print(" KONVERSI SELESAI BERHASIL!")
    print(f" Model RKNN disimpan di: {args.output}")
    print(" Pindahkan file ini ke Orange Pi 5 untuk inferensi NPU.")
    print("====================================================")

    rknn.release()


if __name__ == "__main__":
    main()
