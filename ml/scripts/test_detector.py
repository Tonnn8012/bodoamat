"""
scripts/test_detector.py
=========================
Script untuk testing & validasi VTOLDetector tanpa drone dan tanpa ROS.

Mode yang didukung:
    1. Video    : python test_detector.py --source video --path test_video.mp4
    2. Webcam   : python test_detector.py --source webcam --camera 0
    3. Images   : python test_detector.py --source images --path data/raw/

Tambahan flag:
    --save      : simpan frame hasil anotasi ke folder output/
    --model     : path ke model (.pt / .onnx), default: ../models/best.pt
    --conf      : confidence threshold, default: 0.5
    --onnx      : tandai bahwa model yang dipakai berformat ONNX

File ini MURNI Python, TIDAK ADA import ROS.
Target Python 3.10+
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import cv2

# Tambahkan folder parent ke sys.path agar bisa import vtol_detector.py
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from vtol_detector import VTOLDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Membaca argumen command line untuk menentukan mode testing.

    Returns:
        Namespace hasil parsing argparse, berisi: source, path, camera,
        model, conf, onnx, save.
    """
    parser = argparse.ArgumentParser(
        description="Testing VTOLDetector dari video, webcam, atau folder gambar."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["video", "webcam", "images"],
        help="Sumber input: 'video', 'webcam', atau 'images'.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path ke file video atau folder gambar (wajib untuk --source video/images).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index kamera webcam (default: 0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join("..", "models", "best.pt"),
        help="Path ke model YOLO (.pt atau .onnx). Default: ../models/best.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold minimum (default: 0.5).",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Tandai jika model yang dipakai berformat ONNX.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Simpan frame hasil anotasi ke folder output/.",
    )
    return parser.parse_args()


def siapkan_folder_output() -> str:
    """
    Membuat folder 'output/' jika belum ada (untuk menyimpan hasil anotasi).

    Returns:
        Path ke folder output.
    """
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def cetak_hasil_frame(
    frame_idx: int,
    stable_result: dict | None,
    detector: VTOLDetector,
    frame_width: int,
    frame_height: int,
) -> None:
    """
    Mencetak informasi hasil deteksi ke terminal dengan format yang rapi.

    Format contoh:
        [Frame 042] DETEKSI: gate | x=187.3 y=134.6 | conf=0.89 | error=(27.3, 14.6) | STABLE: YES
        [Frame 043] DETEKSI: - | STABLE: NO (buffer: 2/5)

    Args:
        frame_idx: Nomor urut frame saat ini.
        stable_result: Hasil dari detect_stable() (dict atau None).
        detector: Instance VTOLDetector (untuk akses buffer).
        frame_width: Lebar frame.
        frame_height: Tinggi frame.
    """
    if stable_result is not None:
        error_x, error_y = detector.get_centering_error(
            stable_result, frame_width, frame_height
        )
        print(
            f"[Frame {frame_idx:03d}] DETEKSI: {stable_result['class']} | "
            f"x={stable_result['x']:.1f} y={stable_result['y']:.1f} | "
            f"conf={stable_result['conf']:.2f} | "
            f"error=({error_x:.1f}, {error_y:.1f}) | STABLE: YES"
        )
    else:
        # Hitung berapa frame dalam buffer yang punya deteksi (untuk info debug)
        jumlah_terdeteksi = sum(1 for d in detector.detection_buffer if len(d) > 0)
        total_buffer = len(detector.detection_buffer)
        print(
            f"[Frame {frame_idx:03d}] DETEKSI: - | "
            f"STABLE: NO (buffer: {jumlah_terdeteksi}/{total_buffer})"
        )


def tampilkan_fps(frame: cv2.Mat, fps: float) -> None:
    """
    Menulis nilai FPS di pojok kiri atas frame.

    Args:
        frame: Frame (numpy array BGR) yang akan diberi teks FPS.
        fps: Nilai FPS yang dihitung.
    """
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )


def jalankan_video_atau_webcam(args: argparse.Namespace, detector: VTOLDetector) -> None:
    """
    Menjalankan testing untuk mode 'video' atau 'webcam'.

    Loop:
        - Baca frame
        - Jalankan detect_stable()
        - Anotasi frame + tampilkan FPS
        - Tampilkan ke window OpenCV
        - Cetak info deteksi ke terminal
        - Simpan ke output/ jika --save aktif
        - Tekan 'q' untuk keluar

    Di akhir sesi, cetak rata-rata FPS dan persentase frame dengan deteksi.

    Args:
        args: Hasil parsing argparse.
        detector: Instance VTOLDetector yang sudah siap pakai.
    """
    # Tentukan sumber video: webcam (index kamera) atau file video
    if args.source == "webcam":
        sumber = args.camera
        print(f"Membuka webcam index {args.camera}...")
    else:
        if args.path is None:
            print("Error: --path wajib diisi untuk --source video")
            return
        sumber = args.path
        print(f"Membuka video dari '{args.path}'...")

    cap = cv2.VideoCapture(sumber)

    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka sumber video '{sumber}'.")
        return

    output_dir = siapkan_folder_output() if args.save else None

    frame_idx = 0
    frame_dengan_deteksi = 0
    daftar_fps: list[float] = []

    try:
        while True:
            waktu_mulai = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Frame tidak terbaca / video selesai.")
                break

            frame_idx += 1
            frame_height, frame_width = frame.shape[:2]

            # Jalankan deteksi dengan smoothing
            stable_result = detector.detect_stable(frame)
            if stable_result is not None:
                frame_dengan_deteksi += 1

            # Ambil deteksi mentah (frame terakhir di buffer) untuk anotasi visual
            deteksi_mentah = (
                detector.detection_buffer[-1] if detector.detection_buffer else []
            )
            annotated = detector.annotate_frame(frame, deteksi_mentah)

            # Hitung FPS
            durasi = time.time() - waktu_mulai
            fps = 1.0 / durasi if durasi > 0 else 0.0
            daftar_fps.append(fps)
            tampilkan_fps(annotated, fps)

            # Cetak hasil ke terminal
            cetak_hasil_frame(frame_idx, stable_result, detector, frame_width, frame_height)

            # Tampilkan frame
            cv2.imshow("VTOL Detector - Test", annotated)

            # Simpan frame hasil anotasi jika diminta
            if args.save and output_dir is not None:
                nama_file = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(nama_file, annotated)

            # Tekan 'q' untuk keluar
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Keluar dari testing (ditekan 'q').")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        cetak_ringkasan(frame_idx, frame_dengan_deteksi, daftar_fps)


def jalankan_images(args: argparse.Namespace, detector: VTOLDetector) -> None:
    """
    Menjalankan testing untuk mode 'images' (folder berisi file .jpg/.png).

    Loop:
        - Baca satu gambar
        - Jalankan detect() (tanpa smoothing, karena gambar tidak berurutan)
        - Anotasi gambar
        - Tampilkan ke window OpenCV
        - Cetak info deteksi ke terminal
        - Simpan ke output/ jika --save aktif
        - Tekan 'n' untuk lanjut ke gambar berikutnya, 'q' untuk keluar

    Args:
        args: Hasil parsing argparse.
        detector: Instance VTOLDetector yang sudah siap pakai.
    """
    if args.path is None:
        print("Error: --path wajib diisi untuk --source images")
        return

    # Cari semua file .jpg dan .png di folder
    pola_jpg = os.path.join(args.path, "*.jpg")
    pola_png = os.path.join(args.path, "*.png")
    daftar_file = sorted(glob.glob(pola_jpg) + glob.glob(pola_png))

    if not daftar_file:
        print(f"Tidak ada file .jpg/.png ditemukan di '{args.path}'.")
        return

    print(f"Ditemukan {len(daftar_file)} gambar di '{args.path}'.")

    output_dir = siapkan_folder_output() if args.save else None

    frame_idx = 0
    frame_dengan_deteksi = 0
    daftar_fps: list[float] = []

    for path_gambar in daftar_file:
        waktu_mulai = time.time()

        frame = cv2.imread(path_gambar)
        if frame is None:
            print(f"Gagal membaca gambar: {path_gambar}, dilewati.")
            continue

        frame_idx += 1
        frame_height, frame_width = frame.shape[:2]

        # Untuk mode gambar, gunakan detect() biasa (tanpa smoothing antar-frame)
        hasil_deteksi = detector.detect(frame)
        if hasil_deteksi:
            frame_dengan_deteksi += 1

        annotated = detector.annotate_frame(frame, hasil_deteksi)

        durasi = time.time() - waktu_mulai
        fps = 1.0 / durasi if durasi > 0 else 0.0
        daftar_fps.append(fps)
        tampilkan_fps(annotated, fps)

        # Cetak info deteksi (tanpa konsep 'stable' karena gambar tunggal)
        if hasil_deteksi:
            d = hasil_deteksi[0]
            error_x, error_y = detector.get_centering_error(d, frame_width, frame_height)
            print(
                f"[{os.path.basename(path_gambar)}] DETEKSI: {d['class']} | "
                f"x={d['x']:.1f} y={d['y']:.1f} | conf={d['conf']:.2f} | "
                f"error=({error_x:.1f}, {error_y:.1f})"
            )
        else:
            print(f"[{os.path.basename(path_gambar)}] DETEKSI: - (tidak ada objek terdeteksi)")

        cv2.imshow("VTOL Detector - Test (Images)", annotated)

        if args.save and output_dir is not None:
            nama_output = os.path.join(output_dir, os.path.basename(path_gambar))
            cv2.imwrite(nama_output, annotated)

        # Tunggu input: 'n' lanjut, 'q' keluar
        print("Tekan 'n' untuk gambar berikutnya, 'q' untuk keluar...")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("n"):
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                cetak_ringkasan(frame_idx, frame_dengan_deteksi, daftar_fps)
                return

    cv2.destroyAllWindows()
    cetak_ringkasan(frame_idx, frame_dengan_deteksi, daftar_fps)


def cetak_ringkasan(total_frame: int, frame_dengan_deteksi: int, daftar_fps: list[float]) -> None:
    """
    Mencetak ringkasan akhir sesi testing: rata-rata FPS dan detection rate.

    Args:
        total_frame: Jumlah total frame/gambar yang diproses.
        frame_dengan_deteksi: Jumlah frame yang memiliki deteksi (stabil atau tidak,
                               tergantung mode).
        daftar_fps: List nilai FPS untuk setiap frame.
    """
    print("\n========== RINGKASAN SESI TESTING ==========")
    print(f"Total frame diproses : {total_frame}")

    if daftar_fps:
        rata_fps = sum(daftar_fps) / len(daftar_fps)
        print(f"Rata-rata FPS        : {rata_fps:.2f}")
    else:
        print("Rata-rata FPS        : - (tidak ada frame)")

    if total_frame > 0:
        detection_rate = (frame_dengan_deteksi / total_frame) * 100
        print(f"Detection rate       : {detection_rate:.1f}% ({frame_dengan_deteksi}/{total_frame})")
    else:
        print("Detection rate       : - (tidak ada frame)")

    print("=============================================\n")


def main() -> None:
    """
    Fungsi utama: parsing argumen, inisialisasi detector, dan menjalankan
    mode testing yang sesuai (video, webcam, atau images).
    """
    args = parse_args()

    print("Menginisialisasi VTOLDetector...")
    detector = VTOLDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        use_onnx=args.onnx,
    )

    if detector.is_dummy:
        print(
            "PERHATIAN: Model belum tersedia, berjalan dengan MODEL DUMMY "
            "(tidak akan ada deteksi). Pipeline kamera/centering tetap bisa diuji."
        )

    if args.source in ("video", "webcam"):
        jalankan_video_atau_webcam(args, detector)
    elif args.source == "images":
        jalankan_images(args, detector)


if __name__ == "__main__":
    main()