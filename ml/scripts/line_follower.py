"""
scripts/line_follower.py
=========================
Modul OpenCV untuk mendeteksi dan mengikuti garis hitam di lantai (Misi 4).

Fitur Peningkatan untuk Kompetisi:
1. Adaptive Thresholding (Otsu's Binarization): Mengatasi masalah perubahan 
   intensitas cahaya (indoor/outdoor) secara otomatis.
2. Dashed-Line Bridging: Operasi morfologi dengan vertical kernel untuk menyambung 
   garis putus-putus secara visual di mask.
3. Centroid Hysteresis (Memory State): Menyimpan posisi centroid terakhir saat 
   garis menghilang sementara di area putus-putus agar drone tidak goyang/lost-control.
"""

import cv2
import numpy as np
import math
import logging

logger = logging.getLogger("LineFollower")


class LineFollower:
    def __init__(
        self,
        default_threshold: int = 60,     # Fallback threshold jika Otsu gagal
        min_area: float = 300.0,         # Area minimal contour agar noise terabaikan
        max_lost_frames: int = 15,       # Toleransi frame lost pada garis putus-putus (~0.5 detik di 30FPS)
    ) -> None:
        self.default_threshold = default_threshold
        self.min_area = min_area
        self.max_lost_frames = max_lost_frames
        
        # State memory untuk smoothing/bridging saat garis hilang sesaat
        self.last_cx_near = None
        self.last_cy_near = None
        self.last_cx_far = None
        self.last_cy_far = None
        
        # Counter frame lost
        self.lost_counter_near = 0
        self.lost_counter_far = 0
        
        # Simpan nilai output terakhir
        self.last_cte = 0.0
        self.last_heading_error = 0.0

    def process_frame(
        self, frame: np.ndarray
    ) -> tuple[float | None, float | None, np.ndarray, np.ndarray]:
        """
        Memproses frame kamera bawah untuk mencari garis hitam putus-putus.
        
        Args:
            frame: Image array BGR.
            
        Returns:
            Tuple (cte, heading_error, mask, annotated)
        """
        h, w = frame.shape[:2]
        cx_frame = w // 2
        
        # 1. Preprocessing: Grayscale + Blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Adaptive Binarization menggunakan Otsu's Thresholding
        # Menghitung ambang batas secara otomatis berdasarkan bimodal histogram piksel
        try:
            # Otsu otomatis mencari threshold optimal (parameter threshold_value diabaikan jika pakai THRESH_OTSU)
            otsu_val, mask = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            logger.debug(f"Otsu Adaptive Threshold Value: {otsu_val:.1f}")
            
            # Pengaman: Jika Otsu mendeteksi threshold terlalu tinggi (misal > 180 karena frame terlalu gelap),
            # lakukan fallback ke threshold statis agar lantai terang tidak terdeteksi sebagai garis hitam.
            if otsu_val > 180 or otsu_val < 30:
                _, mask = cv2.threshold(
                    blurred, self.default_threshold, 255, cv2.THRESH_BINARY_INV
                )
        except Exception as e:
            logger.error(f"Gagal melakukan Otsu thresholding: {e}. Menggunakan fallback manual.")
            _, mask = cv2.threshold(
                blurred, self.default_threshold, 255, cv2.THRESH_BINARY_INV
            )
        
        # 3. Dashed-Line Bridging: Menggunakan Kernel Vertikal untuk menutup celah vertikal
        # Karena garis putus-putus umumnya sejajar arah laju drone (vertikal di frame),
        # kernel vertikal (lebar kecil, tinggi besar) sangat efektif menyambung celah.
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 25))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Lakukan juga closing horizontal kecil untuk merapikan tepi garis
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_horizontal)
        
        # 4. Pembagian ROI (Near: 60-100% tinggi frame, Far: 20-60% tinggi frame)
        near_ymin, near_ymax = int(h * 0.6), h
        far_ymin, far_ymax = int(h * 0.2), int(h * 0.6)
        
        # Cari centroid aktual di masing-masing ROI
        cx_near_act, cy_near_act = self._find_centroid_in_y_range(mask, near_ymin, near_ymax)
        cx_far_act, cy_far_act = self._find_centroid_in_y_range(mask, far_ymin, far_ymax)
        
        # 5. Logika Hysteresis / Memory untuk Centroid Near
        if cx_near_act is not None:
            self.last_cx_near = cx_near_act
            self.last_cy_near = cy_near_act
            self.lost_counter_near = 0
        else:
            self.lost_counter_near += 1
            if self.lost_counter_near > self.max_lost_frames:
                self.last_cx_near = None
                self.last_cy_near = None

        # Logika Hysteresis / Memory untuk Centroid Far
        if cx_far_act is not None:
            self.last_cx_far = cx_far_act
            self.last_cy_far = cy_far_act
            self.lost_counter_far = 0
        else:
            self.lost_counter_far += 1
            if self.lost_counter_far > self.max_lost_frames:
                self.last_cx_far = None
                self.last_cy_far = None

        # 6. Kalkulasi Output CTE & Heading Error
        annotated = frame.copy()
        
        # Gambar garis referensi tengah kamera (biru) dan ROI pembatas (cyan)
        cv2.line(annotated, (cx_frame, 0), (cx_frame, h), (255, 0, 0), 1)
        cv2.line(annotated, (0, near_ymin), (w, near_ymin), (255, 255, 0), 1)
        cv2.line(annotated, (0, far_ymin), (w, far_ymin), (255, 255, 0), 1)
        
        cte = None
        heading_error = None
        
        # Gunakan state memory (termasuk interpolasi lost frame)
        cx_near = self.last_cx_near
        cy_near = self.last_cy_near
        cx_far = self.last_cx_far
        cy_far = self.last_cy_far
        
        if cx_near is not None and cy_near is not None:
            # Hitung CTE (-1.0 s/d 1.0)
            cte = float(cx_near - cx_frame) / (w / 2.0)
            self.last_cte = cte
            
            # Visualisasikan Centroid Near (Hijau jika aktual, Kuning jika menggunakan memori/lost)
            color_near = (0, 255, 0) if self.lost_counter_near == 0 else (0, 255, 255)
            cv2.circle(annotated, (cx_near, cy_near), 8, color_near, -1)
            
            if cx_far is not None and cy_far is not None:
                # Visualisasikan Centroid Far (Oranye jika aktual, Kuning jika memori)
                color_far = (0, 165, 255) if self.lost_counter_far == 0 else (0, 255, 255)
                cv2.circle(annotated, (cx_far, cy_far), 8, color_far, -1)
                
                # Gambar arah lintasan garis (Merah)
                cv2.line(annotated, (cx_near, cy_near), (cx_far, cy_far), (0, 0, 255), 2)
                
                # Hitung heading error (kemiringan sudut garis yaw)
                dx = cx_far - cx_near
                dy = cy_far - cy_near
                heading_error = math.degrees(math.atan2(dx, -dy))
                self.last_heading_error = heading_error
                
                # Tampilkan info HUD
                text = f"CTE: {cte:.2f} | Yaw: {heading_error:.1f}deg"
                # Jika sedang bridging
                if self.lost_counter_near > 0 or self.lost_counter_far > 0:
                    text += " [BRIDGING]"
                cv2.putText(
                    annotated, text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
            else:
                heading_error = 0.0
                self.last_heading_error = 0.0
                cv2.putText(
                    annotated,
                    f"CTE: {cte:.2f} | Yaw: N/A [Far Lost]",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    2,
                )
        else:
            # Benar-benar kehilangan garis (melebihi batas toleransi frame lost)
            cv2.putText(
                annotated,
                "LINE LOST (CRITICAL)",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            
        return cte, heading_error, mask, annotated

    def _find_centroid_in_y_range(
        self, mask: np.ndarray, ymin: int, ymax: int
    ) -> tuple[int | None, int | None]:
        """Mencari centroid dari contour terbesar dalam potongan range Y."""
        # Potong mask hanya pada range Y yang diinginkan
        roi_mask = np.zeros_like(mask)
        roi_mask[ymin:ymax, :] = mask[ymin:ymax, :]
        
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None
            
        # Dapatkan contour dengan area terbesar
        largest_cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_cnt)
        
        if area < self.min_area:
            return None, None
            
        # Hitung moment
        M = cv2.moments(largest_cnt)
        if M["m00"] == 0:
            return None, None
            
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return cx, cy
