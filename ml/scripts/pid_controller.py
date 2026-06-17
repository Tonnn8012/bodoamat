"""
scripts/pid_controller.py
==========================
Modul kontroler PID presisi tinggi yang dirancang untuk drone VTOL.

Fitur khusus untuk mengatasi masalah inersia dan gangguan angin:
1. Anti-Windup (Clamping): Menghentikan akumulasi integral saat output jenuh.
2. Low-Pass Filter pada Derivative: Meredam noise dari visual tracking.
3. Slew Rate Limiter: Membatasi akselerasi (kecepatan perubahan output) agar drone tidak tersentak/kebalasan.
4. Deadzone: Mengabaikan error kecil di dekat pusat agar drone stabil (tidak osilasi kecil).
"""

import time
import logging

logger = logging.getLogger("PIDController")

class PIDController:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        min_output: float = -1.0,
        max_output: float = 1.0,
        max_integral: float = 0.5,
        slew_rate_limit: float = None,  # Batas perubahan output per detik (None = dinonaktifkan)
        d_filter_coeff: float = 0.3,    # Alpha filter low-pass untuk D (0.0 - 1.0, makin kecil makin mulus)
        deadzone: float = 0.0,          # Error minimal yang akan direspon
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.min_output = min_output
        self.max_output = max_output
        self.max_integral = max_integral
        self.slew_rate_limit = slew_rate_limit
        self.d_filter_coeff = d_filter_coeff
        self.deadzone = deadzone

        self.reset()

    def reset(self) -> None:
        """Mereset state internal controller."""
        self._prev_error = 0.0
        self._integral = 0.0
        self._prev_d = 0.0
        self._prev_output = 0.0
        self._last_time = None

    def update(self, error: float, current_time: float = None) -> float:
        """
        Hitung output PID berdasarkan error saat ini.
        
        Args:
            error: Nilai error saat ini (misal dalam pixel atau meter).
            current_time: Epoch time dalam detik. Jika None, menggunakan time.time().
            
        Returns:
            Output kontroler yang dibatasi (min_output s/d max_output).
        """
        if current_time is None:
            current_time = time.time()

        if self._last_time is None:
            self._last_time = current_time
            self._prev_error = error
            return 0.0

        dt = current_time - self._last_time
        if dt <= 0.0:
            return self._prev_output

        # 1. Terapkan Deadzone
        if abs(error) < self.deadzone:
            error = 0.0

        # 2. Proportional term
        p_term = self.kp * error

        # 3. Integral term dengan Anti-Windup (Conditional Integration / Clamping)
        # Kita hanya menambah integral jika output sebelumnya belum jenuh (saturated),
        # ATAU arah koreksi integral akan mengurangi kejenuhan.
        output_saturated = (
            self._prev_output >= self.max_output and error > 0
        ) or (
            self._prev_output <= self.min_output and error < 0
        )

        if not output_saturated:
            self._integral += error * dt
            # Batasi nilai integral absolut agar tidak windup terlalu besar
            self._integral = max(min(self._integral, self.max_integral / self.ki if self.ki != 0 else 0), 
                                 -self.max_integral / self.ki if self.ki != 0 else 0)

        i_term = self.ki * self._integral

        # 4. Derivative term dengan Low-Pass Filter
        raw_d = (error - self._prev_error) / dt
        # Terapkan filter eksponensial (EMA) pada derivative
        filtered_d = (self.d_filter_coeff * raw_d) + ((1.0 - self.d_filter_coeff) * self._prev_d)
        d_term = self.kd * filtered_d

        # Simpan state untuk iterasi berikutnya
        self._prev_error = error
        self._prev_d = filtered_d
        self._last_time = current_time

        # 5. Hitung total output mentah
        raw_output = p_term + i_term + d_term

        # 6. Terapkan Slew Rate Limiter (Membatasi akselerasi / hentakan output)
        output = raw_output
        if self.slew_rate_limit is not None:
            max_change = self.slew_rate_limit * dt
            change = output - self._prev_output
            if change > max_change:
                output = self._prev_output + max_change
            elif change < -max_change:
                output = self._prev_output - max_change

        # 7. Batasi output akhir ke rentang fisik drone
        output = max(min(output, self.max_output), self.min_output)
        
        self._prev_output = output
        return output
