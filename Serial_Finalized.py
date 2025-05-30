import serial
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ---- UART Setup ----
ser = serial.Serial(port='COM4', baudrate=115200, timeout=1)
pattern = re.compile(r"Value\[1\]\s*=\s*(\d+)\s*Value\[2\]\s*=\s*(\d+)")
values1, values2 = [], []

print("Reading from UART. Press Ctrl+C to stop and analyze.")

try:
    while True:
        line = ser.readline().decode('ascii', errors='ignore').strip()
        match = pattern.match(line)
        if match:
            v1 = int(match.group(1))
            v2 = int(match.group(2))
            values1.append(v1)
            values2.append(v2)
            print(f"Read Value[1]={v1}, Value[2]={v2}")
except KeyboardInterrupt:
    print("Stopped reading. Starting analysis.")
finally:
    ser.close()

# ---- Save to CSV ----
df = pd.DataFrame({
    'Sample Number': np.arange(len(values1)),
    'Value[1]': values1,
    'Value[2]': values2
})
df.to_csv("uart_data.csv", index=False)

# ---- Parameters ----
sampling_rate = 100  # Hz

# ---- Filter Function ----
def moving_average(signal, window_size=7):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

# ---- Preprocessing ----
v1 = np.array(values1)
v2 = np.array(values2)
v1_dsp = moving_average(v1)
v2_dsp = moving_average(v2)
v1_norm = (v1_dsp - np.min(v1_dsp)) / (np.max(v1_dsp) - np.min(v1_dsp))
v2_norm = (v2_dsp - np.min(v2_dsp)) / (np.max(v2_dsp) - np.min(v2_dsp))
fused_signal = 0.5 * v1_norm + 0.5 * v2_norm
sample_indices = np.arange(len(fused_signal))

# ---- Peak Detection (Corrected) ----
refined_peaks, _ = find_peaks(fused_signal, distance=40, prominence=0.05, width=5)

# ---- Plot 1: Detected Peaks ----
plt.figure(figsize=(16, 6))
plt.plot(sample_indices, fused_signal, label='Fused Signal', color='green')
plt.plot(sample_indices[refined_peaks], fused_signal[refined_peaks], "ro", label='Detected Inhale Peaks')
plt.title(f"Corrected Breath Cycle Detection - {len(refined_peaks)} Cycles Detected")
plt.xlabel("Sample Number")
plt.ylabel("Normalized Signal Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- JND Calculation ----
v1_peaks, _ = find_peaks(v1_norm, distance=25)
v2_peaks, _ = find_peaks(v2_norm, distance=25)
min_len = min(len(v1_peaks), len(v2_peaks))
jnd_samples = np.abs(v1_peaks[:min_len] - v2_peaks[:min_len])
jnd_seconds = jnd_samples / sampling_rate

# ---- Plot 2: JND ----
plt.figure(figsize=(10, 5))
plt.bar(np.arange(len(jnd_seconds)), jnd_seconds, color='purple')
plt.axhline(np.mean(jnd_seconds), color='black', linestyle='--', label=f'Mean JND: {np.mean(jnd_seconds):.3f}s')
plt.title("Just Noticeable Difference (JND) Between Sensor Peaks")
plt.xlabel("Peak Pair Index")
plt.ylabel("Time Difference (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Psychometric Curve ----
scales = np.linspace(0.2, 1.0, 5)
detection_rates = []
for scale in scales:
    scaled = fused_signal * scale
    pks, _ = find_peaks(scaled, distance=40, prominence=0.05 * scale, width=5)
    if len(pks) >= 2:
        if pks[0] < 20: pks = pks[1:]
        if pks[-1] > len(scaled) - 20: pks = pks[:-1]
    accuracy = min(len(pks), 6) / 6
    detection_rates.append(accuracy)

def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

p0 = [1, 0.5, 10, 0]
popt, _ = curve_fit(sigmoid, scales, detection_rates, p0)
x_fit = np.linspace(0.1, 1.1, 100)
y_fit = sigmoid(x_fit, *popt)

# ---- Plot 3: Psychometric Curve ----
plt.figure(figsize=(10, 6))
plt.plot(scales, detection_rates, 'o', label='Observed Accuracy')
plt.plot(x_fit, y_fit, '-', label='Sigmoid Fit')
plt.xlabel('Scaled Signal Amplitude')
plt.ylabel('Detection Accuracy')
plt.title('Psychometric Curve Based on Fused Signal')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---- Plot 4: Ground Truth Comparison ----
duration_sec = len(fused_signal) / sampling_rate
time_vector = np.linspace(0, duration_sec, len(fused_signal))
ground_truth = 0.5 * (1 + np.sin(2 * np.pi * (6 / duration_sec) * time_vector))

plt.figure(figsize=(16, 6))
plt.plot(time_vector, fused_signal, label='Fused Signal', color='green')
plt.plot(time_vector, ground_truth, '--', label='Ground Truth (6 cycles)', color='black')
plt.plot(time_vector[refined_peaks], fused_signal[refined_peaks], "ro", label='Detected Inhale Peaks')
plt.title("Fused Signal vs Ground Truth (6 Cycles)")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Signal Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
