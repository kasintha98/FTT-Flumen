import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
from scipy import signal

# -------- CONFIG --------
MQTT_BROKER = "dev.flumina.de"
MQTT_TOPIC  = "root/acc_fifo_batch"

MAX_POINTS = 800   # show last ~1 second @ 800 Hz
SAMPLE_RATE = 800  # Hz

# -------- DATA BUFFERS --------
x_buf = collections.deque(maxlen=MAX_POINTS)
y_buf = collections.deque(maxlen=MAX_POINTS)
z_buf = collections.deque(maxlen=MAX_POINTS)

# -------- MQTT CALLBACKS --------
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT with code", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        # If data is a list of batch objects, each with 's' key
        if isinstance(data, list):
            for batch in data:
                samples = batch.get("s", [])
                for s in samples:
                    x_buf.append(s[0])
                    y_buf.append(s[1])
                    z_buf.append(s[2])
        # If data is a single object with 's' key (old format)
        elif isinstance(data, dict) and "s" in data:
            samples = data.get("s", [])
            for s in samples:
                x_buf.append(s[0])
                y_buf.append(s[1])
                z_buf.append(s[2])
        else:
            print("Unknown data format:", data)
    except Exception as e:
        print("Parse error:", e)

# -------- MQTT CLIENT --------
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

# -------- PLOTTING --------
plt.ion()
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, :])

# Time-domain plot
line_x, = ax1.plot([], [], label="X")
line_y, = ax1.plot([], [], label="Y")
line_z, = ax1.plot([], [], label="Z")
ax1.set_title("ADXL FIFO data (real-time)")
ax1.set_ylabel("mg")
ax1.set_xlabel("samples")
ax1.legend()
ax1.grid(True)

# FFT plot
fft_x, = ax2.plot([], [], label="X", alpha=0.7)
fft_y, = ax2.plot([], [], label="Y", alpha=0.7)
fft_z, = ax2.plot([], [], label="Z", alpha=0.7)
ax2.set_title("FFT Magnitude Spectrum (0-60 Hz)")
ax2.set_ylabel("Magnitude")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_xlim(0, 60)
ax2.legend()
ax2.grid(True)

# Harmonic Analysis plot
harm_x, = ax3.plot([], [], 'o-', label="X harmonics", markersize=8)
harm_y, = ax3.plot([], [], 's-', label="Y harmonics", markersize=8)
harm_z, = ax3.plot([], [], '^-', label="Z harmonics", markersize=8)
ax3.set_title("Top 5 Harmonics")
ax3.set_ylabel("Magnitude")
ax3.set_xlabel("Frequency (Hz)")
ax3.legend()
ax3.grid(True)

# Spectrogram (Z-axis)
spec_img = None
ax4.set_title("Spectrogram (Z-axis)")
ax4.set_ylabel("Frequency (Hz)")
ax4.set_xlabel("Time (samples)")

while True:
    if len(x_buf) > 0:
        # Update time-domain plot
        line_x.set_ydata(x_buf)
        line_y.set_ydata(y_buf)
        line_z.set_ydata(z_buf)

        line_x.set_xdata(range(len(x_buf)))
        line_y.set_xdata(range(len(y_buf)))
        line_z.set_xdata(range(len(z_buf)))

        ax1.relim()
        ax1.autoscale_view()

        # Compute and update FFT
        if len(x_buf) >= 64:  # Minimum points for meaningful FFT
            x_arr = np.array(x_buf)
            y_arr = np.array(y_buf)
            z_arr = np.array(z_buf)

            # Apply FFT
            fft_x_data = np.fft.rfft(x_arr)
            fft_y_data = np.fft.rfft(y_arr)
            fft_z_data = np.fft.rfft(z_arr)

            # Compute magnitude
            mag_x = np.abs(fft_x_data)
            mag_y = np.abs(fft_y_data)
            mag_z = np.abs(fft_z_data)

            # Frequency axis
            freqs = np.fft.rfftfreq(len(x_arr), 1/SAMPLE_RATE)

            # Update FFT plot
            fft_x.set_xdata(freqs)
            fft_x.set_ydata(mag_x)
            fft_y.set_xdata(freqs)
            fft_y.set_ydata(mag_y)
            fft_z.set_xdata(freqs)
            fft_z.set_ydata(mag_z)

            ax2.relim()
            ax2.autoscale_view()

            # Harmonic Analysis - Find top 5 peaks
            def find_top_harmonics(mag, freqs, n_peaks=5):
                # Only look in 0-60 Hz range
                mask = freqs <= 60
                mag_filtered = mag[mask]
                freqs_filtered = freqs[mask]
                
                # Find peaks
                peaks, properties = signal.find_peaks(mag_filtered, height=np.max(mag_filtered)*0.1)
                
                if len(peaks) > 0:
                    # Sort by magnitude and take top n
                    top_indices = np.argsort(properties['peak_heights'])[-n_peaks:][::-1]
                    top_peaks = peaks[top_indices]
                    return freqs_filtered[top_peaks], mag_filtered[top_peaks]
                return np.array([]), np.array([])
            
            harm_freq_x, harm_mag_x = find_top_harmonics(mag_x, freqs)
            harm_freq_y, harm_mag_y = find_top_harmonics(mag_y, freqs)
            harm_freq_z, harm_mag_z = find_top_harmonics(mag_z, freqs)
            
            # Update harmonic plot
            harm_x.set_xdata(harm_freq_x)
            harm_x.set_ydata(harm_mag_x)
            harm_y.set_xdata(harm_freq_y)
            harm_y.set_ydata(harm_mag_y)
            harm_z.set_xdata(harm_freq_z)
            harm_z.set_ydata(harm_mag_z)
            
            ax3.relim()
            ax3.autoscale_view()

            # Spectrogram - compute for Z-axis
            if len(z_buf) >= 256:
                f, t, Sxx = signal.spectrogram(z_arr, SAMPLE_RATE, nperseg=128, noverlap=64)
                
                # Limit to 0-60 Hz
                freq_mask = f <= 60
                f_limited = f[freq_mask]
                Sxx_limited = Sxx[freq_mask, :]
                
                # Update spectrogram
                ax4.clear()
                spec_img = ax4.pcolormesh(t, f_limited, 10 * np.log10(Sxx_limited + 1e-10), 
                                         shading='gouraud', cmap='viridis')
                ax4.set_title("Spectrogram (Z-axis, 0-60 Hz)")
                ax4.set_ylabel("Frequency (Hz)")
                ax4.set_xlabel("Time (s)")
                ax4.set_ylim(0, 60)

    plt.pause(0.02)
