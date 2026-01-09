import streamlit as st
import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
from scipy import signal
import time

# -------- CONFIG --------
MQTT_BROKER = "dev.flumina.de"
MQTT_TOPIC = "root/acc_fifo_batch"

MAX_POINTS = 800  # show last ~1 second @ 800 Hz
SAMPLE_RATE = 800  # Hz

# Page Config
st.set_page_config(
    page_title="Real-Time Vibration Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🔴 Live Vibration Dashboard")

# Initialize buffers in Session State so they persist between reruns
if "x_buf" not in st.session_state:
    st.session_state.x_buf = collections.deque(maxlen=MAX_POINTS)
    st.session_state.y_buf = collections.deque(maxlen=MAX_POINTS)
    st.session_state.z_buf = collections.deque(maxlen=MAX_POINTS)

# Create a placeholder for the plots
# This allows us to replace the image continuously
placeholder = st.empty()
status_text = st.sidebar.empty()


# -------- MQTT CALLBACKS --------
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        status_text.success("Connected to MQTT Broker")
        client.subscribe(MQTT_TOPIC)
    else:
        status_text.error(f"Failed to connect, code {rc}")


def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())

        new_x, new_y, new_z = [], [], []

        # Handle List format
        if isinstance(data, list):
            for batch in data:
                samples = batch.get("s", [])
                for s in samples:
                    new_x.append(s[0])
                    new_y.append(s[1])
                    new_z.append(s[2])
        # Handle Dict format
        elif isinstance(data, dict) and "s" in data:
            samples = data.get("s", [])
            for s in samples:
                new_x.append(s[0])
                new_y.append(s[1])
                new_z.append(s[2])

        # Update session state buffers
        st.session_state.x_buf.extend(new_x)
        st.session_state.y_buf.extend(new_y)
        st.session_state.z_buf.extend(new_z)

    except Exception as e:
        print("Parse error:", e)


# -------- SETUP MQTT --------
# Use a unique client_id to prevent conflicts on the public broker
client_id = f"streamlit_client_{int(time.time())}"
client = mqtt.Client(client_id=client_id)
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(MQTT_BROKER, 1883, 60)
    client.loop_start()
except Exception as e:
    st.error(f"Could not connect to MQTT Broker: {e}")

# -------- MAIN LOOP --------
# Checkbox to stop the loop manually if needed
run_app = st.checkbox('Start Live Feed', value=True)

while run_app:
    # Access buffers from session state
    x_buf = st.session_state.x_buf
    y_buf = st.session_state.y_buf
    z_buf = st.session_state.z_buf

    if len(x_buf) > 0:
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, :])

        # 1. Time-domain plot
        ax1.plot(list(x_buf), label="X", lw=1)
        ax1.plot(list(y_buf), label="Y", lw=1)
        ax1.plot(list(z_buf), label="Z", lw=1)
        ax1.set_title("ADXL FIFO data (real-time)")
        ax1.set_ylabel("mg")
        ax1.legend(loc='upper right')
        ax1.grid(True)

        # Compute FFT logic
        if len(x_buf) >= 64:
            x_arr = np.array(x_buf)
            y_arr = np.array(y_buf)
            z_arr = np.array(z_buf)

            fft_x_data = np.fft.rfft(x_arr)
            fft_y_data = np.fft.rfft(y_arr)
            fft_z_data = np.fft.rfft(z_arr)

            mag_x = np.abs(fft_x_data)
            mag_y = np.abs(fft_y_data)
            mag_z = np.abs(fft_z_data)

            freqs = np.fft.rfftfreq(len(x_arr), 1 / SAMPLE_RATE)

            # 2. FFT plot
            ax2.plot(freqs, mag_x, label="X", alpha=0.7)
            ax2.plot(freqs, mag_y, label="Y", alpha=0.7)
            ax2.plot(freqs, mag_z, label="Z", alpha=0.7)
            ax2.set_title("FFT Magnitude Spectrum (0-60 Hz)")
            ax2.set_xlim(0, 60)
            ax2.grid(True)


            # Harmonic Analysis Helper
            def find_top_harmonics(mag, freqs, n_peaks=5):
                mask = freqs <= 60
                mag_filtered = mag[mask]
                freqs_filtered = freqs[mask]
                peaks, properties = signal.find_peaks(mag_filtered, height=np.max(mag_filtered) * 0.1)
                if len(peaks) > 0:
                    top_indices = np.argsort(properties['peak_heights'])[-n_peaks:][::-1]
                    top_peaks = peaks[top_indices]
                    return freqs_filtered[top_peaks], mag_filtered[top_peaks]
                return np.array([]), np.array([])


            hx, hy = find_top_harmonics(mag_x, freqs)

            # 3. Harmonics Plot (Showing X only for clarity in demo)
            ax3.plot(hx, hy, 'o-', label="X Peaks", markersize=8)
            ax3.set_title("Top 5 Harmonics (X-axis)")
            ax3.set_xlabel("Frequency (Hz)")
            ax3.legend()
            ax3.grid(True)

            # 4. Spectrogram (Z-axis)
            if len(z_buf) >= 256:
                f, t, Sxx = signal.spectrogram(z_arr, SAMPLE_RATE, nperseg=128, noverlap=64)
                freq_mask = f <= 60

                pcm = ax4.pcolormesh(t, f[freq_mask], 10 * np.log10(Sxx[freq_mask, :] + 1e-10),
                                     shading='gouraud', cmap='viridis')
                ax4.set_title("Spectrogram (Z-axis, 0-60 Hz)")
                ax4.set_ylabel("Frequency (Hz)")
                ax4.set_ylim(0, 60)
                fig.colorbar(pcm, ax=ax4, label="dB")

        # Push the plot to the placeholder
        placeholder.pyplot(fig)

        # Clear figure to prevent memory leak
        plt.close(fig)

    # Sleep briefly to reduce CPU usage on the cloud server
    time.sleep(0.2)