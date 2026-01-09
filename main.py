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
MAX_POINTS = 800
SAMPLE_RATE = 800

# -------- GLOBAL BUFFERS --------
# We use standard global variables for data buffering because
# they are thread-safe enough for this use case and avoid
# Streamlit context errors in background threads.
data_x = collections.deque(maxlen=MAX_POINTS)
data_y = collections.deque(maxlen=MAX_POINTS)
data_z = collections.deque(maxlen=MAX_POINTS)
connection_state = {"status": "Connecting...", "code": None}

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="Real-Time Vibration Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🔴 Live Vibration Dashboard")
status_placeholder = st.empty()  # Placeholder for connection status
plot_placeholder = st.empty()  # Placeholder for the charts


# -------- MQTT CALLBACKS (NO STREAMLIT CALLS HERE) --------
def on_connect(client, userdata, flags, reason_code, properties):
    # Update global variable only, do not call st. functions here
    if reason_code == 0:
        connection_state["status"] = "Connected"
        client.subscribe(MQTT_TOPIC)
    else:
        connection_state["status"] = f"Failed to connect: {reason_code}"


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

        # Extend global buffers
        data_x.extend(new_x)
        data_y.extend(new_y)
        data_z.extend(new_z)

    except Exception as e:
        print(f"Parse error: {e}")


# -------- SETUP MQTT --------
# Helper to cache the client setup so we don't reconnect on every rerun
@st.cache_resource
def start_mqtt_client():
    # generating unique ID
    client_id = f"streamlit_client_{int(time.time())}"
    # Updated to CallbackAPIVersion.VERSION2 to fix deprecation warning
    client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_start()  # Run in background thread
        return client
    except Exception as e:
        return None


# Start Client
client = start_mqtt_client()

# -------- MAIN UI LOOP --------
run_app = st.checkbox('Start Live Feed', value=True)

while run_app:
    # 1. Update Status
    if connection_state["status"] == "Connected":
        status_placeholder.success(f"MQTT Status: {connection_state['status']}")
    else:
        status_placeholder.warning(f"MQTT Status: {connection_state['status']}")

    # 2. Check if we have data
    if len(data_x) > 0:
        # Create local copies of data to avoid threading race conditions during plotting
        x_buf = list(data_x)
        y_buf = list(data_y)
        z_buf = list(data_z)

        # Create figure
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, :])

        # --- Plot 1: Time Domain ---
        ax1.plot(x_buf, label="X", lw=1)
        ax1.plot(y_buf, label="Y", lw=1)
        ax1.plot(z_buf, label="Z", lw=1)
        ax1.set_title("ADXL FIFO data (real-time)")
        ax1.set_ylabel("mg")
        ax1.legend(loc='upper right')
        ax1.grid(True)

        # --- Compute FFT ---
        if len(x_buf) >= 64:
            x_arr = np.array(x_buf)
            y_arr = np.array(y_buf)
            z_arr = np.array(z_buf)

            # FFT Calculation
            fft_x_data = np.fft.rfft(x_arr)
            fft_y_data = np.fft.rfft(y_arr)
            fft_z_data = np.fft.rfft(z_arr)
            mag_x = np.abs(fft_x_data)
            mag_y = np.abs(fft_y_data)
            mag_z = np.abs(fft_z_data)
            freqs = np.fft.rfftfreq(len(x_arr), 1 / SAMPLE_RATE)

            # --- Plot 2: FFT ---
            ax2.plot(freqs, mag_x, label="X", alpha=0.7)
            ax2.plot(freqs, mag_y, label="Y", alpha=0.7)
            ax2.plot(freqs, mag_z, label="Z", alpha=0.7)
            ax2.set_title("FFT Magnitude Spectrum (0-60 Hz)")
            ax2.set_xlim(0, 60)
            ax2.grid(True)


            # --- Plot 3: Harmonics ---
            def find_top_harmonics(mag, freqs, n_peaks=5):
                mask = freqs <= 60
                mag_filtered = mag[mask]
                freqs_filtered = freqs[mask]
                if len(mag_filtered) == 0: return np.array([]), np.array([])

                peaks, properties = signal.find_peaks(mag_filtered, height=np.max(mag_filtered) * 0.1)
                if len(peaks) > 0:
                    top_indices = np.argsort(properties['peak_heights'])[-n_peaks:][::-1]
                    top_peaks = peaks[top_indices]
                    return freqs_filtered[top_peaks], mag_filtered[top_peaks]
                return np.array([]), np.array([])


            hx, hy = find_top_harmonics(mag_x, freqs)
            ax3.plot(hx, hy, 'o-', label="X Peaks", markersize=8)
            ax3.set_title("Top 5 Harmonics (X-axis)")
            ax3.set_xlabel("Frequency (Hz)")
            ax3.legend()
            ax3.grid(True)

            # --- Plot 4: Spectrogram ---
            if len(z_buf) >= 256:
                f, t, Sxx = signal.spectrogram(z_arr, SAMPLE_RATE, nperseg=128, noverlap=64)
                freq_mask = f <= 60

                # Sxx might be empty or small, handle safely
                if Sxx.size > 0:
                    pcm = ax4.pcolormesh(t, f[freq_mask], 10 * np.log10(Sxx[freq_mask, :] + 1e-10),
                                         shading='gouraud', cmap='viridis')
                    ax4.set_title("Spectrogram (Z-axis, 0-60 Hz)")
                    ax4.set_ylabel("Frequency (Hz)")
                    ax4.set_ylim(0, 60)
                    try:
                        fig.colorbar(pcm, ax=ax4, label="dB")
                    except:
                        pass

        # Push the plot to the placeholder
        plot_placeholder.pyplot(fig)

        # Clean up memory
        plt.close(fig)

    # Small sleep to prevent CPU overload
    time.sleep(0.5)