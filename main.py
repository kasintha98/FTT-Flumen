import streamlit as st
import json
import collections
import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt
from scipy import signal
import time
import sys

# -------- CONFIG --------
MQTT_BROKER = "dev.flumina.de"
MQTT_TOPIC = "root/acc_fifo_batch"
MAX_POINTS = 800
SAMPLE_RATE = 800

# -------- PAGE CONFIG --------
st.set_page_config(page_title="Vibration Monitor", layout="wide")
st.title("🔴 Live Vibration Dashboard")


# -------- SHARED STATE CLASS --------
# This class lives in the memory and is shared between the MQTT thread
# and the Streamlit main loop.
class SharedState:
    def __init__(self):
        self.x_buf = collections.deque(maxlen=MAX_POINTS)
        self.y_buf = collections.deque(maxlen=MAX_POINTS)
        self.z_buf = collections.deque(maxlen=MAX_POINTS)
        self.status = "Disconnected"
        self.message_count = 0
        self.last_update = time.time()
        self.logs = collections.deque(maxlen=10)  # Keep last 10 logs

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {msg}")
        print(f"LOG: {msg}")


# Use @st.cache_resource so this object is created ONLY ONCE and never reset
@st.cache_resource
def get_shared_state():
    return SharedState()


state = get_shared_state()


# -------- MQTT CALLBACKS --------
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        state.status = "Connected"
        state.log("Connected to Broker! Subscribing...")
        client.subscribe(MQTT_TOPIC)
    else:
        state.status = f"Failed: {reason_code}"
        state.log(f"Connection failed code: {reason_code}")


def on_message(client, userdata, msg):
    try:
        # Update heartbeat
        state.last_update = time.time()
        state.message_count += 1

        payload = msg.payload.decode()
        data = json.loads(payload)

        new_x, new_y, new_z = [], [], []

        # Helper to extract points
        def extract(samples):
            for s in samples:
                new_x.append(s[0])
                new_y.append(s[1])
                new_z.append(s[2])

        # Handle List format or Dict format
        if isinstance(data, list):
            for batch in data:
                extract(batch.get("s", []))
        elif isinstance(data, dict) and "s" in data:
            extract(data.get("s", []))

        # Push to shared state
        state.x_buf.extend(new_x)
        state.y_buf.extend(new_y)
        state.z_buf.extend(new_z)

    except Exception as e:
        state.log(f"Parse Error: {e}")


# -------- START MQTT --------
@st.cache_resource
def start_mqtt():
    state = get_shared_state()
    client_id = f"streamlit_vib_{int(time.time())}"

    # Try creating client (Handle both Paho v1 and v2)
    try:
        client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        # Fallback for older Paho versions
        client = mqtt.Client(client_id=client_id)

    client.on_connect = on_connect
    client.on_message = on_message

    state.log(f"Connecting to {MQTT_BROKER}...")
    try:
        client.connect(MQTT_BROKER, 1883, 60)
        client.loop_start()
        return client
    except Exception as e:
        state.log(f"Connection Error: {e}")
        state.status = "Error"
        return None


# Start the client (only runs once due to cache_resource)
client = start_mqtt()

# -------- DASHBOARD LAYOUT --------
col1, col2, col3 = st.columns(3)
status_indicator = col1.empty()
msg_counter = col2.empty()
debug_expander = st.sidebar.expander("Debug Console", expanded=True)
debug_text = debug_expander.empty()

plot_placeholder = st.empty()

# -------- MAIN LOOP --------
st.sidebar.text("Press 'Stop' in top right to kill script")

# We loop here to update the charts
while True:
    # 1. Update Diagnostics
    if state.status == "Connected":
        status_indicator.success(f"Status: {state.status}")
    else:
        status_indicator.warning(f"Status: {state.status}")

    msg_counter.metric("Messages Received", state.message_count)
    debug_text.code("\n".join(list(state.logs)[::-1]))  # Show logs in sidebar

    # 2. Check Data & Plot
    if len(state.x_buf) > 10:
        # Copy data safely
        x_data = list(state.x_buf)
        y_data = list(state.y_buf)
        z_data = list(state.z_buf)

        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 1, hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Plot Time Domain
        ax1.plot(x_data, label="X", lw=1)
        ax1.plot(y_data, label="Y", lw=1)
        ax1.plot(z_data, label="Z", lw=1)
        ax1.set_title(f"Live Data ({len(x_data)} points)")
        ax1.grid(True)
        ax1.legend(loc="upper right")

        # Plot FFT
        if len(x_data) >= 64:
            x_arr = np.array(x_data)
            freqs = np.fft.rfftfreq(len(x_arr), 1 / SAMPLE_RATE)
            mag_x = np.abs(np.fft.rfft(x_arr))

            ax2.plot(freqs, mag_x, color='blue', alpha=0.7)
            ax2.set_title("FFT (X-Axis)")
            ax2.set_xlim(0, 60)
            ax2.grid(True)

        plot_placeholder.pyplot(fig)
        plt.close(fig)
    else:
        # Show waiting message if connected but no data yet
        if state.status == "Connected":
            plot_placeholder.info("Connected! Waiting for data stream...")

    time.sleep(0.5)