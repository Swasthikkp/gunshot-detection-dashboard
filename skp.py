# Testing_code_mod_ctk_option1.py
# Replaced Zero Crossing Rate with Spectral Centroid & Spectral Roll-off

import os
import threading
import time
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import load_model
from playsound import playsound
import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
# --- Twilio and Environment Setup ---
from dotenv import load_dotenv
from twilio.rest import Client
import os

load_dotenv()

def send_sms_notification(message_text):
    """Send SMS alert using Twilio credentials stored in .env"""
    try:
        sid = os.getenv("TWILIO_SID")
        token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")

        # all 4 verified numbers
        to_numbers = [
            os.getenv("TO_NUMBER"),
            os.getenv("TO_NUMBER_2"),
            os.getenv("TO_NUMBER_3"),
            os.getenv("TO_NUMBER_4")
        ]

        client = Client(sid, token)
        for num in to_numbers:
            if num:  # skip empty slots
                client.messages.create(
                    body=message_text,
                    from_=from_number,
                    to=num
                )
                print(f"SMS notification sent to {num}")

    except Exception as e:
        print("SMS send failed:", e)

MODEL_PATH = "gunshot_classifier.h5"
ALERT_SOUND = "alert.wav"

GRAD_TOP = "#031423"
GRAD_BOTTOM = "#0b2c46"
SIDEBAR_GLASS = "#0d2b40"
GRAPH_BG = "#051922"
ACCENT = "#06b6d4"
GREEN = "#10b981"
RED = "#ef4444"
TEXT = "#e6f0ff"
MUTED = "#9bb7c9"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded.")
    except Exception as e:
        print("Model load failed:", e)
else:
    print("Model not found; running in demo mode.")

# ---------- Helper functions ----------
def play_alert(path):
    try:
        playsound(path)
        # Trigger SMS when the siren plays
        send_sms_notification("⚠️ Gunshot detected! Please check the system immediately.")
    except Exception as e:
        print("Alert or SMS failed:", e)

def preprocess_audio(file_path, max_len=128, duration=5.0):
    y, sr = librosa.load(file_path, sr=22050, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0,0),(0,max_len - mel_db.shape[1])), mode="constant")
    else:
        mel_db = mel_db[:,:max_len]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return mel_db, y, sr, chroma, spectral_centroid, spectral_rolloff

# ---------- Main App ----------
class FinalBlueApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Gunshot Classifier — Clean Blue Console")
        self.geometry("1280x800")
        self.minsize(1100, 700)

        self.file_path = None
        self.configure(fg_color=GRAD_TOP)

        self._bg_canvas = Canvas(self, highlightthickness=0)
        self._bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.bind("<Configure>", self._on_resize_draw_gradient)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, minsize=360)
        self.grid_columnconfigure(1, weight=3)

        self.sidebar = ctk.CTkFrame(self, corner_radius=24, fg_color=SIDEBAR_GLASS)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(24,12), pady=20)
        self.sidebar.grid_rowconfigure(12, weight=1)

        self.graph_panel = ctk.CTkFrame(self, corner_radius=14, fg_color=GRAPH_BG)
        self.graph_panel.grid(row=0, column=1, sticky="nsew", padx=(12,24), pady=20)
        self.graph_panel.grid_rowconfigure(0, weight=1)
        self.graph_panel.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(self.sidebar, text="Gunshot Detection System", font=ctk.CTkFont(size=22, weight="bold"), text_color=ACCENT)
        title.grid(row=0, column=0, padx=16, pady=(18,6), sticky="w")
        subtitle = ctk.CTkLabel(self.sidebar, text="Clean · Reliable · Enterprise", font=ctk.CTkFont(size=12, slant="italic"), text_color=MUTED)
        subtitle.grid(row=1, column=0, padx=16, pady=(0,12), sticky="w")

        self.file_label = ctk.CTkLabel(self.sidebar, text="Selected: —", text_color=TEXT)
        self.file_label.grid(row=2, column=0, padx=16, pady=(6,8), sticky="w")

        self.btn_upload = ctk.CTkButton(self.sidebar, text="Upload Audio", fg_color=ACCENT, command=self.upload_file)
        self.btn_upload.grid(row=3, column=0, padx=16, pady=(6,6), sticky="ew")
        self.btn_process = ctk.CTkButton(self.sidebar, text="Process Audio", fg_color=GREEN, command=self.process_audio)
        self.btn_process.grid(row=4, column=0, padx=16, pady=(6,6), sticky="ew")
        self.btn_clear = ctk.CTkButton(self.sidebar, text="Clear", fg_color=RED, command=self.clear_all)
        self.btn_clear.grid(row=5, column=0, padx=16, pady=(6,12), sticky="ew")

        self.status_label = ctk.CTkLabel(self.sidebar, text="Status: Idle", text_color=MUTED)
        self.status_label.grid(row=7, column=0, padx=16, pady=(0,8), sticky="w")

        ctk.CTkLabel(self.sidebar, text="Detection Confidence", text_color=TEXT, font=ctk.CTkFont(size=11, weight="bold")).grid(row=8, column=0, padx=16, pady=(8,6), sticky="w")
        self.conf_bar = ctk.CTkProgressBar(self.sidebar)
        self.conf_bar.grid(row=9, column=0, padx=16, pady=(0,10), sticky="ew")
        self.conf_bar.set(0)

        # Stats + Logs
        stats_card = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color="#08293c")
        stats_card.grid(row=10, column=0, padx=16, pady=(6,10), sticky="ew")
        ctk.CTkLabel(stats_card, text="Audio Statistics", text_color=TEXT, font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, padx=12, pady=(10,6), sticky="w")
        self.stat_file = ctk.CTkLabel(stats_card, text="File: —", text_color=MUTED)
        self.stat_file.grid(row=1, column=0, padx=12, sticky="w")
        self.stat_dur = ctk.CTkLabel(stats_card, text="Duration: —", text_color=MUTED)
        self.stat_dur.grid(row=2, column=0, padx=12, sticky="w")
        self.stat_sr = ctk.CTkLabel(stats_card, text="Sampling Rate: —", text_color=MUTED)
        self.stat_sr.grid(row=3, column=0, padx=12, sticky="w")
        self.stat_conf = ctk.CTkLabel(stats_card, text="Confidence: —", text_color=MUTED)
        self.stat_conf.grid(row=4, column=0, padx=12, pady=(0,10), sticky="w")

        log_card = ctk.CTkFrame(self.sidebar, corner_radius=10, fg_color="#08293c")
        log_card.grid(row=11, column=0, padx=16, pady=(6,18), sticky="nsew")
        ctk.CTkLabel(log_card, text="Detection Log", text_color=TEXT, font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, padx=12, pady=(10,6), sticky="w")
        self.log_frame = ctk.CTkScrollableFrame(log_card, corner_radius=6, fg_color="#071a2a", height=160)
        self.log_frame.grid(row=1, column=0, padx=12, pady=(0,12), sticky="nsew")

        self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 8), dpi=100)
        self._style_axes()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=12)

    # ----- Gradient -----
    def _on_resize_draw_gradient(self, event):
        w = max(self.winfo_width(), 300)
        h = max(self.winfo_height(), 300)
        steps = 60
        self._bg_canvas.delete("grad")
        top_rgb = tuple(int(GRAD_TOP.strip("#")[i:i+2], 16) for i in (0,2,4))
        bot_rgb = tuple(int(GRAD_BOTTOM.strip("#")[i:i+2], 16) for i in (0,2,4))
        for i in range(steps):
            t = i / (steps - 1)
            color = "#%02x%02x%02x" % tuple(int(top_rgb[c] + (bot_rgb[c] - top_rgb[c]) * t) for c in range(3))
            y0 = int(i * (h / steps)); y1 = int((i + 1) * (h / steps))
            self._bg_canvas.create_rectangle(0, y0, w, y1, fill=color, outline=color, tags="grad")
        self._bg_canvas.lower("grad")

    def _style_axes(self):
        self.fig.patch.set_facecolor(GRAPH_BG)
        for ax in self.axs:
            ax.set_facecolor(GRAPH_BG)
            ax.title.set_color(TEXT)
            ax.tick_params(colors=TEXT)
            for s in ax.spines.values(): s.set_color("#0e2430")
        titles = ["Waveform", "Mel Spectrogram", "Chroma Energy", "Spectral Centroid & Roll-off"]
        for ax, t in zip(self.axs, titles):
            ax.set_title(t, color=TEXT, fontsize=12)

    # ----- Log -----
    def _append_log(self, text, tag="info"):
        color = MUTED if tag=="info" else RED if tag=="gunshot" else GREEN
        label = ctk.CTkLabel(self.log_frame, text=f"[{time.strftime('%I:%M:%S %p')}] {text}", text_color=color, anchor="w", wraplength=320)
        label.pack(fill="x", padx=6, pady=4)
        self.log_frame.update_idletasks()

    # ----- Upload / Process -----
    def upload_file(self):
        path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg")])
        if path:
            self.file_path = path
            self.file_label.configure(text=f"Selected: {os.path.basename(path)}")
            self._append_log(f"Loaded: {os.path.basename(path)}")

    def process_audio(self):
        if not self.file_path:
            messagebox.showwarning("No file", "Please upload an audio file first.")
            return
        self.status_label.configure(text="Status: Processing...")
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        mel_db, y, sr, chroma, spec_cent, spec_roll = preprocess_audio(self.file_path)
        if model is not None:
            X = mel_db[np.newaxis, ..., np.newaxis]
            preds = model.predict(X)
            classes = ["Gunshot", "Normal"]
            idx = int(np.argmax(preds))
            prediction = classes[idx]
            conf = float(np.max(preds)) * 100.0
        else:
            prediction, conf = "Gunshot", 88.0
        if prediction == "Gunshot":
            threading.Thread(target=lambda: play_alert(ALERT_SOUND), daemon=True).start()
        if self.winfo_exists():
            self.after(0, self._display_results, mel_db, y, sr, chroma, spec_cent, spec_roll, prediction, conf)

    # ----- Display -----
    def _display_results(self, mel_db, y, sr, chroma, spec_cent, spec_roll, prediction, conf):
        for ax in self.axs: ax.cla()
        librosa.display.waveshow(y, sr=sr, ax=self.axs[0], color=ACCENT)
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=self.axs[1], cmap="magma")
        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=self.axs[2], cmap="viridis")
        times = np.linspace(0, len(y)/sr, spec_cent.shape[1])
        self.axs[3].plot(times, spec_cent[0], color="#06b6d4", linewidth=1.5, label="Spectral Centroid")
        self.axs[3].plot(times, spec_roll[0], color="#f59e0b", linewidth=1.5, label="Spectral Roll-off")
        self.axs[3].legend(facecolor=GRAPH_BG, labelcolor=TEXT, fontsize=8)

        for ax in self.axs:
            ax.tick_params(colors=TEXT)
            for s in ax.spines.values(): s.set_color("#0e2430")
        self.fig.tight_layout()
        self.canvas.draw_idle()

        self.stat_file.configure(text=f"File: {os.path.basename(self.file_path)}")
        self.stat_dur.configure(text=f"Duration: {len(y)/sr:.2f} sec")
        self.stat_sr.configure(text=f"Sampling Rate: {sr}")
        self.stat_conf.configure(text=f"Confidence: {conf:.2f}%")
        self.conf_bar.set(min(conf/100.0,1.0))

        if prediction == "Gunshot":
            self.conf_bar.configure(progress_color=RED)
            self._append_log(f"{prediction} ({conf:.2f}%)", tag="gunshot")
            messagebox.showwarning("Alert", f"🚨 GUNSHOT DETECTED ({conf:.2f}%)")
        else:
            self.conf_bar.configure(progress_color=GREEN)
            self._append_log(f"{prediction} ({conf:.2f}%)", tag="normal")
        self.status_label.configure(text="Status: Complete")

    def clear_all(self):
        self.file_path = None
        self.file_label.configure(text="Selected: —")
        self.conf_bar.set(0)
        for ax in self.axs: ax.cla()
        self.canvas.draw_idle()
        for c in self.log_frame.winfo_children(): c.destroy()
        self._append_log("Cleared dashboard.")
        self.status_label.configure(text="Status: Idle")

if __name__ == "__main__":
    app = FinalBlueApp()
    app.mainloop()
