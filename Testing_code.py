import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pydub import AudioSegment
from pydub.playback import play
import os
from twilio.rest import Client

# ---------------------------
# CONFIG
# ---------------------------
account_sid = 'ACffe9378883f273cc43b466661ed25f76'
auth_token = '133c0d9dd5110efce57e48e944d74692'
client = Client(account_sid, auth_token)





MODEL_PATH = "gunshot_classifier.h5"
ALERT_SOUND ='C:/Users/kpswa/OneDrive/Desktop/Gunshot_sound/alert.WAV'

model = load_model(MODEL_PATH)


# ---------------------------
# AUDIO PROCESSING
# ---------------------------
def preprocess_audio(file_path, max_len=128):
    y, sr = librosa.load(file_path, sr=22050, duration=2.0)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec_db.shape[1] < max_len:
        pad_width = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_len]
    return mel_spec_db, y, sr


# ---------------------------
# GUI CLASS
# ---------------------------
class GunshotDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔫 Gunshot Sound Classifier")
        self.root.configure(bg="#1e1e2e")

        # Window setup
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        w, h = int(screen_w * 0.8), int(screen_h * 0.8)
        x, y = (screen_w - w) // 2, (screen_h - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(900, 700)

        self.file_path = None

        # Title
        tk.Label(root, text="Gunshot Sound Classifier",
                 font=("Arial Rounded MT Bold", 28),
                 fg="white", bg="#1e1e2e").pack(pady=20)

        # Buttons
        btn_frame = tk.Frame(root, bg="#1e1e2e")
        btn_frame.pack(pady=10)

        self.upload_btn = tk.Button(
            btn_frame, text="📂 Upload Audio File", font=("Arial", 14),
            bg="#3b82f6", fg="white", relief="flat",
            command=self.upload_file, width=20)
        self.upload_btn.grid(row=0, column=0, padx=15, pady=5)

        self.process_btn = tk.Button(
            btn_frame, text="🧠 Process Audio", font=("Arial", 14),
            bg="#22c55e", fg="white", relief="flat",
            command=self.process_audio, width=20, state="disabled")
        self.process_btn.grid(row=0, column=1, padx=15, pady=5)

        # 🧹 Clear Plot Button
        self.clear_btn = tk.Button(
            btn_frame, text="🧹 Clear Plot", font=("Arial", 14),
            bg="#ef4444", fg="white", relief="flat",
            command=self.clear_plot, width=20)
        self.clear_btn.grid(row=0, column=2, padx=15, pady=5)

        # File label
        self.file_label = tk.Label(root, text="No file selected",
                                   font=("Arial", 12),
                                   bg="#1e1e2e", fg="#cbd5e1")
        self.file_label.pack(pady=5)

        # Result label
        self.result_label = tk.Label(root, text="",
                                     font=("Arial Rounded MT Bold", 22),
                                     bg="#1e1e2e")
        self.result_label.pack(pady=10)

        # Matplotlib plot area
        self.fig, self.ax = plt.subplots(2, 1, figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor('#1e1e2e')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=20, pady=20)

        # Responsive
        self.root.bind("<Configure>", self.on_resize)

    # ---------------------------
    def on_resize(self, event):
        self.canvas_widget.config(width=event.width - 40, height=event.height - 200)

    # ---------------------------
    def upload_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac")]
        )
        if self.file_path:
            filename = os.path.basename(self.file_path)
            self.file_label.config(text=f"✅ Selected: {filename}", fg="#a5f3fc")
            self.process_btn.config(state="normal")
            self.result_label.config(text="")
        else:
            self.file_label.config(text="❌ No file selected", fg="red")
            self.process_btn.config(state="disabled")

    # ---------------------------
    def process_audio(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select an audio file first.")
            return

        mel_spec_db, y, sr = preprocess_audio(self.file_path)
        X = mel_spec_db[np.newaxis, ..., np.newaxis]
        pred = model.predict(X)
        classes = ["Gunshot", "Normal"]
        prediction = classes[np.argmax(pred)]
        confidence = float(np.max(pred)) * 100

        # Plot waveform + spectrogram
        self.ax[0].cla()
        librosa.display.waveshow(y, sr=sr, ax=self.ax[0], color="#60a5fa")
        self.ax[0].set_title("Audio Waveform", color="white")
        self.ax[1].cla()
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel',
                                 ax=self.ax[1], cmap="viridis")
        self.ax[1].set_title("Mel Spectrogram", color="white")
        for a in self.ax:
            a.tick_params(colors="white")
            a.set_facecolor("#1e1e2e")
        self.fig.tight_layout()
        self.canvas.draw()

        # Result display first
        if prediction == "Gunshot":
            self.result_label.config(text=f"🚨 GUNSHOT DETECTED! ({confidence:.2f}%)", fg="#f87171")
            # Schedule sound to play after UI updates
            self.root.after(500, self.play_alert)
        else:
            self.result_label.config(text=f"✅ Normal Sound ({confidence:.2f}%)", fg="#4ade80")

    # ---------------------------
    def clear_plot(self):
        for ax in self.ax:
            ax.cla()
        self.canvas.draw()
        self.result_label.config(text="")
        self.file_label.config(text="No file selected", fg="#cbd5e1")
        self.process_btn.config(state="disabled")
        self.file_path = None

    # ---------------------------
    def play_alert(self):
        if not os.path.exists(ALERT_SOUND):
            messagebox.showwarning("Missing Alert Sound", "⚠️ alert.wav or alert.mp3 file not found!")
            return
        try:
            sound = AudioSegment.from_file(ALERT_SOUND)
            short_sound = sound[:10000]
            play(short_sound)
            client.messages.create(
                            from_='+18148014575',
                            body='Gunshot Detected,Alert.. ',
                            to='+919036393685'
                        )
        except Exception as e:
            messagebox.showerror("Playback Error", f"Unable to play alert sound.\n{e}")


# ---------------------------
# MAIN APP
# ---------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = GunshotDetectorApp(root)
    root.mainloop()
