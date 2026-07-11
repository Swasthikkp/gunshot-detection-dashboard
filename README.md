# Gunshot Detection Dashboard

A real-time gunshot detection desktop application built with deep learning. It classifies uploaded or live audio as gunshot vs. normal, displays a live audio-analysis dashboard, and automatically triggers an alarm sound and SMS alerts to multiple recipients when a gunshot is detected.

## Features

- **Desktop GUI dashboard** (built with CustomTkinter) — upload an audio file, process it, and view results in a clean dark-themed interface
- **Deep learning classification** — a trained Keras/TensorFlow model (`gunshot_classifier.h5`) classifies audio as "Gunshot" or "Normal" with a confidence score
- **Multi-feature audio analysis** — extracts and visualizes:
  - Waveform
  - Mel spectrogram
  - Chroma energy
  - Spectral centroid & spectral roll-off
- **Automatic siren alert** — plays an alert sound (`alert.wav`) the moment a gunshot is detected
- **Multi-recipient SMS alerts** — sends an emergency SMS via Twilio to up to 4 verified phone numbers simultaneously
- **Live detection log & confidence meter** — sidebar log of all detections with timestamps, plus a live confidence progress bar
- **Demo mode fallback** — if the model file isn't found, the app still runs and simulates a detection so the UI can be demoed without the trained model

## Tech stack

- **Python**
- **TensorFlow / Keras** — model inference (`load_model`, `.h5` model)
- **Librosa** — audio loading and feature extraction (mel spectrogram, chroma, spectral centroid, spectral roll-off)
- **CustomTkinter** — desktop GUI framework
- **Matplotlib** — real-time waveform/spectrogram visualization embedded in the GUI
- **Twilio** — SMS alert delivery
- **python-dotenv** — environment variable management for API credentials
- **playsound** — local audio alert playback

## How it works

1. User uploads an audio file (`.wav`, `.mp3`, `.flac`, `.ogg`) through the dashboard
2. Audio is loaded via Librosa and converted into a mel spectrogram, chroma features, spectral centroid, and spectral roll-off
3. The mel spectrogram is fed into the trained CNN model, which outputs a Gunshot/Normal prediction with a confidence score
4. If classified as a gunshot:
   - An alert sound plays locally
   - An SMS notification is sent via Twilio to all configured recipient numbers
   - The detection is logged in the sidebar with a timestamp
5. All four audio visualizations (waveform, mel spectrogram, chroma, spectral centroid/roll-off) update live in the dashboard

## Setup

```bash
git clone https://github.com/Swasthikkp/gunshot-detection-dashboard.git
cd gunshot-detection-dashboard
pip install tensorflow librosa customtkinter matplotlib twilio python-dotenv playsound numpy
```

Create your own `.env` file in the project root (this file must **never** be committed — it's excluded via `.gitignore`):

```
TWILIO_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_FROM_NUMBER=your_twilio_phone_number
TO_NUMBER=recipient_1_number
TO_NUMBER_2=recipient_2_number
TO_NUMBER_3=recipient_3_number
TO_NUMBER_4=recipient_4_number
```

Make sure `gunshot_classifier.h5` (the trained model) is present in the project root — see [`datasets and ml (.h5)file.md`](<datasets and ml (.h5)file.md>) for the model/dataset reference. If the model file isn't found, the app runs in demo mode with simulated predictions.

Run the app:

```bash
python skp.py
```

## Project structure

```
gunshot-detection-dashboard/
├── skp.py                          # Main GUI application (CustomTkinter dashboard)
├── Training_code.py                 # Model training script
├── Testing_code.py                  # Model testing/evaluation script
├── Testing_code_mod_ctk.py          # CustomTkinter-based testing variant
├── datasets and ml (.h5)file.md     # Notes on dataset and trained model file
├── alert.wav / alert.mp3            # Alert sounds played on detection
├── test_env.py                      # Environment variable test script
└── README.md
```

## Security note

Twilio credentials are loaded from a `.env` file at runtime and must be kept private. If you're forking or reusing this project, generate your own Twilio credentials and never commit your `.env` file.

## Future improvements

- Continuous live-microphone monitoring instead of file-upload-only detection
- Geolocation tagging on SMS alerts
- Model robustness testing against false positives (fireworks, construction, vehicle backfire)
- Packaged executable for non-technical deployment

## Author

Swasthik K P — [LinkedIn](https://linkedin.com/in/swasthik-k-p-7b927b377) · [GitHub](https://github.com/Swasthikkp)
