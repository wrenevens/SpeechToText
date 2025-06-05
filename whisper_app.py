import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import whisper
import os
from pathlib import Path
import threading
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import queue

class WhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Offline Speech-to-Text")
        self.root.geometry("600x500")

        # Available Whisper models
        self.models = ["tiny", "base", "small", "medium", "large"]
        self.selected_model = tk.StringVar(value=self.models[0])
        self.model = None

        # Whisper model cache directory
        self.cache_dir = Path.home() / ".cache" / "whisper"

        # Audio recording parameters
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.record_duration = 5  # Seconds
        self.output_wav = "recorded_audio.wav"

        # Recording state
        self.is_recording = False
        self.recording_data = []
        self.recording_stream = None
        self.audio_queue = queue.Queue()

        # GUI Elements
        self.create_gui()

    def create_gui(self):
        # Model selection frame
        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=10, padx=10, fill="x")

        tk.Label(model_frame, text="Select Whisper Model:").pack(side="left")
        model_menu = ttk.Combobox(model_frame, textvariable=self.selected_model, 
                                values=self.models, state="readonly")
        model_menu.pack(side="left", padx=5)

        # Load/Download model button
        load_button = tk.Button(model_frame, text="Load/Download Model", command=self.load_model_thread)
        load_button.pack(side="left", padx=5)

        # Audio device selection frame
        device_frame = tk.Frame(self.root)
        device_frame.pack(pady=5, padx=10, fill="x")
        tk.Label(device_frame, text="Select Input Device:").pack(side="left")
        self.input_devices = self.get_input_devices()
        self.selected_device = tk.StringVar()
        device_names = [f"{i}: {name}" for i, name in self.input_devices]
        if device_names:
            self.selected_device.set(device_names[0])
        self.device_menu = ttk.Combobox(device_frame, textvariable=self.selected_device, values=device_names, state="readonly", width=40)
        self.device_menu.pack(side="left", padx=5)

        # File selection frame
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10, padx=10, fill="x")

        tk.Label(file_frame, text="Select WAV File:").pack(side="left")
        self.file_path = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.file_path, width=40).pack(side="left", padx=5)
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(side="left")

        # Record/Stop buttons
        tk.Button(self.root, text="Record Audio", command=self.start_recording_thread).pack(pady=5)
        tk.Button(self.root, text="Stop Recording", command=self.stop_recording).pack(pady=5)

        # Transcribe button
        tk.Button(self.root, text="Transcribe Selected File", command=self.transcribe_thread).pack(pady=5)

        # Transcription output
        tk.Label(self.root, text="Transcription Result:").pack(pady=5)
        self.result_text = tk.Text(self.root, height=10, width=60)
        self.result_text.pack(pady=5, padx=10)

        # Status label
        self.status = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status).pack(pady=5)

    def browse_file(self):
        file = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file:
            self.file_path.set(file)

    def check_model_exists(self, model_name):
        # Check if model file exists in Whisper cache
        model_path = self.cache_dir / f"{model_name}.pt"
        return model_path.exists()

    def load_model_thread(self):
        # Run model loading in a separate thread
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        model_name = self.selected_model.get()
        self.status.set(f"Checking if {model_name} model is downloaded...")
        self.root.update()

        if self.check_model_exists(model_name):
            self.status.set(f"Model {model_name} found locally. Loading...")
            self.root.update()
        else:
            self.status.set(f"Downloading model {model_name}...")
            self.root.update()

        try:
            self.model = whisper.load_model(model_name)
            self.status.set(f"Model {model_name} loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status.set("Error loading model")
        self.root.update()

    def get_input_devices(self):
        # Returns a list of (index, name) for input devices
        devices = []
        try:
            info = sd.query_devices()
            for idx, dev in enumerate(info):
                if dev['max_input_channels'] > 0:
                    devices.append((idx, dev['name']))
        except Exception as e:
            messagebox.showerror("Error", f"Could not query audio devices: {str(e)}")
        return devices

    def start_recording_thread(self):
        threading.Thread(target=self.start_recording, daemon=True).start()

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.recording_data = []
        self.status.set("Recording... Press Stop to finish.")
        self.result_text.delete(1.0, tk.END)
        self.root.update()
        device_str = self.selected_device.get()
        device_idx = int(device_str.split(":")[0]) if device_str else None

        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.recording_data.append(indata.copy())

        self.recording_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            device=device_idx,
            callback=callback
        )
        self.recording_stream.start()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.recording_stream:
            self.recording_stream.stop()
            self.recording_stream.close()
            self.recording_stream = None

        # Concatenate all recorded chunks
        if self.recording_data:
            audio = np.concatenate(self.recording_data, axis=0)
            recording_int16 = np.int16(audio * 32767)
            abs_wav_path = os.path.abspath(self.output_wav)
            with open(abs_wav_path, 'wb') as f:
                wavfile.write(f, self.sample_rate, recording_int16)
                f.flush()
                os.fsync(f.fileno())
            self.file_path.set(abs_wav_path)
            self.status.set(f"Audio recorded and saved to {abs_wav_path}")
        else:
            self.status.set("No audio recorded.")

    def record_and_transcribe_thread(self):
        # Run recording and transcription in a separate thread
        threading.Thread(target=self.record_and_transcribe, daemon=True).start()

    def record_and_transcribe(self):
        if not self.model:
            messagebox.showerror("Error", "Please load a model first!")
            return

        self.status.set("Recording audio...")
        self.result_text.delete(1.0, tk.END)
        self.root.update()

        try:
            import time
            # Get selected device index
            device_str = self.selected_device.get()
            device_idx = int(device_str.split(":")[0]) if device_str else None
            # Record audio
            recording = sd.rec(int(self.record_duration * self.sample_rate), 
                             samplerate=self.sample_rate, channels=1, device=device_idx)
            sd.wait()  # Wait until recording is finished
            self.status.set("Recording finished, saving as WAV...")
            self.root.update()

            # Convert float32 to int16 for WAV
            recording_int16 = np.int16(recording * 32767)
            abs_wav_path = os.path.abspath(self.output_wav)
            # Write WAV file using context manager
            with open(abs_wav_path, 'wb') as f:
                wavfile.write(f, self.sample_rate, recording_int16)
                f.flush()
                os.fsync(f.fileno())
            # Short delay to ensure file is written
            time.sleep(0.1)

            # Update file path to recorded audio
            self.file_path.set(abs_wav_path)

            # Ensure file is written before transcription
            if not os.path.exists(abs_wav_path):
                raise FileNotFoundError(f"WAV file {abs_wav_path} not found after saving.")

            # Print path for debugging
            print(f"Transcribing file: {abs_wav_path}")

            # Transcribe the recorded audio
            self.status.set("Transcribing recorded audio...")
            self.root.update()
            result = self.model.transcribe(abs_wav_path)
            self.result_text.insert(tk.END, result["text"])
            self.status.set("Transcription completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Recording or transcription failed: {str(e)}")
            self.status.set("Error during recording/transcription")
        self.root.update()

    def transcribe_thread(self):
        # Run transcription of selected file in a separate thread
        threading.Thread(target=self.transcribe, daemon=True).start()

    def transcribe(self):
        if not self.model:
            messagebox.showerror("Error", "Please load a model first!")
            return

        audio_file = self.file_path.get()
        if not audio_file or not os.path.exists(audio_file):
            messagebox.showerror("Error", "Please select a valid WAV file!")
            return

        self.status.set("Transcribing audio...")
        self.result_text.delete(1.0, tk.END)
        self.root.update()

        try:
            result = self.model.transcribe(audio_file)
            self.result_text.insert(tk.END, result["text"])
            self.status.set("Transcription completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Transcription failed: {str(e)}")
            self.status.set("Error during transcription")
        self.root.update()

    def record_audio_only(self):
        """Record audio from the selected device and save as a WAV file, but do not transcribe."""
        self.status.set("Recording audio only...")
        self.result_text.delete(1.0, tk.END)
        self.root.update()
        try:
            import time
            # Get selected device index
            device_str = self.selected_device.get()
            device_idx = int(device_str.split(":")[0]) if device_str else None
            # Record audio
            recording = sd.rec(int(self.record_duration * self.sample_rate), 
                             samplerate=self.sample_rate, channels=1, device=device_idx)
            sd.wait()  # Wait until recording is finished
            self.status.set("Recording finished, saving as WAV...")
            self.root.update()
            # Convert float32 to int16 for WAV
            recording_int16 = np.int16(recording * 32767)
            abs_wav_path = os.path.abspath(self.output_wav)
            # Write WAV file using context manager
            with open(abs_wav_path, 'wb') as f:
                wavfile.write(f, self.sample_rate, recording_int16)
                f.flush()
                os.fsync(f.fileno())
            # Short delay to ensure file is written
            time.sleep(0.1)
            # Update file path to recorded audio
            self.file_path.set(abs_wav_path)
            # Ensure file is written
            if not os.path.exists(abs_wav_path):
                raise FileNotFoundError(f"WAV file {abs_wav_path} not found after saving.")
            self.status.set(f"Audio recorded and saved to {abs_wav_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Recording failed: {str(e)}")
            self.status.set("Error during recording")
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = WhisperApp(root)
    root.mainloop()