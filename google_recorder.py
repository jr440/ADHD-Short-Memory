import os
import time
from datetime import datetime
import collections
import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd
import webrtcvad
from dotenv import load_dotenv
import sqlite3

# Import Google Cloud clients
from google.cloud import speech
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- Google Cloud Configuration ---
# This script uses Application Default Credentials (ADC).
# Ensure you have set the GOOGLE_APPLICATION_CREDENTIALS environment variable.
try:
    speech_client = speech.SpeechClient()
    # Configure the Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Summarization will not work.")
        summarization_model = None
    else:
        genai.configure(api_key=gemini_api_key)
        summarization_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    print(f"Error initializing Google Cloud clients: {e}")
    print("Please ensure you have authenticated correctly and have the necessary APIs enabled.")
    exit()


# --- Audio Configuration ---
FS = 16000  # Sample rate (Hz) - Must be 8k, 16k, 32k, or 48k for VAD
CHANNELS = 1  # Mono
DTYPE = 'int16'  # Data type for recording

# VAD specific parameters
VAD_MODE = 3  # Aggressiveness of VAD (0=least aggressive, 3=most aggressive)
VAD_FRAME_DURATION_MS = 30  # Frame size for VAD (10, 20, or 30 ms)
VAD_FRAME_SIZE = int(FS * VAD_FRAME_DURATION_MS / 1000)

# Buffering for speech detection
PADDING_FRAMES = 10  # Number of silent frames to add around speech
RING_BUFFER_DURATION = 1  # Seconds of audio to keep in buffer
RING_BUFFER_SIZE = int(FS * RING_BUFFER_DURATION / VAD_FRAME_SIZE)

# Thresholds for recording
SPEECH_INITIATE_FRAMES = 5  # 150ms of speech to start
SPEECH_END_FRAMES = 10  # 300ms of non-speech to end

# Project Directories
RECORDING_DIR = 'audio_segments'
TRANSCRIPT_DIR = 'transcripts'
SUMMARY_DIR = 'summaries'


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    os.makedirs(RECORDING_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

def init_db():
    """Initializes the database and creates the transcripts table if it doesn't exist."""
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    content TEXT NOT NULL
    )
    ''')
    conn.commit()
    conn.close()

def save_transcript_to_db(transcript_text):
    """Saves a single transcript segment to the SQLite database."""
    conn = sqlite3.connect('memory.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO transcripts (content) VALUES (?)", (transcript_text,))
    conn.commit()
    conn.close()
    print(f"Saved to database: {transcript_text[:50]}...")

def get_audio_input_device_index():
    """Lists available audio devices and returns the selected index."""
    print("Available audio input devices:")
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    for i, device in enumerate(input_devices):
        print(f"  {device['index']}: {device['name']} (Max input channels: {device['max_input_channels']})")

    # Manually specify the index of the input device
    device_index = 1  # Replace with a valid index from the list
    print(f"Using device index: {device_index}")
    return device_index


def record_and_vad_process():
    """
    Records audio, applies VAD, and yields speech segments.
    """
    vad = webrtcvad.Vad(VAD_MODE)
    ring_buffer = collections.deque(maxlen=RING_BUFFER_SIZE)
    voiced_frames_counter = 0
    unvoiced_frames_counter = 0
    recording_active = False

    device_index = get_audio_input_device_index()
    if device_index is None:
        return

    print("Listening for voice activity... (Press Ctrl+C to stop)")

    try:
        with sd.InputStream(samplerate=FS, channels=CHANNELS, dtype=DTYPE, device=device_index) as stream:
            audio_data_buffer = []

            while True:
                frame_data, overflowed = stream.read(VAD_FRAME_SIZE)
                if overflowed:
                    print("Audio input buffer overflowed!")

                is_speech = vad.is_speech(frame_data.tobytes(), FS)
                ring_buffer.append((frame_data, is_speech))

                if not recording_active:
                    if is_speech:
                        voiced_frames_counter += 1
                        if voiced_frames_counter >= SPEECH_INITIATE_FRAMES:
                            recording_active = True
                            print("Speech detected! Starting recording.")
                            for buffered_frame, _ in list(ring_buffer):
                                audio_data_buffer.append(buffered_frame)
                    else:
                        voiced_frames_counter = 0
                else:
                    audio_data_buffer.append(frame_data)
                    if not is_speech:
                        unvoiced_frames_counter += 1
                        if unvoiced_frames_counter >= SPEECH_END_FRAMES:
                            recording_active = False
                            print("Silence detected. Ending recording.")
                            yield np.concatenate(audio_data_buffer)
                            audio_data_buffer = []
                            voiced_frames_counter = 0
                            unvoiced_frames_counter = 0
                    else:
                        unvoiced_frames_counter = 0

    except KeyboardInterrupt:
        print("\nStopping listening.")
    except Exception as e:
        print(f"Error during audio stream: {e}")
    finally:
        if recording_active and audio_data_buffer:
            print("\nEnding last recording segment.")
            yield np.concatenate(audio_data_buffer)

def transcribe_audio_google(audio_numpy_data):
    """Transcribes audio using Google Cloud Speech-to-Text."""
    print("Transcribing audio with Google Cloud Speech-to-Text...")
    try:
        # In-memory audio content
        content = audio_numpy_data.tobytes()
        audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=FS,
            language_code="en-US",
        )

        response = speech_client.recognize(config=config, audio=audio)

        if response.results:
            return response.results[0].alternatives[0].transcript
        else:
            return "[No speech detected in audio]"
    except Exception as e:
        print(f"Google Cloud Speech-to-Text transcription failed: {e}")
        return "[Cloud Transcription Failed]"

def summarize_text_gemini(text):
    """Summarizes text using the Google Gemini API."""
    print("Summarizing text with Google Gemini API...")
    if not summarization_model:
        return "[Summarization model not available]"
    try:
        prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        response = summarization_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Google Gemini API summarization failed: {e}")
        return "[Cloud Summarization Failed]"

def main():
    """Main function to run the recording, transcription, and summarization loop."""
    ensure_dirs()
    init_db()
    all_recorded_text_in_hour = []
    current_hour_start_time = time.time()

    for segment_audio_data in record_and_vad_process():
        if segment_audio_data.size == 0:
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the audio segment
        segment_filename = f"vad_segment_{timestamp}.wav"
        segment_path = os.path.join(RECORDING_DIR, segment_filename)
        wavfile.write(segment_path, FS, segment_audio_data)
        print(f"Saved audio segment to {segment_path}")

        transcript_segment = transcribe_audio_google(segment_audio_data)
        all_recorded_text_in_hour.append(transcript_segment)
        
        print(f"Transcript: {transcript_segment}")

        # Save individual transcript to the database
        save_transcript_to_db(transcript_segment)

        # Check for hourly summarization
        if (time.time() - current_hour_start_time) >= 3600:
            full_hour_text = " ".join(all_recorded_text_in_hour)
            if full_hour_text.strip():
                hourly_summary = summarize_text_gemini(full_hour_text)
                summary_timestamp = datetime.now().strftime("%Y%m%d_%H")
                summary_filename = f"summary_{summary_timestamp}.txt"
                with open(os.path.join(SUMMARY_DIR, summary_filename), "w") as f:
                    f.write(hourly_summary)
                print(f"\n--- Hourly Summary ({summary_timestamp}) ---\n{hourly_summary}\n---------------------------------\n")
            else:
                print("No new speech to summarize in the last hour.")

            all_recorded_text_in_hour = []
            current_hour_start_time = time.time()

if __name__ == "__main__":
    main()