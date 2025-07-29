import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import webrtcvad # Import the VAD library
import collections # For a deque (double-ended queue)
import time
import os
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv() # This line loads the variables from .env into os.environ

# --- Configuration ---
FS = 16000  # Sample rate (Hz) - Crucial for VAD (must be 8k, 16k, 32k, or 48k)
CHANNELS = 1 # Mono
DTYPE = 'int16' # Data type for recording (16-bit samples)

# VAD specific parameters
VAD_MODE = 3 # Aggressiveness of VAD (0=least aggressive, 3=most aggressive). Test to find best.
VAD_FRAME_DURATION_MS = 30 # Frame size for VAD (10, 20, or 30 ms).
VAD_FRAME_SIZE = int(FS * VAD_FRAME_DURATION_MS / 1000) # Number of samples per VAD frame

# Buffering for continuous speech detection
# How many non-speech frames to allow before considering speech ended
# And how many non-speech frames to pre-pend to a speech segment
PADDING_FRAMES = 10 # Number of silent frames to add around speech (e.g., 300ms if 30ms frames)
RING_BUFFER_DURATION = 1 # Seconds of audio to keep in buffer before current speech
RING_BUFFER_SIZE = int(FS * RING_BUFFER_DURATION / VAD_FRAME_SIZE)

# Thresholds for starting/stopping recording
# How many speech frames needed to start recording
SPEECH_INITIATE_FRAMES = 5 # e.g., 5 * 30ms = 150ms of speech to start
# How many non-speech frames needed to stop recording
SPEECH_END_FRAMES = 10 # e.g., 10 * 30ms = 300ms of non-speech to end

# Project Directories
RECORDING_DIR = 'audio_segments' # Where raw detected speech segments are saved
TRANSCRIPT_DIR = 'transcripts' # Where transcribed text goes
SUMMARY_DIR = 'summaries' # Where summaries go

# --- AI Models (Placeholders for now) ---
# For Local Whisper (Speech-to-Text):
# import whisper
# whisper_model = None
# try:
#     print("Loading Whisper model (base.en)... This might take a while on first run.")
#     # Using fp16=False for CPU compatibility. Remove if using a proper GPU setup.
#     whisper_model = whisper.load_model("base.en", device="cpu", fp16=False)
#     print("Whisper model loaded.")
# except Exception as e:
#     print(f"Error loading Whisper model: {e}. Make sure you have enough RAM and internet connection on first download.")

# For Cloud API (e.g., OpenAI for STT and Summarization):
# from openai import OpenAI
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure this is set as an environment variable
# if not OPENAI_API_KEY:
#     print("WARNING: OPENAI_API_KEY environment variable not set. Cloud APIs will not work.")
# client = OpenAI(api_key=OPENAI_API_KEY)


def ensure_dirs():
    os.makedirs(RECORDING_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

def get_audio_input_device_index():
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
    voiced_frames_counter = 0 # Count consecutive voiced frames
    unvoiced_frames_counter = 0 # Count consecutive unvoiced frames
    recording_active = False # State: Are we currently capturing a speech segment?
    
    # Get the audio input device index
    device_index = get_audio_input_device_index()
    print("Listening for voice activity... (Press Ctrl+C to stop)")

    try:
        with sd.InputStream(samplerate=FS, channels=CHANNELS, dtype=DTYPE, device=device_index) as stream:
            audio_data_buffer = [] # Buffer to store the current speech segment

            while True:
                # Read a frame of audio
                frame_data, overflowed = stream.read(VAD_FRAME_SIZE)
                if overflowed:
                    print("Audio input buffer overflowed!")

                # Convert to bytes for VAD (webrtcvad expects bytes)
                frame_bytes = frame_data.tobytes()

                # Perform VAD on the frame
                is_speech = vad.is_speech(frame_bytes, FS)
                
                # Add current frame (raw data) to the ring buffer
                ring_buffer.append((frame_data, is_speech))

                if not recording_active:
                    # Not recording, looking for start of speech
                    if is_speech:
                        voiced_frames_counter += 1
                        unvoiced_frames_counter = 0 # Reset unvoiced if speech found
                        if voiced_frames_counter >= SPEECH_INITIATE_FRAMES:
                            recording_active = True
                            print(f"Speech detected! Starting recording for new segment.")
                            # Prepend buffered silent frames before actual speech started
                            for buffered_frame, _ in list(ring_buffer)[-voiced_frames_counter - PADDING_FRAMES : -voiced_frames_counter]:
                                audio_data_buffer.append(buffered_frame)
                            audio_data_buffer.append(frame_data) # Add current speech frame
                            
                    else:
                        voiced_frames_counter = 0 # Reset voiced if silence found
                        # Keep adding to ring buffer even if not speech
                else: # recording_active is True
                    # Recording, collecting speech frames
                    audio_data_buffer.append(frame_data)
                    if not is_speech:
                        unvoiced_frames_counter += 1
                        if unvoiced_frames_counter >= SPEECH_END_FRAMES:
                            recording_active = False
                            print(f"Silence detected. Ending recording for current segment.")
                            # Yield the full speech segment
                            yield np.concatenate(audio_data_buffer)
                            audio_data_buffer = [] # Reset buffer for next segment
                            voiced_frames_counter = 0
                            unvoiced_frames_counter = 0
                    else:
                        unvoiced_frames_counter = 0 # Reset unvoiced if speech continues

    except KeyboardInterrupt:
        print("\nStopping listening for voice activity.")
    except Exception as e:
        print(f"Error during audio stream: {e}")
    finally:
        # If recording was active when stopped, yield the last segment
        if recording_active and audio_data_buffer:
            print("\nEnding last recording segment upon stop.")
            yield np.concatenate(audio_data_buffer)

def transcribe_audio(audio_numpy_data, segment_filename):
    print(f"Transcribing audio segment: {segment_filename}...")
    temp_wav_path = os.path.join(RECORDING_DIR, segment_filename)
    wavfile.write(temp_wav_path, FS, audio_numpy_data) # Save temp file for Whisper/API

    # --- Option 1: Local Whisper ---
    # if whisper_model:
    #     try:
    #         result = whisper_model.transcribe(temp_wav_path, fp16=False)
    #         os.remove(temp_wav_path) # Clean up temp file
    #         return result["text"]
    #     except Exception as e:
    #         print(f"Whisper transcription failed: {e}")
    #         os.remove(temp_wav_path)
    #         return "[Transcription Failed]"
    # else:
    #     os.remove(temp_wav_path)
    #     return "[Whisper Model Not Loaded]"

    # --- Option 2: Cloud API (e.g., OpenAI Whisper API) ---
    # if client and os.path.exists(temp_wav_path):
    #     try:
    #         with open(temp_wav_path, "rb") as audio_file:
    #             transcript = client.audio.transcriptions.create(
    #                 model="whisper-1",
    #                 file=audio_file
    #             )
    #         os.remove(temp_wav_path)
    #         return transcript.text
    #     except Exception as e:
    #         print(f"OpenAI Whisper API transcription failed: {e}")
    #         os.remove(temp_wav_path)
    #         return "[Cloud Transcription Failed]"
    # else:
    #     if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
    #     return "[API Client Not Ready or Audio File Missing]"

    os.remove(temp_wav_path) # Clean up temp file
    return "This is a placeholder transcript for testing purposes." # Placeholder for initial testing

def summarize_text(text):
    print("Summarizing text...")
    # --- Option 1: Local LLM ---
    # (Requires a very small, quantized summarization model loaded here)
    # from transformers import pipeline
    # summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6") # Example, consider smaller for Pi
    # try:
    #     summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    #     return summary[0]['summary_text']
    # except Exception as e:
    #     print(f"Local summarization failed: {e}")
    #     return "[Local Summarization Failed]"

    # --- Option 2: Cloud API (e.g., OpenAI GPT-3.5/4) ---
    # if client:
    #     try:
    #         response = client.chat.completions.create(
    #             model="gpt-3.5-turbo",
    #             messages=[
    #                 {"role": "system", "content": "You are a concise summarizer."},
    #                 {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
    #             ]
    #         )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         print(f"OpenAI API summarization failed: {e}")
    #         return "[Cloud Summarization Failed]"
    # else:
    #     return "[API Client Not Ready]"
    
    return "This is a placeholder summary for testing purposes." # Placeholder for initial testing


def main():
    ensure_dirs()
    all_recorded_text_in_hour = []
    current_hour_start_time = time.time()

    for segment_audio_data in record_and_vad_process():
        if segment_audio_data.size == 0:
            continue # Skip empty segments

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        segment_filename_base = f"vad_segment_{timestamp}.wav"
        
        transcript_segment = transcribe_audio(segment_audio_data, segment_filename_base)
        all_recorded_text_in_hour.append(transcript_segment)

        # Save individual transcript
        with open(os.path.join(TRANSCRIPT_DIR, f"transcript_{timestamp}.txt"), "w") as f:
            f.write(transcript_segment)
        print(f"Saved transcript segment: {transcript_segment[:50]}...")

        # Check if an hour has passed for summarization
        if (time.time() - current_hour_start_time) >= 3600: # 3600 seconds = 1 hour
            full_hour_text = " ".join(all_recorded_text_in_hour)
            if full_hour_text.strip(): # Only summarize if there's text
                hourly_summary = summarize_text(full_hour_text)
                summary_timestamp = datetime.now().strftime("%Y%m%d_%H")
                with open(os.path.join(SUMMARY_DIR, f"summary_{summary_timestamp}.txt"), "w") as f:
                    f.write(hourly_summary)
                print(f"\n--- Hourly Summary ({summary_timestamp}) ---\n{hourly_summary}\n---------------------------------\n")
            else:
                print("No speech detected in the last hour to summarize.")

            all_recorded_text_in_hour = [] # Reset for next hour
            current_hour_start_time = time.time() # Reset timer

if __name__ == "__main__":
    main()