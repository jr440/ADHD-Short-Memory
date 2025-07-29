import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please ensure you have a .env file in your project directory")
    print("with OPENAI_API_KEY='sk-proj-YOUR_ACTUAL_KEY_HERE' and that python-dotenv is installed.")
else:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        print("Making a chat completion API call (haiku about AI)...")
        chat_response = client.chat.completions.create(
            model="gpt-4o-mini", # Using the smaller, cost-effective model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "write a haiku about ai"}
            ],
            max_tokens=50 # Limit response length to save tokens/cost
        )
        print("\n--- Haiku from OpenAI ---")
        print(chat_response.choices[0].message.content)
        print("-------------------------\n")

        # You can uncomment and adapt this section later to test Whisper API
        # This requires an actual audio file to send.
        # from pathlib import Path
        # audio_file_path = Path("path/to/your/test_audio.wav") # Replace with a small WAV file path

        # if audio_file_path.exists():
        #     print("Making an audio transcription API call (conceptual)...")
        #     with open(audio_file_path, "rb") as audio_file:
        #         audio_transcript = client.audio.transcriptions.create(
        #             model="whisper-1",
        #             file=audio_file
        #         )
        #     print("\n--- Audio Transcription (Conceptual) ---")
        #     print(audio_transcript.text)
        #     print("----------------------------------------\n")
        # else:
        #     print("\nSkipping audio transcription test: No audio file found at specified path.")

    except ImportError:
        print("Error: OpenAI library not found. Please run 'pip install openai'")
    except Exception as e:
        print(f"\nAn error occurred during OpenAI API call: {e}")
        print("Possible reasons:")
        print("- Internet connection issue.")
        print("- Invalid or revoked API key.")
        print("- Insufficient credits on your OpenAI account.")
        print("- OpenAI API might be temporarily down.")