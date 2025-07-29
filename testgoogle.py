import os
from dotenv import load_dotenv
import numpy as np
import scipy.io.wavfile as wavfile

# Import Google Cloud clients
from google.cloud import speech
import google.generativeai as genai

def test_google_cloud_apis():
    """
    A standalone script to test Google Cloud Speech-to-Text and Gemini API credentials.
    """
    # Load environment variables from .env file
    load_dotenv()

    # --- Authentication Check ---
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
        print("Please set it to the full path of your JSON key file.")
        return
    
    print(f"Using credentials from: {credentials_path}")

    # --- Speech-to-Text API Test ---
    print("\n--- Testing Google Cloud Speech-to-Text API ---")
    try:
        speech_client = speech.SpeechClient()
        print("Speech-to-Text client initialized successfully.")

        # Create a dummy silent audio file for testing the API call
        sample_rate = 16000
        duration = 1  # seconds
        dummy_audio_data = np.zeros(sample_rate * duration, dtype=np.int16)
        
        # In-memory audio content
        content = dummy_audio_data.tobytes()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
        )

        print("Making a test API call to Speech-to-Text...")
        response = speech_client.recognize(config=config, audio=audio)
        
        if not response.results:
            print("Speech-to-Text API call successful (received an empty transcription for silent audio, as expected).")
        else:
            print(f"Speech-to-Text API call returned an unexpected result: {response.results}")

    except Exception as e:
        print(f"\nAn error occurred during the Speech-to-Text API test: {e}")
        print("Possible reasons:")
        print("- The Speech-to-Text API may not be enabled on your Google Cloud project.")
        print("- Your service account may not have the required 'roles/speech.recognizer' IAM role.")
        print("- There might be an issue with your internet connection or Google Cloud services.")
        return

    # --- Gemini API Test ---
    print("\n--- Testing Google Cloud Gemini API ---")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please add it to your .env file or set it as an environment variable.")
        return

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Gemini API client initialized successfully.")
        
        print("Making a test API call to Gemini (for a simple summary)...")
        prompt = "Summarize this short text: 'The quick brown fox jumps over the lazy dog.'"
        response = model.generate_content(prompt)
        
        print("\n--- Gemini Test Response ---")
        print(response.text)
        print("----------------------------\n")
        print("Gemini API call successful.")

    except Exception as e:
        print(f"\nAn error occurred during the Gemini API test: {e}")
        print("Possible reasons:")
        print("- The Generative Language API may not be enabled on your Google Cloud project.")
        print("- Your API key may be invalid or have restrictions.")
        print("- There might be an issue with your internet connection or Google AI Platform services.")
        return
        
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_google_cloud_apis()