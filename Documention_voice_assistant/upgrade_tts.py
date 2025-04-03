"""
TTS Model Upgrade Script

This script helps you download better TTS models for clearer voice output.
Run this script once to download the models before using the main app.
"""

import os
import sys
from TTS.api import TTS


def download_better_models():
    print("Downloading better TTS models for clearer voice output...")

    try:
        # First try to download the high-quality XTTS model
        print(
            "\nAttempting to download XTTS v2 (best quality, requires more resources)..."
        )
        try:
            tts_xtts = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=True,
            )
            print("✓ Successfully downloaded XTTS v2!")
        except Exception as e:
            print(f"× Could not download XTTS v2: {str(e)}")
            print("Don't worry, we'll try other models.")

        # Download the VITS model (more compatible)
        print("\nDownloading VITS model (good quality, more compatible)...")
        tts_vits = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True)
        print("✓ Successfully downloaded VITS model!")

        # List and print available speakers for VITS
        print("\nAvailable VITS speakers:")
        for i, speaker in enumerate(tts_vits.speakers):
            if i < 10 or i % 5 == 0:  # Print first 10 and then every 5th speaker
                print(f"- {speaker}")
        print(f"Total speakers available: {len(tts_vits.speakers)}")

        # Try to download a CoquiTTS model for highest quality
        print(
            "\nAttempting to download YourTTS model (high quality with voice cloning)..."
        )
        try:
            tts_your = TTS(
                model_name="tts_models/multilingual/multi-dataset/your_tts",
                progress_bar=True,
            )
            print("✓ Successfully downloaded YourTTS model!")
        except Exception as e:
            print(f"× Could not download YourTTS model: {str(e)}")

        print("\nAll available models downloaded successfully!")
        print(
            "\nTo use these models in the app, make sure to run the app with 'streamlit run app.py'"
        )
        return True

    except Exception as e:
        print(f"\nERROR downloading models: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have TTS installed properly: pip install TTS==0.22.0")
        print("2. Check your internet connection")
        print("3. Try running the script with administrator privileges")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("TTS Model Downloader for Documentation Voice Assistant")
    print("=" * 60)
    download_better_models()
    print("\nPress any key to exit...")
    input()
