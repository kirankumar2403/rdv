
import whisper
import sounddevice as sd
import numpy as np
import queue
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
from playsound import playsound
import threading
import time

# Initialize models
asr_model = whisper.load_model("base")
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
tts = TTS("tts_models/en/ljspeech/glow-tts")

# Global flags and queues
audio_queue = queue.Queue()
text_queue = queue.Queue()
translation_active = False

def audio_callback(indata, frames, time, status):
    if translation_active:
        audio_queue.put(indata.copy())

def start_translation_process(url, title):
    global translation_active
    translation_active = True
    print(f"Starting translation for {title} - {url}")

    threading.Thread(target=capture_and_transcribe).start()
    threading.Thread(target=translate_and_speak).start()

def capture_and_transcribe():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        while translation_active:
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                result = asr_model.transcribe(np.array(audio_data))
                if result["text"].strip():
                    text_queue.put(result["text"])

def translate_and_speak():
    while translation_active:
        if not text_queue.empty():
            original_text = text_queue.get()
            # Translate the captured text
            inputs = translation_tokenizer(original_text, return_tensors="pt", padding=True)
            translated = translation_model.generate(**inputs)
            translated_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
            print(f"Translated: {translated_text}")

            # Generate speech from the translated text
            tts.tts_to_file(text=translated_text, file_path="output.wav")
            playsound("output.wav")  # Play the translated speech

def stop_translation_process():
    global translation_active
    translation_active = False
    print("Translation process stopped.")
