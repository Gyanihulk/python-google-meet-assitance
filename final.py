import os
import threading
import queue
import pyaudio
from pydub import AudioSegment
from datetime import datetime
import speech_recognition as sr
import openai
from dotenv import load_dotenv
from openai import OpenAI
# Load the .env file to get the OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for the OpenAI library
openai.api_key = openai_api_key

# Ensure the audio directory exists
audio_folder = "audio"
os.makedirs(audio_folder, exist_ok=True)
audio_queue = queue.Queue()
response_queue = queue.Queue()  # New queue for ChatGPT responses

client = OpenAI(
    # This is the default and can be omitted
    api_key=openai.api_key,
)

def chatgpt_response(text):
    """Send the transcribed text to the OpenAI API and get a response."""
    try:
        response = client.chat.completions.create(
                    messages=[
                {
                    "role": "user",
                    "content": text,
                }
                ],
                    model="gpt-3.5-turbo",
                    )
        # print(response)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting response from ChatGPT: {str(e)}"

def transcribe_worker():
    recognizer = sr.Recognizer()
    while True:
        audio_file_path = audio_queue.get()  # Wait until an audio file is available
        if audio_file_path is None:
            break  # None is a signal to stop the worker
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            print(f"Transcription of {os.path.basename(audio_file_path)}: {text}")
            response_queue.put(text)  # Put the transcription in the response queue
        except (sr.UnknownValueError, sr.RequestError) as e:
            print(f"Error transcribing {os.path.basename(audio_file_path)}: {str(e)}")
        finally:
            audio_queue.task_done()

def chatgpt_worker():
    """Worker to handle sending transcriptions to ChatGPT and printing responses."""
    while True:
        transcribed_text = response_queue.get()  # Get transcription from response queue
        if transcribed_text is None:
            break  # None is a signal to stop the worker

        # Send the transcription to ChatGPT and get a response
        chatgpt_response_text = chatgpt_response(transcribed_text)
        print(f"Transcription: {transcribed_text}\nChatGPT Response: {chatgpt_response_text}")
        response_queue.task_done()

def save_audio_data(sample_width, frames, rate, channels):
    """Save audio data to a WAV file with a date-time stamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file_path = os.path.join(audio_folder, f"recorded_audio_{timestamp}.wav")
    audio_segment = AudioSegment(
        data=b"".join(frames),
        sample_width=sample_width,
        frame_rate=rate,
        channels=channels
    )
    audio_segment.export(audio_file_path, format="wav")
    print(f"Audio saved to {audio_file_path}")
    return audio_file_path

def continuous_audio_capture():
    p = pyaudio.PyAudio()
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    buffer_size = 1024
    sample_width = p.get_sample_size(format)

    # Start transcription and ChatGPT response workers in separate threads
    transcription_thread = threading.Thread(target=transcribe_worker)
    transcription_thread.start()
    
    chatgpt_thread = threading.Thread(target=chatgpt_worker)
    chatgpt_thread.start()

    print("Starting continuous audio capture...")
    while True:
        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=buffer_size,
                        input_device_index=1)

        frames = []
        silent_frames = 0
        silent_threshold = -40  # dB
        silence_limit = 2  # seconds

        while True:
            data = stream.read(buffer_size)
            frames.append(data)
            audio_segment = AudioSegment(data, sample_width=sample_width, frame_rate=rate, channels=channels)
            loudness = audio_segment.dBFS

            if loudness < silent_threshold:
                silent_frames += 1
            else:
                silent_frames = 0

            if silent_frames >= rate / buffer_size * silence_limit:
                print("Silence detected, stopping recording.")
                break

        stream.stop_stream()
        stream.close()

        audio_file_path = save_audio_data(sample_width, frames, rate, channels)
        audio_queue.put(audio_file_path)  # Add file path to the queue for transcription

        print("Restarting capture for next segment...")

    # Signal the transcription and ChatGPT threads to stop and wait for them to finish
    audio_queue.put(None)
    response_queue.put(None)
    transcription_thread.join()
    chatgpt_thread.join()

def main():
    continuous_audio_capture()

if __name__ == "__main__":
    main()
