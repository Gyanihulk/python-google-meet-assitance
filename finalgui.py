import os
import threading
import queue
import pyaudio
from pydub import AudioSegment
from datetime import datetime
import speech_recognition as sr
import openai
from dotenv import load_dotenv
import tkinter as tk

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

# Tkinter setup
root = tk.Tk()
root.title("Transcription and ChatGPT Response")
root.geometry("1920x1080")

# Text box to display transcription and ChatGPT response
text_box = tk.Text(root, wrap='word', state=tk.DISABLED)
text_box.pack(padx=10, pady=10, expand=True, fill='both')

# Function to update the text on the GUI
def update_text(text):
    """Update the text box with new content."""
    separator = "\n" + "-"*50 + "\n"  # Define the separator (e.g., 50 dashes)
    text_box.config(state=tk.NORMAL)  # Enable editing to insert text
    text_box.insert(tk.END, separator)  # Insert the separator
    text_box.insert(tk.END, text + "\n\n")  # Insert the new text
    text_box.config(state=tk.DISABLED)  # Disable editing again
    text_box.see(tk.END)

client = openai

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
    """Worker to handle sending transcriptions to ChatGPT and updating the GUI."""
    while True:
        transcribed_text = response_queue.get()  # Get transcription from response queue
        if transcribed_text is None:
            break  # None is a signal to stop the worker

        # Send the transcription to ChatGPT and get a response
        chatgpt_response_text = chatgpt_response(transcribed_text)
        print(f"Transcription: {transcribed_text}\nChatGPT Response: {chatgpt_response_text}")
        
        # Update the GUI with the transcription and response
        update_text(f"Transcription: {transcribed_text}\nChatGPT Response: {chatgpt_response_text}")
        
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

    min_audio_length = 6  # Minimum audio length in seconds (2 minutes)

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
        total_frames = 0  # Keep track of the total frames recorded

        while True:
            data = stream.read(buffer_size)
            frames.append(data)
            audio_segment = AudioSegment(data, sample_width=sample_width, frame_rate=rate, channels=channels)
            loudness = audio_segment.dBFS

            # Update the total number of frames
            total_frames += buffer_size

            if loudness < silent_threshold:
                silent_frames += 1
            else:
                silent_frames = 0

            if silent_frames >= rate / buffer_size * silence_limit:
                print("Silence detected, stopping recording.")
                break

        # Calculate the duration of the recorded audio in seconds
        audio_duration = total_frames / rate

        # Save and transcribe the audio only if the duration is more than 2 minutes
        if audio_duration > min_audio_length:
            print(f"Audio length: {audio_duration / 60:.2f} minutes. Saving and transcribing...")
            stream.stop_stream()
            stream.close()

            audio_file_path = save_audio_data(sample_width, frames, rate, channels)
            audio_queue.put(audio_file_path)  # Add file path to the queue for transcription
        else:
            print(f"Audio too short ({audio_duration:.2f} seconds). Discarding.")

        print("Restarting capture for next segment...")

    # Signal the transcription and ChatGPT threads to stop and wait for them to finish
    audio_queue.put(None)
    response_queue.put(None)
    transcription_thread.join()
    chatgpt_thread.join()

def main():
    # Start GUI and audio capture in parallel
    capture_thread = threading.Thread(target=continuous_audio_capture, daemon=True)
    capture_thread.start()
    
    # Start Tkinter main loop (GUI)
    root.mainloop()

    # After closing the GUI, signal to stop the audio capture
    audio_queue.put(None)
    response_queue.put(None)


if __name__ == "__main__":
    main()
