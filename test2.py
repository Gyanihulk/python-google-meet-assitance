import os
import pyaudio
from pydub import AudioSegment
from datetime import datetime
import speech_recognition as sr

# Ensure the audio directory exists
audio_folder = "audio"
os.makedirs(audio_folder, exist_ok=True)

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

def transcribe_audio(audio_file_path):
    """Transcribe the audio to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)  # read the entire audio file

    try:
        text = recognizer.recognize_google(audio)
        print(f"Transcription: {text}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def continuous_audio_capture():
    p = pyaudio.PyAudio()
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    buffer_size = 1024
    sample_width = p.get_sample_size(format)

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
        transcribe_audio(audio_file_path)

        print("Restarting capture for next segment...")

def main():
    continuous_audio_capture()

if __name__ == "__main__":
    main()
