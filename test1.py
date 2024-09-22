import os
import pyaudio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from datetime import datetime
import speech_recognition as sr


# Ensure the audio directory exists
audio_folder = "audio"
os.makedirs(audio_folder, exist_ok=True)

def save_audio_data(sample_width, frames, rate, channels):
    """Save audio data to a WAV file with a date-time stamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file_path = os.path.join(audio_folder, f"recorded_audio_{timestamp}.wav")
    
    # Convert the raw audio data to an audio segment
    audio_segment = AudioSegment(
        data=b"".join(frames),
        sample_width=sample_width,
        frame_rate=rate,
        channels=channels
    )
    
    # Export the audio segment to a WAV file
    audio_segment.export(audio_file_path, format="wav")
    print(f"Audio saved to {audio_file_path}")
    return audio_file_path

def transcribe_audio(audio_file_path):
    """Transcribe the audio to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)  # read the entire audio file

    try:
        # Using Google Web Speech API to transcribe the audio
        text = recognizer.recognize_google(audio)
        print(f"Transcription: {text}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")


def get_system_audio_text():
    p = pyaudio.PyAudio()
    stream = None
    try:
        # Define the audio stream format
        format = pyaudio.paInt16
        channels = 1  # Mono audio
        rate = 44100
        buffer_size = 1024
        
        sample_width = p.get_sample_size(format)

        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=buffer_size,
                        input_device_index=1)  # Correct index

        print("Listening to system audio...")
        frames = []
        silent_frames = 0
        silent_threshold = -40  # dB, threshold to consider silence
        silence_limit = 2  # seconds of silence to stop recording

        while True:
            data = stream.read(buffer_size)
            frames.append(data)
            audio_segment = AudioSegment(data, sample_width=sample_width, frame_rate=rate, channels=channels)
            loudness = audio_segment.dBFS

            if loudness < silent_threshold:
                silent_frames += 1
            else:
                silent_frames = 0  # Reset silence counter if noise detected
            
            if silent_frames >= rate / buffer_size * silence_limit:
                print("Silence detected, stopping recording.")
                break

        audio_file_path = save_audio_data(sample_width, frames, rate, channels)
        transcribe_audio(audio_file_path)
        

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()

def main():
    get_system_audio_text()

if __name__ == "__main__":
    main()
