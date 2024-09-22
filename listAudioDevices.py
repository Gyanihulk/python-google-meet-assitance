import pyaudio

def list_audio_devices():
    p = pyaudio.PyAudio()
    print("Available audio devices and their capabilities:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Index: {info['index']}, Name: {info['name']}")
        print(f"  Default Sample Rate: {info['defaultSampleRate']}")
        print(f"  Max Input Channels: {info['maxInputChannels']}")
        print(f"  Max Output Channels: {info['maxOutputChannels']}")
    p.terminate()

if __name__ == "__main__":
    list_audio_devices()
