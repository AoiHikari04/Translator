import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import threading
import queue

# Audio configuration
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono audio
RATE = 16000  # Sample rate (Whisper expects 16kHz)
CHUNK = 1024  # Number of frames per buffer
RECORD_SECONDS = 5  # Duration of each transcription chunk (can be adjusted)

# Whisper model configuration
model_size = "small"  # You can choose "tiny", "base", "small", "medium", "large"
model = WhisperModel(model_size, device="cuda", compute_type="float16")  # Use GPU

# Queue to hold audio chunks for transcription
audio_queue = queue.Queue()

# Function to capture audio from the microphone
def capture_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    print("Recording... Press Ctrl+C to stop.")

    try:
        while True:
            # Read audio data from the stream
            data = stream.read(CHUNK)
            # Convert raw audio data to numpy array
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            # Put the audio data into the queue
            audio_queue.put(audio_data)
    except KeyboardInterrupt:
        print("Stopping recording.")
    finally:
        # Clean up
        stream.stop_stream()
        stream.close()
        audio.terminate()

# Function to transcribe audio chunks
def transcribe_audio():
    while True:
        # Get audio data from the queue
        audio_data = audio_queue.get()
        if audio_data is None:  # Sentinel value to stop the thread
            break

        # Transcribe the audio chunk
        segments, info = model.transcribe(audio_data, beam_size=5)

        # Print the transcription
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

# Main function
def main():
    # Start the audio capture thread
    capture_thread = threading.Thread(target=capture_audio)
    capture_thread.start()

    # Start the transcription thread
    transcribe_thread = threading.Thread(target=transcribe_audio)
    transcribe_thread.start()

    # Wait for threads to finish (Ctrl+C to stop)
    capture_thread.join()
    transcribe_thread.join()

if __name__ == "__main__":
    main()