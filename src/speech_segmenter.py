import torch
import pyaudio
import numpy as np
import wave
from colorama import Fore, Back, Style
from state_manager import STTStateManager, STTState

# For normalization of 16-bit PCM audio into [-1, 1], PCM range is [-32768, 32767]
PCM_MAX_INT16 = 32768  # 2^15 bits
# Set the sampling rate for the audio to 16kHz (the default for the Silero VAD model)
SAMPLE_RATE = 16000  # 16kHz
# the amount of seconds of detected silence that will be attached to the begining and end of every speech chunk to avoid cutting off low power speech parts
SILENCE_BUFFER_DURATION = 0.2  # 200ms


# Audio processing constants
TARGET_LOUDNESS = 0.3  # Target RMS level (between 0 and 1)
ATTACK_TIME = 0.010  # Attack time in seconds
RELEASE_TIME = 0.100  # Release time in seconds
THRESHOLD = 0.15  # Compression threshold
RATIO = 4.0  # Compression ratio
MAKEUP_GAIN = 1.2  # Gain after compression


class DynamicProcessor:
    """Audio dynamics processor for real-time compression and normalization"""

    def __init__(self, sample_rate, attack_time=ATTACK_TIME, release_time=RELEASE_TIME):
        self.sample_rate = sample_rate
        self.attack = np.exp(-1.0 / (sample_rate * attack_time))
        self.release = np.exp(-1.0 / (sample_rate * release_time))
        self.envelope = 0
        self.gain = 1.0

    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply dynamic processing to audio data"""
        # Compute envelope
        envelope = np.abs(audio_data)
        self.envelope = np.maximum(
            envelope, self.attack * self.envelope + (1 - self.attack) * envelope
        )

        # Apply compression
        gain_reduction = np.ones_like(audio_data)
        mask = self.envelope > THRESHOLD
        gain_reduction[mask] = np.power(THRESHOLD / self.envelope[mask], 1 - 1 / RATIO)

        # Smooth gain changes
        if self.gain < gain_reduction[0]:
            self.gain = self.attack * self.gain + (1 - self.attack) * gain_reduction[0]
        else:
            self.gain = (
                self.release * self.gain + (1 - self.release) * gain_reduction[0]
            )

        # Apply makeup gain
        processed = audio_data * self.gain * MAKEUP_GAIN

        # Apply normalization to target loudness
        current_rms = np.sqrt(np.mean(processed**2))
        if current_rms > 0:
            gain_factor = TARGET_LOUDNESS / current_rms
            processed = np.clip(processed * gain_factor, -1.0, 1.0)

        return processed


class SpeechSilenceDetector:
    """
    Class that detects speech and silence parts in an audio file.

    Args:
        audio_file (str): Path to the audio file.

    Attributes:
        model: The VAD model loaded from the snakers4/silero-vad repository.
        utils: The necessary functions extracted from the utils module.
        SAMPLE_RATE: The sampling rate for the audio.
        speech_parts: A list of tuples representing the start and end times of speech parts in the audio.
        silence_parts: A list of tuples representing the start and end times of silence parts in the audio.

    Methods:
        get_parts_of_speech_and_silence: Returns the speech and silence parts of the audio.

    """

    def __init__(self, audio_file):
        # Set the number of threads for Torch
        # silero vad is supposed to run with 1 thread according to their github page
        torch.set_num_threads(8)

        # Load the VAD model from the snakers4/silero-vad repository
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )  # type: ignore

        # Extract necessary functions from the utils module
        (self.get_speech_timestamps, _, self.read_audio, *_) = self.utils

        # Set the sampling rate for the audio
        self.SAMPLE_RATE = SAMPLE_RATE

        # Read the audio file
        wav = self.read_audio(audio_file, sampling_rate=self.SAMPLE_RATE)

        # Get speech timestamps from the full audio file
        speech_timestamps = self.get_speech_timestamps(
            wav, self.model, sampling_rate=self.SAMPLE_RATE
        )

        prev_end = 0
        self.speech_parts = []
        self.silence_parts = []

        for timestamp in speech_timestamps:
            start = timestamp["start"] / self.SAMPLE_RATE
            end = timestamp["end"] / self.SAMPLE_RATE

            # Add silence before speech if there is any
            if start > prev_end:
                self.silence_parts.append((prev_end, start))

            self.speech_parts.append((start, end))
            prev_end = end

        # Add silence after the last speech segment if there is any
        if prev_end < len(wav) / self.SAMPLE_RATE:
            self.silence_parts.append((prev_end, len(wav) / self.SAMPLE_RATE))

    def get_parts_of_speech_and_silence(self):
        """
        Returns the speech and silence parts of the audio.

        Returns:
            tuple: A tuple containing two lists. The first list contains tuples representing the start and end times of speech parts.
                   The second list contains tuples representing the start and end times of silence parts.

        """
        return self.speech_parts, self.silence_parts


# STT best works for 5-15s audio chunks anyway. - snakers4
# I'll trust him :D
class MicrophoneSpeechDetector:
    """
    Class for detecting speech from a microphone stream.

    Args:
        silence_threshold (float): The duration of silence (in seconds) that indicates the end of speech. Default is 1 second.

    Attributes:
        model: The VAD model loaded from the snakers4/silero-vad repository.
        utils: The utility functions extracted from the utils module of the VAD model.
        SAMPLE_RATE (int): The sampling rate for the audio. Default is 16000 samples per second.
        CHUNK_SIZE (int): The chunk size for microphone input. Default is 500ms.
        audio_stream: The initialized microphone stream.
        speech_buffer: The buffer to store speech data.
        is_silence (bool): Flag to indicate if silence is detected.
        silence_threshold (float): The duration of silence (in seconds) that indicates the end of speech.

    Methods:
        initialize_microphone_stream: Initializes the microphone stream.
        get_speech: Reads audio data from the microphone stream and detects speech.
        close_microphone_stream: Closes the microphone stream.

    """

    def __init__(self, silence_threshold=1):
        self.state_manager = STTStateManager()
        # Set the number of threads for Torch
        # silero vad is supposed to run with 1 thread according to their github page
        torch.set_num_threads(8)
        # Load the VAD model from the snakers4/silero-vad repository
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )  # type: ignore
        # Extract necessary functions from the utils module
        (self.get_speech_timestamps, _, _, *_) = self.utils
        # Set the sampling rate for the audio
        self.SAMPLE_RATE = 16000  # Samples per second
        # Set the chunk size for microphone input
        self.CHUNK_SIZE = int(self.SAMPLE_RATE / 2)  # 500ms
        # Initialize the microphone stream
        self.is_stream_open = False
        self.audio_stream = self.initialize_microphone_stream()
        # Initialize the speech buffer
        self.speech_buffer = []
        # Initialize the silence flag
        self.is_silence = True
        # Set the silence threshold
        self.silence_threshold = silence_threshold

        # Initialize dynamic processor
        self.dynamics = DynamicProcessor(self.SAMPLE_RATE)

        # Check if the silence threshold is a multiple of the chunk size
        if self.silence_threshold % (self.CHUNK_SIZE / self.SAMPLE_RATE) == 0:
            # Increase the silence threshold slightly (seems to perform better this way in regards to capturing the end of speech chunks)
            self.silence_threshold += 0.2  # +200ms

    def initialize_microphone_stream(self):
        """
        Initializes the microphone stream.

        Returns:
            stream: The initialized microphone stream.

        """
        if not self.is_stream_open:
            self.is_stream_open = True
            self.audio = pyaudio.PyAudio()
            stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE,
            )
            print(f"{Style.DIM}Audio stream started.{Style.RESET_ALL}")
            return stream
        else:
            return self.audio_stream

    def process_audio_chunk(self, audio_data: np.ndarray) -> np.ndarray:
        """Process raw audio data with dynamics processing"""
        # Convert to float32 and normalize
        float_data = audio_data.astype(np.float32) / PCM_MAX_INT16
        # Apply dynamic processing
        processed = self.dynamics.process(float_data)
        return processed

    def get_speech(self):
        """
        Reads audio data from the microphone stream and detects speech.

        Returns:
            speech_chunk (numpy.ndarray): The detected speech chunk as a numpy array.

        """
        try:
            self.state_manager.set_state(STTState.LISTENING)
            # Reset buffer and reinitialize stream to flush pending audio
            self.speech_buffer = []
            if self.is_stream_open:
                print(
                    f"{Style.DIM}Restarting Audio Stream to flush buffer{Style.RESET_ALL}"
                )
                self.close_microphone_stream()
            self.audio_stream = self.initialize_microphone_stream()
            print(f"{Style.DIM}Ready for Speech{Style.RESET_ALL}")

            silence_duration = 0
            self.is_silence = True  # Reset silence flag

            while True:
                # Read audio data from the microphone stream
                audio_data = np.frombuffer(
                    self.audio_stream.read(self.CHUNK_SIZE), dtype=np.int16
                )

                # Process audio with dynamics processing
                processed_audio = self.process_audio_chunk(audio_data)

                # Convert audio data to float32 and normalize (using paInt16 in stream so normalize for INT16)
                audio_data = audio_data.astype(np.float32) / PCM_MAX_INT16
                # Get speech timestamps from the audio data
                speech_timestamps = self.get_speech_timestamps(
                    processed_audio, self.model, sampling_rate=self.SAMPLE_RATE
                )
                # Check if speech is detected
                if len(speech_timestamps) > 0:
                    # Append audio data to the speech buffer
                    # print(f"{Style.DIM}Speech detected{Style.RESET_ALL}")
                    self.speech_buffer.extend(audio_data)
                    # Set silence flag to False
                    self.is_silence = False
                    # Reset silence duration to allow for the full silence threshold
                    silence_duration = 0
                else:
                    # Check if silence flag is False
                    if not self.is_silence:
                        if silence_duration == 0:
                            # Add the first SILENCE_BUFFER_DURATION seconds to the buffer to avoid cutting off the last part of the speech
                            self.speech_buffer.extend(
                                audio_data[
                                    : int(SILENCE_BUFFER_DURATION * self.SAMPLE_RATE)
                                ]
                            )
                        # convert silence_duration to seconds
                        silence_duration += len(audio_data) / self.SAMPLE_RATE
                        # Print silence duration
                        # print(f"Silence in Speech for: {silence_duration}s")
                        # Check if silence duration is greater than the silence threshold
                        # Silence is not saved to the buffer so that a pause in speech is not included in the speech chunk
                        if silence_duration >= self.silence_threshold:
                            print(f"{Style.DIM}Speech ended{Style.RESET_ALL}")
                            # Add the first SILENCE_BUFFER_DURATION seconds to the buffer to avoid cutting off the last part of the speech
                            self.speech_buffer.extend(
                                audio_data[
                                    : int(SILENCE_BUFFER_DURATION * self.SAMPLE_RATE)
                                ]
                            )
                            # Convert speech buffer to numpy array
                            speech_chunk = np.array(self.speech_buffer)

                            # DEBUG: Uncomment the following code to save the speech chunk as a .wav file
                            # # Save the speech chunk as a .wav file
                            # wav_file = "speech_chunk.wav"
                            # with wave.open(wav_file, "wb") as wf:
                            #     wf.setnchannels(1)  # mono
                            #     wf.setsampwidth(2)  # 2 bytes (16 bits) per sample
                            #     wf.setframerate(self.SAMPLE_RATE)
                            #     # Convert the speech chunk to int16
                            #     speech_chunk = (speech_chunk * PCM_MAX_INT16).astype(
                            #         np.int16
                            #     )
                            #     # Write the speech chunk to the .wav file
                            #     wf.writeframes(speech_chunk.tobytes())
                            # Reset speech buffer
                            self.speech_buffer = []
                            # Set silence flag to True
                            self.is_silence = True
                            self.close_microphone_stream()
                            # Return the speech chunk
                            return speech_chunk
                    else:
                        # print(f"{Style.DIM}Silence detected{Style.RESET_ALL}")
                        # Save the last SILENCE_BUFFER_DURATION seconds of audio data so that the moment we detect speech we still have SILENCE_BUFFER_DURATION seconds of history
                        # minus symbol results in negative indexing, meaning we take the last SILENCE_BUFFER_DURATION seconds of audio data and not the first SILENCE_BUFFER_DURATION seconds
                        self.speech_buffer = list(
                            audio_data[
                                -int(SILENCE_BUFFER_DURATION * self.SAMPLE_RATE) :
                            ]
                        )
        except Exception as e:
            self.state_manager.set_state(STTState.ERROR, {"error": str(e)})
            raise

    def close_microphone_stream(self):
        """
        Closes the microphone stream.

        """
        self.is_stream_open = False
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio.terminate()
        print(f"{Style.DIM}Audio stream stopped.{Style.RESET_ALL}")


def main():
    # Initialize the MicrophoneSpeechDetector
    detector = MicrophoneSpeechDetector()

    try:
        while True:
            # Detect speech from microphone input
            speech_chunk = detector.get_speech()

            # Print the speech chunk
            print(speech_chunk)
    except KeyboardInterrupt:
        # Close the microphone stream
        detector.close_microphone_stream()


if __name__ == "__main__":
    main()
