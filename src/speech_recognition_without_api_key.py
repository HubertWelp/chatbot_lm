#! /home/student/catkin_ws/src/chatbot_lm/.venv python
import whisperx
import gc
import torch
import os
from state_manager import STTStateManager, STTState
from typing import Optional, Tuple, Dict

# to turn off tensorflow warnings (doesn't work :( )
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class WhisperXProcessor:
    """
    A class that represents a WhisperXProcessor for speech recognition.

    Args:
        device (str): The device to use for processing (default: "cuda").
        batch_size (int): The batch size for processing (default: 16).
        compute_type (str): The compute type for processing (default: "float16").
        model_dir (str): The directory to load the model from (default: "model").

    Attributes:
        device (str): The device used for processing.
        batch_size (int): The batch size used for processing.
        compute_type (str): The compute type used for processing.
        model_dir (str): The directory to load the model from.
        model: The loaded WhisperX model.
        model_a: The loaded alignment model.
        metadata: The metadata associated with the alignment model.
        diarize_model: The loaded diarization model.

    Methods:
        load_model: Loads the WhisperX model.
        transcribe: Transcribes the audio file.
        align_transcription: Aligns the transcription with the audio.
        assign_speaker_labels: Assigns speaker labels to the audio segments.
        print_speaker_text: Prints the text spoken by each speaker.
        clear_gpu_memory: Clears the GPU memory.
    """

    def __init__(
        self, device="cuda", batch_size=16, compute_type="float16", model_dir="model"
    ):
        self.state_manager = STTStateManager()
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.model_dir = model_dir
        self.model = None
        self.model_a = None
        self.metadata = None
        self.diarize_model = None

    def load_model(self, model_name="large-v3", language="de"):
        """
        Loads the WhisperX model.

        Args:
            model_name (str): The name of the model to load (default: "large-v3").
            language (str): The language of the model (default: "de").
        """
        try:
            self.state_manager.set_state(STTState.INIT)
            self.model = whisperx.load_model(
                model_name,
                self.device,
                compute_type=self.compute_type,
                download_root=self.model_dir,
                language=language,
            )
            self.state_manager.set_state(STTState.IDLE)
        except Exception as e:
            self.state_manager.set_state(STTState.ERROR, {"error": str(e)})
            raise

    def transcribe(self, audio_file) -> Tuple[Optional[Dict], Optional[torch.Tensor]]:
        """
        Transcribes the audio file.

        Args:
            audio_file (str): The path to the audio file.

        Returns:
            tuple: A tuple containing the transcription result and the loaded audio.
        """
        try:
            self.state_manager.set_state(STTState.TRANSCRIBING)
            audio = whisperx.load_audio(audio_file)
            result = self.model.transcribe(audio, batch_size=self.batch_size)
            return result, audio
        except Exception as e:
            self.state_manager.set_state(STTState.ERROR, {"error": str(e)})
            return None, None
        finally:
            # Always return to IDLE state after transcription
            self.state_manager.set_state(STTState.IDLE)

    def align_transcription(self, segments, language_code, audio):
        """
        Aligns the transcription with the audio.

        Args:
            segments (list): List of segments to align.
            language_code (str): The language code for alignment.
            audio: The loaded audio.

        Returns:
            dict: The aligned result.
        """
        try:
            self.model_a, self.metadata = whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
            aligned_result = whisperx.align(
                segments,
                self.model_a,
                self.metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            return aligned_result
        except Exception as e:
            self.state_manager.set_state(STTState.ERROR, {"error": str(e)})
            return None

    def assign_speaker_labels(self, audio, segments):
        """
        Assigns speaker labels to the audio segments.

        Args:
            audio: The loaded audio.
            segments (list): List of segments to assign speaker labels to.

        Returns:
            tuple: A tuple containing the result and the diarized segments.
        """
        try:
            self.diarize_model = whisperx.DiarizationPipeline(
                use_auth_token="replace with API-Key",
                device=self.device,
                model_name="pyannote/speaker-diarization-3.1",
            )
            diarize_segments = self.diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, segments)
            return result, diarize_segments
        except Exception as e:
            self.state_manager.set_state(STTState.ERROR, {"error": str(e)})
            return None, None

    def print_speaker_text(self, segments):
        """
        Print the text spoken by each speaker.

        Args:
            segments (list): List of segments containing speaker information and text.

        Returns:
            dict: Dictionary mapping speaker IDs to their spoken text.
        """
        # Create a dictionary to store the combined text spoken by each speaker
        speakers = {}
        # Iterate over each segment in the list of segments
        for segment in segments:
            # Get the speaker ID and text from the segment
            speaker_id = segment["speaker"]
            text = segment["text"]

            # If the speaker ID is not already in the dictionary, add it as a key with the text as the value
            if speaker_id not in speakers:
                speakers[speaker_id] = text.strip()
            else:
                # If the speaker ID is already in the dictionary, concatenate the text to the existing text for the corresponding speaker ID
                speakers[speaker_id] += " " + text.strip()

        return speakers

    def clear_gpu_memory(self):
        """
        Clears the GPU memory by unloading the model.
        """
        gc.collect()
        torch.cuda.empty_cache()
        if self.model:
            del self.model
        if self.model_a:
            del self.model_a


def main():
    processor = WhisperXProcessor()

    try:
        processor.load_model()

        audio_file = "/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/reference.wav"
        transcription_result, audio = processor.transcribe(audio_file)
        print(transcription_result["segments"])

        aligned_result = processor.align_transcription(
            transcription_result["segments"], transcription_result["language"], audio
        )
        print(aligned_result["segments"])

        final_result, diarize_segments = processor.assign_speaker_labels(
            audio, aligned_result
        )
        print(diarize_segments)
        print(final_result["segments"])

        # Print text spoken by each speaker
        processor.print_speaker_text(final_result["segments"])

    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        processor.clear_gpu_memory()


if __name__ == "__main__":
    main()
