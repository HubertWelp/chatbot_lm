from pathlib import Path
import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from typing import Optional, List, Tuple, Generator, Union, Literal, Annotated
import logging
import shutil
import torch
from dataclasses import dataclass
import sounddevice as sd
import wave
from state_manager import TTSStateManager, TTSState
from pydantic import BaseModel, Field, conint


class ServeReferenceAudio(BaseModel):
    """Reference audio for voice cloning"""

    audio: bytes  # Raw audio data
    text: str

    @classmethod
    def from_file(cls, audio_path: str, text: str) -> "ServeReferenceAudio":
        """Create a ServeReferenceAudio from an audio file path"""
        try:
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            return cls(audio=audio_data, text=text)
        except Exception as e:
            raise ValueError(f"Failed to read audio file {audio_path}: {e}")


class ServeTTSRequest(BaseModel):
    """TTS request parameters matching fish-speech implementation"""

    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    format: Literal["wav", "pcm", "mp3"] = "wav"
    references: List[ServeReferenceAudio] = []
    reference_id: Optional[str] = None
    seed: Optional[int] = None
    use_memory_cache: Literal["on", "off"] = "off"
    normalize: bool = True
    streaming: bool = False
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.2
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_params(
        cls,
        text: str,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> "ServeTTSRequest":
        """
        Create a ServeTTSRequest from parameters.

        Args:
            text: Text to synthesize
            reference_audio: Optional path to reference audio file
            reference_text: Optional reference text
            **kwargs: Additional parameters
        """
        references = []
        if reference_audio and reference_text:
            ref = ServeReferenceAudio.from_file(
                audio_path=reference_audio, text=reference_text
            )
            references = [ref]

        return cls(text=text, references=references, seed=seed, **kwargs)


@dataclass
class AudioResult:
    """Holds the result of audio generation"""

    audio_data: Union[
        Tuple[int, np.ndarray], np.ndarray
    ]  # (sample_rate, data) or just data
    sample_rate: int
    file_path: Optional[str] = None


class FishSpeechTTS:
    """
    Text to Speech generator using Fish-Speech 1.5.
    Provides functionality to generate speech from text with optional voice reference.
    """

    def __init__(
        self,
        base_dir: str = "/home/student/catkin_ws/src/chatbot_lm",
        checkpoint_dir: str = "checkpoints/fish-speech-1.5",
        use_compile: bool = True,
        use_half: bool = False,
        device: str = "cuda",
        default_sample_rate: int = 22050,
    ):
        """
        Initialize the Fish-Speech TTS system.

        Args:
            base_dir (str): Base directory containing the fish-speech installation
            checkpoint_dir (str): Directory containing the model checkpoints
            use_compile (bool): Whether to use CUDA kernel fusion for faster inference
            use_half (bool): Whether to use half precision (for GPUs that don't support bf16)
            device (str): Device to run inference on ("cuda", "cpu", or "mps")
        """
        self.state_manager = TTSStateManager()
        self.state_manager.set_state(TTSState.INITIALIZING)
        self.base_dir = Path(base_dir)
        self.checkpoint_dir = self.base_dir / checkpoint_dir
        self.tools_dir = self.base_dir / "fish-speech"
        self.use_compile = use_compile
        self.use_half = use_half
        self.logger = logging.getLogger(__name__)
        self.device = self._determine_device(device)
        self.precision = torch.half if use_half else torch.bfloat16
        self.default_sample_rate = default_sample_rate

        try:
            # Validate and setup directories
            self._setup_environment()

            # Initialize inference engine components
            self.llama_queue = None
            self.decoder_model = None
            self.inference_engine = None

            # Import required modules dynamically after repo clone
            self._setup_inference_engine()

            # If all initialization succeeds, set state to IDLE
            self.state_manager.set_state(TTSState.IDLE)
        except Exception as e:
            self.state_manager.set_state(TTSState.ERROR, {"error": str(e)})
            raise

    def _setup_environment(self):
        """Set up the fish-speech environment and validate required files."""
        # Create directories if they don't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tools_dir, exist_ok=True)

        # Clone fish-speech repository if not present
        if not (self.tools_dir / "tools").exists():
            self._clone_fish_speech()

        # Validate checkpoints
        if not self._validate_checkpoints():
            self._download_models()

    def _clone_fish_speech(self):
        """Clone the fish-speech repository."""
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/fishaudio/fish-speech.git",
                    str(self.tools_dir),
                ],
                check=True,
            )
            self.logger.info("Successfully cloned fish-speech repository")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to clone fish-speech repository: {e}")
            raise

    def _validate_checkpoints(self) -> bool:
        """
        Validate that required model files exist in checkpoint directory.

        Returns:
            bool: True if all required files exist, False otherwise
        """
        required_files = ["firefly-gan-vq-fsq-8x1024-21hz-generator.pth", "llama.pth"]
        return all((self.checkpoint_dir / file).exists() for file in required_files)

    def _download_models(self):
        """Download the required model files using huggingface-cli."""
        try:
            subprocess.run(
                [
                    "huggingface-cli",
                    "download",
                    "fishaudio/fish-speech-1.5",
                    "--local-dir",
                    str(self.checkpoint_dir),
                ],
                check=True,
            )
            self.logger.info("Successfully downloaded model files")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to download models: {e}")
            raise

    def _determine_device(self, requested_device: str) -> str:
        """Determine the appropriate device for inference."""
        if torch.backends.mps.is_available() and requested_device == "mps":
            return "mps"
        if torch.cuda.is_available() and requested_device == "cuda":
            return "cuda"
        return "cpu"

    def _setup_inference_engine(self):
        """Set up the inference engine components."""
        try:
            self.state_manager.set_state(TTSState.LOADING_MODEL)

            import sys

            tools_path = str(self.tools_dir)
            if tools_path not in sys.path:
                sys.path.append(tools_path)

            from tools.llama.generate import launch_thread_safe_queue
            from tools.vqgan.inference import load_model as load_decoder_model
            from tools.inference_engine import TTSInferenceEngine

            self.logger.info("Loading Llama model...")
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=self.checkpoint_dir,
                device=self.device,
                precision=self.precision,
                compile=self.use_compile,
            )

            self.logger.info("Loading VQ-GAN model...")
            self.decoder_model = load_decoder_model(
                config_name="firefly_gan_vq",
                checkpoint_path=self.checkpoint_dir
                / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
                device=self.device,
            )

            self.logger.info("Initializing inference engine...")
            self.inference_engine = TTSInferenceEngine(
                llama_queue=self.llama_queue,
                decoder_model=self.decoder_model,
                compile=self.use_compile,
                precision=self.precision,
            )

            # Warm up the model
            # self._warmup_model()
        except Exception as e:
            self.state_manager.set_state(TTSState.ERROR, {"error": str(e)})
            raise

    def _warmup_model(self):
        """Perform a warmup inference to avoid first-time latency."""
        try:
            self.logger.info("Warming up the model...")
            list(
                self.inference_engine.inference(
                    ServeTTSRequest(
                        text="Hallo! Ich bin SweetPicker",
                        references=[],
                    )
                )
            )
            self.logger.info("Model warmup complete")
        except Exception as e:
            self.state_manager.set_state(TTSState.ERROR, {"error": str(e)})
            raise

    def generate_speech(
        self,
        text: str,
        reference_audio: Optional[str] = None,
        reference_text: Optional[str] = None,
        output_format: Literal["numpy", "wav"] = "numpy",
        output_file: Optional[str] = None,
        **kwargs,
    ) -> AudioResult:
        """
        Generate speech from text with flexible output format.

        Args:
            text: Text to convert to speech
            reference_audio: Optional path to reference audio file for voice style
            reference_text: Text corresponding to reference audio
            output_format: Desired output format ("numpy" or "wav")
            output_file: Path to save WAV file if output_format is "wav"
            **kwargs: Additional parameters for ServeTTSRequest

        Returns:
            AudioResult containing the generated audio data and metadata
        """
        try:
            self.state_manager.set_state(TTSState.GENERATING)

            request = ServeTTSRequest.from_params(
                text=text,
                reference_audio=reference_audio,
                reference_text=reference_text,
                **kwargs,
            )

            # Get inference results
            results = list(self.inference_engine.inference(request))

            # Extract audio data from results
            if not results or not hasattr(results[0], "audio"):
                raise ValueError("No audio data in inference results")

            sample_rate, audio_data = results[0].audio

            # Convert numpy array to the desired format
            if output_format == "wav":
                if output_file is None:
                    output_file = tempfile.mktemp(suffix=".wav")

                audio_data = np.clip(audio_data, -1.0, 1.0)
                int16_data = (audio_data * 32767).astype(np.int16)

                with wave.open(output_file, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(int16_data.tobytes())

                result = AudioResult(
                    audio_data=int16_data,
                    sample_rate=sample_rate,
                    file_path=output_file,
                )
            else:
                result = AudioResult(
                    audio_data=audio_data, sample_rate=sample_rate, file_path=None
                )

            self.state_manager.set_state(TTSState.IDLE)
            return result

        except Exception as e:
            self.state_manager.set_state(TTSState.ERROR, {"error": str(e)})
            raise

    def play_audio(self, audio_result: AudioResult):
        """
        Play the generated audio.

        Args:
            audio_result: AudioResult from generate_speech
        """
        try:
            self.state_manager.set_state(TTSState.PLAYING)

            if audio_result.file_path:
                # If we have a file path, read and play the WAV file
                with wave.open(audio_result.file_path, "rb") as wf:
                    audio_data = wf.readframes(wf.getnframes())
                    sample_width = wf.getsampwidth()

                    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
                    audio_array = np.frombuffer(audio_data, dtype=dtype)

                    if dtype == np.int16:
                        audio_array = audio_array.astype(np.float32) / 32767.0

                    sd.play(audio_array, audio_result.sample_rate, blocking=True)
            else:
                audio_data = audio_result.audio_data
                if isinstance(audio_data, tuple):
                    audio_data = audio_data[1]

                sd.play(audio_data, audio_result.sample_rate, blocking=True)

            self.state_manager.set_state(TTSState.IDLE)

        except Exception as e:
            self.state_manager.set_state(TTSState.ERROR, {"error": str(e)})
            raise

    def save_wav(self, file_path, audio_data, sample_rate=22050):
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)

    def cleanup(self):
        """Clean up resources and temporary files"""
        try:
            if self.llama_queue:
                self.llama_queue.put(None)

            temp_files = ["fake.npy", *[f"codes_{i}.npy" for i in range(1000)]]
            for file in temp_files:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary file {file}: {e}")

            self.state_manager.set_state(TTSState.IDLE)
        except Exception as e:
            self.state_manager.set_state(TTSState.ERROR, {"error": str(e)})
            raise


def main():
    """Example usage of FishSpeechTTS."""
    logging.basicConfig(level=logging.INFO)

    tts = FishSpeechTTS(
        base_dir="/home/student/catkin_ws/src/chatbot_lm",
        checkpoint_dir="checkpoints/fish-speech-1.5",
    )
    try:
        # Example with reference voice
        output_file = tts.generate_speech(
            text="Ich habe dir bisher keine Süßigkeiten gegeben. Möchtest du gerne eine Maoam, einen Snickers, einen Kinderriegel oder eine Milkyway haben?",
            seed=5,
            # reference_audio="/home/student/catkin_ws/src/chatbot_lm/examples_German.wav",
            # reference_text="Ich habe gehört, dass es auch eine API geöffnet hat und dass es schnell in jedes Produkt integriert werden kann.",
            # reference_audio="/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/reference.wav",
            # reference_text="Die bittere Wahrheit ist nun, bis ein einheitlicher Schutz der Lebensadern unserer Demokratie in einer neuen Legislaturperiode auf den Weg gebracht wird, werden viele Monate ins Land gehen, sagten von Nords und Kahn. Das ist Zeit, die wir eigentlich nicht mehr haben.",
            temperature=0.7,
            top_p=0.7,
            repetition_penalty=1.2,
        )
        print(f"Generated audio file: {output_file}")
    except Exception as e:
        print(f"Error during speech synthesis: {e}")
    finally:
        tts.cleanup()


if __name__ == "__main__":
    main()
