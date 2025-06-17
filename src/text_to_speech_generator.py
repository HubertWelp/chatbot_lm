import os
from urllib.error import URLError
import torch
import torchaudio
import tempfile
import sounddevice as sd
from scipy.io.wavfile import read as read_wav
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.api import TTS
from nltk import downloader
from typing import List, Optional, Tuple
import numpy as np
from state_manager import TTSStateManager, TTSState


# Download the punkt tokenizer for sentence tokenization
try:
    if not downloader.Downloader().is_installed("punkt"):
        downloader.download("punkt")
except URLError:
    print(
        "can't reach nltk servers. assuming offline use and setup has been done once with an internet connection to download all files and models."
    )

from nltk.tokenize import sent_tokenize


class TTSHandler:
    def __init__(self):
        # Initialize the state manager
        self.state_manager = TTSStateManager()
        self.state_manager.set_state(TTSState.INITIALIZING)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.xtts_model = None
        self.xtts_checkpoint = None
        self.xtts_config = None
        self.xtts_vocab = None
        self.max_batch_length = 253  # XTTS limitation
        self.sample_rate = 24000
        os.environ["TRAINER_TELEMETRY"] = "0"

        # Enable CUDA optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

    def clear_inference_memory(self):
        """Clear GPU memory after inference"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def optimize_model(self):
        """Apply performance optimizations to the model"""
        if self.xtts_model is not None:
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.xtts_model = self.xtts_model.to(self.device)

            # Enable eval mode for inference
            self.xtts_model.eval()

    def load_model(self, xtts_checkpoint, xtts_config, xtts_vocab):
        """Load and optimize the XTTS model"""
        try:
            self.state_manager.set_state(TTSState.LOADING_MODEL)
            self.clear_inference_memory()

            if not all([xtts_checkpoint, xtts_config, xtts_vocab]):
                print("Missing required model paths!")
                self.state_manager.set_state(
                    TTSState.ERROR, {"error": "Missing required model paths"}
                )
                return

            # Load configuration
            config = XttsConfig()
            config.load_json(xtts_config)

            # Store paths
            self.xtts_checkpoint = xtts_checkpoint
            self.xtts_config = xtts_config
            self.xtts_vocab = xtts_vocab

            # Initialize model
            print("Loading XTTS model...")
            self.xtts_model = Xtts.init_from_config(config)

            # Load checkpoint with optimizations
            self.xtts_model.load_checkpoint(
                config,
                checkpoint_path=xtts_checkpoint,
                vocab_path=xtts_vocab,
                use_deepspeed=True,
            )

            # Apply additional optimizations
            self.optimize_model()
            print("Model loaded and optimized!")

            # Set state to IDLE after successful model loading
            self.state_manager.set_state(TTSState.IDLE)

        except Exception as e:
            print(f"Error loading model: {e}")
            self.xtts_model = None
            self.state_manager.set_state(TTSState.ERROR, {"error": str(e)})

    def prepare_batches(self, text: str, language: str) -> List[str]:
        """Prepare optimized batches for processing"""
        sentences = sent_tokenize(text, language=language)
        batches = []
        current_batch = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Handle long sentences
            if sentence_length > self.max_batch_length:
                if current_batch:
                    batches.append(" ".join(current_batch))
                    current_batch = []
                    current_length = 0

                # Split long sentence
                words = sentence.split()
                current_sentence = []
                current_sent_length = 0

                for word in words:
                    if current_sent_length + len(word) + 1 <= self.max_batch_length:
                        current_sentence.append(word)
                        current_sent_length += len(word) + 1
                    else:
                        batches.append(" ".join(current_sentence))
                        current_sentence = [word]
                        current_sent_length = len(word)

                if current_sentence:
                    batches.append(" ".join(current_sentence))

            # Handle normal sentences
            elif current_length + sentence_length + 1 <= self.max_batch_length:
                current_batch.append(sentence)
                current_length += sentence_length + 1
            else:
                batches.append(" ".join(current_batch))
                current_batch = [sentence]
                current_length = sentence_length

        if current_batch:
            batches.append(" ".join(current_batch))

        return batches

    def run_tts(
        self, lang: str, tts_text: str, speaker_audio_file: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Generate speech with optimized batch processing"""
        try:
            self.state_manager.set_state(TTSState.GENERATING)

            supported_languages = {
                "ar": "arabic",
                "pt": "portuguese",
                "zh-cn": "chinese",
                "cs": "czech",
                "nl": "dutch",
                "en": "english",
                "fr": "french",
                "de": "german",
                "it": "italian",
                "pl": "polish",
                "ru": "russian",
                "es": "spanish",
                "tr": "turkish",
                "ja": "japanese",
                "ko": "korean",
                "hu": "hungarian",
                "hi": "hindi",
            }

            # Validate inputs
            if lang not in supported_languages and lang is not None:
                print(f"Unsupported language '{lang}'. Using 'de' as default.")
                lang = "de"

            if self.xtts_model is None or not speaker_audio_file:
                self.state_manager.set_state(
                    TTSState.ERROR,
                    {"error": "Model not loaded or missing speaker audio"},
                )
                return None, None

            # Get conditioning latents
            with torch.no_grad():
                gpt_cond_latent, speaker_embedding = (
                    self.xtts_model.get_conditioning_latents(
                        audio_path=speaker_audio_file,
                        gpt_cond_len=int(self.xtts_model.config.gpt_cond_len),
                        max_ref_length=int(self.xtts_model.config.max_ref_len),
                        sound_norm_refs=bool(self.xtts_model.config.sound_norm_refs),
                    )
                )

            # Prepare batches
            batches = self.prepare_batches(tts_text, supported_languages[lang])

            # Process batches
            wav_chunks = []
            for batch in batches:
                with torch.no_grad():
                    out = self.xtts_model.inference(
                        text=batch,
                        language=lang,
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        temperature=float(0.001),
                        length_penalty=float(1),
                        repetition_penalty=float(5),
                        top_k=int(80),
                        top_p=float(0.95),
                    )
                    wav_chunks.append(torch.tensor(out["wav"]).unsqueeze(0))
                    self.clear_inference_memory()

            # Concatenate results
            final_wav = (
                torch.cat(wav_chunks, dim=1) if len(wav_chunks) > 1 else wav_chunks[0]
            )

            # Save output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
                torchaudio.save(fp.name, final_wav, self.sample_rate)
                self.state_manager.set_state(TTSState.IDLE)
                return fp.name, speaker_audio_file

        except Exception as e:
            print(f"Error generating speech: {e}")
            self.state_manager.set_state(TTSState.ERROR, {"error": str(e)})
            return None, None
        finally:
            self.clear_inference_memory()

    def play_audio(self, audio_file_path: str, device: Optional[str] = None) -> None:
        """Play the generated audio"""
        try:
            self.state_manager.set_state(TTSState.PLAYING)

            fs, data = read_wav(audio_file_path)
            sd.play(data, fs, device=device, blocking=True)

            self.state_manager.set_state(TTSState.IDLE)
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.state_manager.set_state(TTSState.ERROR, {"error": str(e)})


if __name__ == "__main__":
    tts_handler = TTSHandler()
    tts_handler.load_model(
        xtts_checkpoint="/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/model.pth",
        xtts_config="/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/config.json",
        xtts_vocab="/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/vocab.json",
    )

    import time

    start = time.time()
    out_path, _ = tts_handler.run_tts(
        lang="de",
        tts_text="Hallo! Ich bin SweetPicker, dein persönlicher Süßigkeiten-Partner von der Technischen Hochschule Georg Agricola. Ich kann dir eine Auswahl an vier leckeren Süßigkeiten anbieten: Maoam, Snickers, Kinderriegel und Milkyway. Welche davon möchtest du gerne haben?",
        speaker_audio_file="/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/reference.wav",
    )
    if out_path:
        end = time.time()
        print(end - start)
        print("Speech generated!", out_path)

        tts_handler.play_audio(out_path)

        input("\nPress Enter to exit")
        os.remove(out_path)
