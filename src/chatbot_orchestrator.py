#!/usr/bin/env python3
import tempfile
import torch
from speech_recognition import WhisperXProcessor
from speech_segmenter import MicrophoneSpeechDetector, SAMPLE_RATE
import soundfile as sf
import os
import time
import argparse
import socket
from state_manager import (
    SystemStateManager,
    ChatStateManager,
    STTStateManager,
    RobotStateManager,
    TTSStateManager,
)
from state_manager import SystemState, ChatState, STTState, RobotState, TTSState
from config_reader import ConfigManager
from llm import LLM
from colorama import Fore, Back, Style
import subprocess
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from udp_communication import UDPCommunication, MessageType
from sweet_types import SweetType
from text_to_speech_generator_fish import FishSpeechTTS
from text_to_speech_generator import TTSHandler
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close
    #   ip = "172.20.11.72"
    return ip


class ChatbotOrchestrator:
    def __init__(self, remote_ip_adress: str = "127.0.0.1"):
        self.remote_ip_adress = remote_ip_adress
        if self.remote_ip_adress == "127.0.0.1":
            self.local_ip_adress: str = "127.0.0.1"
        else:
            self.local_ip_adress: str = get_local_ip()

        print(f"remote IP-Adress: {self.remote_ip_adress} ")
        print(f"local IP-Adress: {self.local_ip_adress} ")

        self.initialize()

    def initialize(self):
        # Initialize state managers
        self.system_state = SystemStateManager()
        self.speech_state = STTStateManager()
        self.llm_state = ChatStateManager()
        self.robot_state = RobotStateManager()
        self.tts_state = TTSStateManager()

        # Register state change observers
        self.speech_state.register_observer(STTState.ERROR, self._handle_speech_error)
        self.llm_state.register_observer(ChatState.ERROR, self._handle_llm_error)
        self.robot_state.register_observer(RobotState.ERROR, self._handle_robot_error)
        self.tts_state.register_observer(TTSState.ERROR, self._handle_tts_error)

        # Register observers for robot state transitions
        self.robot_state.register_observer(
            RobotState.SEARCHING_FOR_SWEET, self._handle_robot_searching
        )
        self.robot_state.register_observer(
            RobotState.SEARCHING_FOR_SNICKERS, self._handle_robot_searching
        )
        self.robot_state.register_observer(
            RobotState.SEARCHING_FOR_MAOAM, self._handle_robot_searching
        )
        self.robot_state.register_observer(
            RobotState.SEARCHING_FOR_MILKYWAY, self._handle_robot_searching
        )
        self.robot_state.register_observer(
            RobotState.SEARCHING_FOR_KINDERRIEGEL, self._handle_robot_searching
        )
        self.robot_state.register_observer(
            RobotState.EXECUTING_PICKUP, self._handle_robot_pickup
        )
        self.robot_state.register_observer(
            RobotState.OBJECT_GIVEN, self._handle_robot_complete
        )

        # Set initial system state
        self.system_state.set_state(SystemState.INITIALIZING)

        try:

            def init_whisper():
                print("Loading WhisperXProcessor...")
                recognizer = WhisperXProcessor()
                recognizer.load_model()
                return recognizer

            # Initialize config manager
            self.config_manager = ConfigManager()

            # Initialize TTS Handler, loading is a seperate method.
            self.tts_handler = TTSHandler()

            # Run model-loading initializations in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                print("Loading components in parallel...")
                llm_future = executor.submit(LLM, verbose=False)
                whisper_future = executor.submit(init_whisper)

                # Fish Speech
                # tts_future = executor.submit(
                #     FishSpeechTTS,
                #     base_dir="/home/student/catkin_ws/src/chatbot_lm",
                #     checkpoint_dir="checkpoints/fish-speech-1.5",
                # )

                # XTTS-2
                tts_future = (
                    executor.submit(
                        self.tts_handler.load_model,
                        xtts_checkpoint="/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/model.pth",
                        xtts_config="/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/config.json",
                        xtts_vocab="/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/vocab.json",
                    ),
                )
                detector_future = executor.submit(MicrophoneSpeechDetector)

                # Get results
                self.sweetpicker = llm_future.result()
                self.recognizer = whisper_future.result()
                # self.tts_handler = tts_future.result()
                self.detector = detector_future.result()

            # # Warmup model (has to be done with same reference speaker)
            # self._generate_speech_response(
            #     "Möchtest du eine der verfügbaren Süßigkeiten haben? Es sind Maoam, Snickers, Kinderriegel oder Milkyway verfügbar.",
            #     speak=False,
            # )
            # self._generate_speech_response(
            #     "Kann ich Ihnen eine der folgenden Süßigkeiten anbieten?: Maoam, Snickers, Kinderriegel oder Milkyway. Welche möchten Sie haben?",
            #     speak=False,
            # )

            # Initialize UDP communication
            self.udp = UDPCommunication(
                local_ip=self.local_ip_adress,
                local_port=5001,
                remote_ip=self.remote_ip_adress,
                remote_port=5000,
            )
            self.udp.register_handler(
                MessageType.ROBOT_STATUS, self._handle_robot_status
            )

            # Set robot to initial waiting state
            self.robot_state.set_state(RobotState.WAITING_FOR_SELECTION)

            # Set states to idle after successful initialization
            self.system_state.set_state(SystemState.IDLE)

        except Exception as e:
            print(f"Initialization error: {e}")
            self.system_state.set_state(SystemState.ERROR)
            raise

    # def _generate_speech_response(self, text, speak=True):
    #     """Generate and play speech response using enhanced FishSpeechTTS"""
    #     try:
    #         # Generate speech using numpy output for better performance
    #         audio_result = self.tts_handler.generate_speech(
    #             text=text,
    #             output_format="numpy",
    #             temperature=0.7,
    #             reference_audio="/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/reference.wav",
    #             reference_text="Die bittere Wahrheit ist nun, bis ein einheitlicher Schutz der Lebensadern unserer Demokratie in einer neuen Legislaturperiode auf den Weg gebracht wird, werden viele Monate ins Land gehen, sagten von Nords und Kahn. Das ist Zeit, die wir eigentlich nicht mehr haben.",
    #             top_p=0.7,
    #             repetition_penalty=1.2,
    #         )

    #         # Play the audio directly
    #         if speak:
    #             self.tts_handler.play_audio(audio_result)
    #         return True

    #     except Exception as e:
    #         self.system_state.set_state(
    #             SystemState.ERROR, {"error": f"TTS error: {str(e)}"}
    #         )
    #         return False

    def _generate_speech_response(self, text, speak=True):
        """Generate and play speech response using XTTS"""
        try:
            # Generate speech file
            audio_file_path, _ = self.tts_handler.run_tts(
                lang="de",
                tts_text=text,
                speaker_audio_file="/home/student/catkin_ws/src/chatbot_lm/tts_train/optimized_model/reference.wav",
            )

            # Play the audio file
            if speak and audio_file_path:
                self.tts_handler.play_audio(audio_file_path)

            # Cleanup
            if audio_file_path is not None:
                if os.path.exists(audio_file_path):
                    try:
                        os.remove(audio_file_path)
                    except Exception as e:
                        # non critical just print and be done with it
                        print(e)
                        pass

            return True if audio_file_path else False

        except Exception as e:
            self.system_state.set_state(
                SystemState.ERROR, {"error": f"TTS error: {str(e)}"}
            )
            return False

    def _handle_robot_status(self, message):
        """Handle incoming robot status messages and update state accordingly"""
        status_data = message.data
        print(f"{Style.DIM}Robot status: {status_data}{Style.RESET_ALL}")

        # Map status messages to robot states
        status_mapping = {
            "SEARCHING_FOR_SWEET": RobotState.SEARCHING_FOR_SWEET,
            "EXECUTING_PICKUP": RobotState.EXECUTING_PICKUP,
            "OBJECT_GIVEN": RobotState.OBJECT_GIVEN,
            "WAITING_FOR_SELECTION": RobotState.WAITING_FOR_SELECTION,
            "ERROR": RobotState.ERROR,
        }

        if isinstance(status_data, dict):
            status = status_data.get("state")
        else:
            status = status_data

        if status in status_mapping:
            self.robot_state.set_state(status_mapping[status], {"message": status_data})
        elif "error" in str(status).lower():
            self.robot_state.set_state(RobotState.ERROR, {"error": status_data})

    def _handle_robot_searching(self, transition):
        """Handle robot entering search state"""
        sweet_type = self.robot_state.get_current_sweet()
        if sweet_type:
            print(
                f"{Style.DIM}Robot searching for: {sweet_type.display_name}{Style.RESET_ALL}"
            )

    def _handle_robot_pickup(self, transition):
        """Handle robot entering pickup state"""
        sweet_type = self.robot_state.get_current_sweet()
        if sweet_type:
            print(
                f"{Style.DIM}Robot executing pickup for: {sweet_type.display_name}{Style.RESET_ALL}"
            )

    def _handle_robot_complete(self, transition):
        """Handle robot completing sweet delivery"""
        print(f"{Style.DIM}Robot completed sweet delivery{Style.RESET_ALL}")
        sweet_type = self.robot_state.get_current_sweet()

        if sweet_type:
            # Generate success message
            completion_text = (
                f"Hier ist dein {sweet_type.display_name}. Viel Spaß damit!"
            )
            print(
                f"{Fore.YELLOW}{Style.BRIGHT}SweetPicker:{Style.RESET_ALL}{Fore.YELLOW} {completion_text}{Style.RESET_ALL}"
            )

            # Generate speech for completion
            self.system_state.set_state(SystemState.GENERATING_RESPONSE)
            self._generate_speech_response(completion_text)

        # self.robot_state.set_state(RobotState.WAITING_FOR_SELECTION)

    def _handle_speech_error(self, transition):
        """Handle speech processing errors"""
        error_msg = transition.metadata.get("error", "Unknown speech error")
        # TODO: when no speech is detected in audio we can enter this error. pass for now
        pass
        print(f"{Style.DIM}Speech error: {error_msg}{Style.RESET_ALL}")
        print(f"{Style.DIM}Attempting reinitialization...{Style.RESET_ALL}")

        try:
            self.cleanup()
            self.initialize()
            print(f"{Style.DIM}Reinitialization complete{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Style.DIM}Reinitialization failed: {e}{Style.RESET_ALL}")
            self.system_state.set_state(SystemState.ERROR)

    def _handle_llm_error(self, transition):
        """Handle LLM processing errors"""
        error_msg = transition.metadata.get("error", "Unknown LLM error")
        print(f"{Style.DIM}LLM error: {error_msg}{Style.RESET_ALL}")
        print(f"{Style.DIM}Attempting reinitialization...{Style.RESET_ALL}")

        try:
            self.cleanup()
            self.initialize()
            print(f"{Style.DIM}Reinitialization complete{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Style.DIM}Reinitialization failed: {e}{Style.RESET_ALL}")
            self.system_state.set_state(SystemState.ERROR)

    def _handle_robot_error(self, transition):
        """Handle robot error state"""
        error_msg = transition.metadata.get("error", "Unknown robot error")
        print(f"{Style.DIM}Robot error: {error_msg}{Style.RESET_ALL}")

        # Generate error message for user
        error_text = "Entschuldigung, aber ich konnte die Süßigkeit leider nicht finden. Möchtest du eine andere Süßigkeit probieren?"
        print(
            f"{Fore.YELLOW}{Style.BRIGHT}SweetPicker:{Style.RESET_ALL}{Fore.YELLOW} {error_text}{Style.RESET_ALL}"
        )

        # Generate speech for error
        self.system_state.set_state(SystemState.GENERATING_RESPONSE)
        self._generate_speech_response(error_text)
        self.system_state.set_state(SystemState.IDLE)
        self.robot_state.set_state(RobotState.WAITING_FOR_SELECTION)

    def _handle_tts_error(self, transition):
        """Handle TTS processing errors"""
        error_msg = transition.metadata.get("error", "Unknown TTS error")
        print(f"{Style.DIM}TTS error: {error_msg}{Style.RESET_ALL}")
        print(f"{Style.DIM}Attempting reinitialization...{Style.RESET_ALL}")

        try:
            self.cleanup()
            self.initialize()
            print(f"{Style.DIM}Reinitialization complete{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Style.DIM}Reinitialization failed: {e}{Style.RESET_ALL}")
            self.system_state.set_state(SystemState.ERROR)

    def request_sweet(self, sweet_id: int):
        """Send sweet selection request via UDP and update robot state"""
        try:
            sweet_type = SweetType(sweet_id)

            # First update robot state
            if self.robot_state.start_searching(sweet_type):
                # Then send UDP message
                success = self.udp.send_message(MessageType.SWEET_SELECTION, sweet_id)
                if success:
                    print(
                        f"{Style.DIM}Sweet selection request sent for: {sweet_type.display_name}{Style.RESET_ALL}"
                    )

                    # Wait for completion
                    if self.wait_for_robot_completion():
                        # Success case is handled by _handle_robot_complete
                        pass
                    else:
                        # Timeout or error case
                        error_msg = "Robot operation timed out"
                        self.robot_state.set_state(
                            RobotState.ERROR, {"error": error_msg}
                        )
                else:
                    error_msg = "Failed to send sweet request"
                    print(f"{Style.DIM}{error_msg}{Style.RESET_ALL}")
                    self.robot_state.set_state(RobotState.ERROR, {"error": error_msg})
            else:
                print(
                    f"{Style.DIM}Failed to initiate search for sweet{Style.RESET_ALL}"
                )
        except ValueError as e:
            error_msg = f"Invalid sweet ID: {e}"
            print(f"{Style.DIM}{error_msg}{Style.RESET_ALL}")
            self.robot_state.set_state(RobotState.ERROR, {"error": error_msg})

    def wait_for_robot_completion(self, timeout: float = 60.0) -> bool:
        """Wait for robot to complete current operation"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_state = self.robot_state.get_state()
            if current_state == RobotState.WAITING_FOR_SELECTION:
                return True
            elif current_state == RobotState.ERROR:
                return False
            time.sleep(0.5)
        return False

    def run(self):
        """Main processing loop"""
        temp_file_path = None
        speech_chunk_filtered_path = None
        temp_files = []

        try:
            while True:
                # Wait for idle states
                while not (
                    self.system_state.get_state() == SystemState.IDLE
                    and self.speech_state.get_state() == STTState.IDLE
                    and self.llm_state.get_state() == ChatState.IDLE
                    and self.robot_state.get_state() == RobotState.WAITING_FOR_SELECTION
                    and self.tts_state.get_state() == TTSState.IDLE
                ):
                    system_state = self.system_state.get_state()
                    speech_state = self.speech_state.get_state()
                    llm_state = self.llm_state.get_state()
                    robot_state = self.robot_state.get_state()
                    tts_state = self.tts_state.get_state()
                    print(
                        f"{system_state}\n{speech_state}\n{llm_state}\n{robot_state}\n{tts_state}\n"
                    )
                    time.sleep(0.5)

                # print(f"{Style.DIM}Ready for Speech{Style.RESET_ALL}")

                # Process speech
                self.system_state.set_state(SystemState.LISTENING)

                speech_chunk = self.detector.get_speech()

                # Save and filter audio
                temp_file_path = save_temp_wav(speech_chunk, SAMPLE_RATE)
                if temp_file_path:
                    temp_files.append(temp_file_path)
                speech_chunk_filtered_path = ffmpeg_filter(
                    temp_file_path, trim_silence=False
                )
                if speech_chunk_filtered_path:
                    temp_files.append(speech_chunk_filtered_path)

                # Process speech
                self.system_state.set_state(SystemState.PROCESSING_SPEECH)
                transcription = self._process_speech(speech_chunk_filtered_path)
                if not transcription:
                    continue

                # Process with LLM
                self.system_state.set_state(SystemState.PROCESSING_LLM)

                response, selection = self.sweetpicker.chat(transcription)
                print(
                    f"{Fore.YELLOW}{Style.BRIGHT}SweetPicker:{Style.RESET_ALL}{Fore.YELLOW} {response}{Style.RESET_ALL}"
                )

                # Generate speech response
                self.system_state.set_state(SystemState.GENERATING_RESPONSE)
                self._generate_speech_response(response)

                # Request the sweet from the robot
                if selection:
                    self.request_sweet(selection.value)

                # Reset states
                self.system_state.set_state(SystemState.IDLE)
                self._cleanup_temp_files(temp_files)
                temp_files = []

        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.system_state.set_state(SystemState.ERROR, {"error": str(e)})
        finally:
            self._cleanup_temp_files(temp_files)
            temp_files = []

    def _process_speech(self, audio_path):
        """Process speech from audio file and return transcription"""
        try:
            transcription_result, audio = self.recognizer.transcribe(audio_path)

            if len(transcription_result) == 0:
                print("No speech detected")
                return None

            aligned_result = self.recognizer.align_transcription(
                transcription_result["segments"],
                transcription_result["language"],
                audio,
            )

            final_result, _ = self.recognizer.assign_speaker_labels(
                audio, aligned_result
            )
            speakers = self.recognizer.print_speaker_text(final_result["segments"])

            speaker_id, speaker_text = wake_word_detection(
                speakers, ("Sweet Picker", "SweetPicker", "Sweetpika")
            )

            if speaker_id is None:
                speaker_text = next(iter(speakers.values()))

            print(
                f"{Fore.YELLOW}{Style.BRIGHT}Student:{Style.RESET_ALL}{Fore.YELLOW} {speaker_text}{Style.RESET_ALL}"
            )
            return speaker_text

        except Exception as e:
            self.speech_state.set_state(STTState.ERROR, {"error": str(e)})
            return None

    def cleanup(self):
        """Clean up resources"""
        try:
            print("Cleaning up resources...")
            if hasattr(self, "detector"):
                self.detector.close_microphone_stream()
            if hasattr(self, "udp"):
                self.udp.close()
            # if hasattr(self, "tts_handler"):
            #     self.tts_handler.cleanup()
            if hasattr(self, "tts_handler"):
                self.tts_handler.clear_inference_memory()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def reinitialize(self):
        """Clean up and reinitialize the orchestrator"""
        print("Reinitializing ChatbotOrchestrator...")
        self.cleanup()
        self.initialize()
        print("Reinitialization complete")

    def _cleanup_temp_files(self, temp_files: list) -> None:
        """
        Clean up a list of temporary files.

        Args:
            temp_files (list): List of file paths to clean up
        """
        if not temp_files:
            return

        for file_path in temp_files:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except (OSError, FileNotFoundError):
                pass  # Silently handle any file cleanup errors


# Helper functions remain unchanged
def save_temp_wav(data, sample_rate):
    """Save the numpy array as a temporary WAV file"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        print({temp_file.name})
        sf.write(temp_file.name, data, sample_rate)
        return temp_file.name
    finally:
        temp_file.close()


def wake_word_detection(speakers, wake_words, threshold=0.61):
    """Detect wake words in speech"""
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5)).fit(wake_words)

    for speaker_id, text in speakers.items():
        words = text.split()
        for i in range(len(words)):
            segment = " ".join(words[i : i + len(wake_words)])
            if not segment:
                continue

            segment_vector = vectorizer.transform([segment])
            wake_word_vector = vectorizer.transform(wake_words)
            similarities = cosine_similarity(segment_vector, wake_word_vector).flatten()

            if any(sim > threshold for sim in similarities):
                best_match_index = similarities.argmax()
                match_word = wake_words[best_match_index]
                index = text.lower().find(match_word.lower())
                spoken_text = text[index:].strip()
                return (speaker_id, spoken_text)

    return (None, None)


def ffmpeg_filter(
    wave_file, bandpass_filter=None, trim_silence=None, noise_filter=None
):
    """Apply audio filters using FFmpeg"""
    if bandpass_filter is None:
        bandpass_filter = "lowpass=7000,highpass=200,"
    else:
        bandpass_filter = ""
    if trim_silence is None:
        trim_silence = "areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,"
    else:
        trim_silence = ""
    if noise_filter is None:
        noise_filter = "afftdn=nr=12:nf=-50:tn=1"
    else:
        noise_filter = ""

    try:
        out_filename = wave_file + str(uuid.uuid4()) + ".wav"
        shell_command = f"ffmpeg -y -i {wave_file} -af {bandpass_filter}{trim_silence}{noise_filter} {out_filename}".split(
            " "
        )
        subprocess.run(
            [item for item in shell_command],
            capture_output=False,
            text=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out_filename
    except subprocess.CalledProcessError:
        return wave_file


def main():
    parser = argparse.ArgumentParser(description="IP-Adress parser")
    parser.add_argument("--IP", help="IP-Adress of ros_bridge")

    args = parser.parse_args()
    if args.IP:
        remote_ip: str = args.IP
    else:
        remote_ip: str = "127.0.0.1"

    orchestrator = ChatbotOrchestrator(remote_ip)
    orchestrator.run()


if __name__ == "__main__":
    main()
