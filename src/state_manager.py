from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Any, Union
from dataclasses import dataclass
from threading import RLock
import time
import logging
import inspect
from functools import wraps
from colorama import Fore, Back, Style
from sweet_types import SweetType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemState(Enum):
    """System-wide states that represent the overall state of the chatbot system"""

    INITIALIZING = auto()
    IDLE = auto()
    LISTENING = auto()
    PROCESSING_SPEECH = auto()
    PROCESSING_LLM = auto()
    GENERATING_RESPONSE = auto()
    ERROR = auto()


class RobotState(Enum):
    """States specific to the robot component"""

    WAITING_FOR_SELECTION = auto()
    SEARCHING_FOR_SWEET = auto()
    SEARCHING_FOR_SNICKERS = auto()
    SEARCHING_FOR_MILKYWAY = auto()
    SEARCHING_FOR_MAOAM = auto()
    SEARCHING_FOR_KINDERRIEGEL = auto()
    EXECUTING_PICKUP = auto()
    OBJECT_GIVEN = auto()
    ERROR = auto()

    @classmethod
    def get_search_state(cls, sweet_type: SweetType) -> "RobotState":
        """Get the appropriate search state for a given sweet type"""
        mapping = {
            SweetType.SNICKERS: cls.SEARCHING_FOR_SNICKERS,
            SweetType.MILKYWAY: cls.SEARCHING_FOR_MILKYWAY,
            SweetType.MAOAM: cls.SEARCHING_FOR_MAOAM,
            SweetType.KINDERRIEGEL: cls.SEARCHING_FOR_KINDERRIEGEL,
        }
        return mapping[sweet_type]

    @classmethod
    def get_sweet_type(cls, state: "RobotState") -> Optional[SweetType]:
        """Get the sweet type associated with a search state"""
        mapping = {
            cls.SEARCHING_FOR_SNICKERS: SweetType.SNICKERS,
            cls.SEARCHING_FOR_MILKYWAY: SweetType.MILKYWAY,
            cls.SEARCHING_FOR_MAOAM: SweetType.MAOAM,
            cls.SEARCHING_FOR_KINDERRIEGEL: SweetType.KINDERRIEGEL,
        }
        return mapping.get(state)


class ChatState(Enum):
    """States specific to the chat/LLM component"""

    IDLE = auto()
    BUSY = auto()
    ERROR = auto()


class STTState(Enum):
    """States specific to the speech-to-text component"""

    INIT = auto()
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    ERROR = auto()


class TTSState(Enum):
    """States specific to the text-to-speech component"""

    INITIALIZING = auto()
    IDLE = auto()
    LOADING_MODEL = auto()
    GENERATING = auto()
    PLAYING = auto()
    ERROR = auto()


@dataclass
class StateTransition:
    """Represents a state transition with metadata"""

    old_state: Union[SystemState, ChatState, STTState, RobotState]
    new_state: Union[SystemState, ChatState, STTState, RobotState]
    timestamp: float
    metadata: Dict[str, Any] = None


def validate_caller(allowed_modules: List[str]):
    """Decorator to validate the calling module"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            frame = inspect.stack()[1]
            calling_module = frame.filename.split("/")[-1].split(".")[0]
            if calling_module not in allowed_modules:
                raise PermissionError(f"Access denied for module: {calling_module}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


class StateManagerBase:
    """Base class for state management implementation"""

    def __init__(self, state_enum: type, initial_state: Enum):
        self._lock = RLock()
        self._state_enum = state_enum
        self._current_state = initial_state
        self._previous_state = initial_state
        self._observers: Dict[Enum, List[Callable[[StateTransition], None]]] = {
            state: [] for state in state_enum
        }
        self._state_history: List[StateTransition] = []
        self._allowed_transitions = self._define_allowed_transitions()

    def _define_allowed_transitions(self) -> Dict[Enum, List[Enum]]:
        """Define valid state transitions for the specific state type"""
        raise NotImplementedError

    @validate_caller(
        [
            "chatbot_orchestrator",
            "__main__",
            "llm",
            "state_manager",
            "speech_recognition",
            "speech_segmenter",
            "llm",
            "text_to_speech_generator_fish",
            "text_to_speech_generator",
        ]
    )
    def set_state(self, new_state: Enum, metadata: Dict[str, Any] = None) -> None:
        """Thread-safe state transition with validation"""
        with self._lock:
            if (
                not self._is_valid_transition(new_state)
                and new_state is not self._current_state
            ):
                logger.warning(
                    f"{Style.RESET_ALL}{Fore.RED}Invalid state transition: {self._current_state} -> {new_state}{Style.RESET_ALL}"
                )
                return

            self._previous_state = self._current_state
            self._current_state = new_state

            transition = StateTransition(
                old_state=self._previous_state,
                new_state=new_state,
                timestamp=time.time(),
                metadata=metadata,
            )

            self._state_history.append(transition)
            self._notify_observers(transition)
            logger.info(
                f"{Style.RESET_ALL}{Fore.GREEN}: {self._previous_state} -> {self._current_state}{Style.RESET_ALL}"
            )

    def get_state(self) -> Enum:
        """Get current state"""
        with self._lock:
            return self._current_state

    def register_observer(
        self, state: Enum, observer: Callable[[StateTransition], None]
    ) -> None:
        """Register an observer for a specific state"""
        with self._lock:
            if state not in self._state_enum:
                raise ValueError(f"Invalid state: {state}")
            self._observers[state].append(observer)

    def remove_observer(
        self, state: Enum, observer: Callable[[StateTransition], None]
    ) -> None:
        """Remove an observer"""
        with self._lock:
            if observer in self._observers[state]:
                self._observers[state].remove(observer)

    def _is_valid_transition(self, new_state: Enum) -> bool:
        """Check if state transition is valid"""
        return new_state in self._allowed_transitions[self._current_state]

    def _notify_observers(self, transition: StateTransition) -> None:
        """Notify observers of state change"""
        observers = self._observers[transition.new_state]
        for observer in observers:
            try:
                observer(transition)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}", exc_info=True)


class SystemStateManager(StateManagerBase):
    """Manager for system-wide states"""

    def __init__(self):
        super().__init__(SystemState, SystemState.INITIALIZING)

    def _define_allowed_transitions(self) -> Dict[SystemState, List[SystemState]]:
        return {
            SystemState.INITIALIZING: [SystemState.IDLE, SystemState.ERROR],
            SystemState.IDLE: [SystemState.LISTENING, SystemState.ERROR],
            SystemState.LISTENING: [
                SystemState.PROCESSING_SPEECH,
                SystemState.IDLE,
                SystemState.ERROR,
            ],
            SystemState.PROCESSING_SPEECH: [
                SystemState.PROCESSING_LLM,
                SystemState.IDLE,
                SystemState.ERROR,
            ],
            SystemState.PROCESSING_LLM: [
                SystemState.GENERATING_RESPONSE,
                SystemState.ERROR,
            ],
            SystemState.GENERATING_RESPONSE: [SystemState.IDLE, SystemState.ERROR],
            SystemState.ERROR: [SystemState.IDLE],
        }


class RobotStateManager(StateManagerBase):
    """Manager for robot states"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RobotStateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            super().__init__(RobotState, RobotState.WAITING_FOR_SELECTION)
            self._current_sweet = None
            self._initialized = True

    def _define_allowed_transitions(self) -> Dict[RobotState, List[RobotState]]:
        # All search states can transition to EXECUTING_PICKUP or ERROR
        search_transitions = [RobotState.EXECUTING_PICKUP, RobotState.ERROR]

        return {
            RobotState.WAITING_FOR_SELECTION: [
                RobotState.SEARCHING_FOR_SNICKERS,
                RobotState.SEARCHING_FOR_MILKYWAY,
                RobotState.SEARCHING_FOR_MAOAM,
                RobotState.SEARCHING_FOR_KINDERRIEGEL,
                RobotState.SEARCHING_FOR_SWEET,
                RobotState.ERROR,
            ],
            RobotState.SEARCHING_FOR_SWEET: search_transitions,
            RobotState.SEARCHING_FOR_SNICKERS: search_transitions,
            RobotState.SEARCHING_FOR_MILKYWAY: search_transitions,
            RobotState.SEARCHING_FOR_MAOAM: search_transitions,
            RobotState.SEARCHING_FOR_KINDERRIEGEL: search_transitions,
            RobotState.EXECUTING_PICKUP: [RobotState.OBJECT_GIVEN, RobotState.ERROR],
            RobotState.OBJECT_GIVEN: [
                RobotState.WAITING_FOR_SELECTION,
                RobotState.ERROR,
            ],
            RobotState.ERROR: [RobotState.WAITING_FOR_SELECTION],
        }

    def start_searching(self, sweet_type: SweetType) -> bool:
        """
        Start searching for a specific sweet type

        Args:
            sweet_type: The type of sweet to search for

        Returns:
            bool: True if transition was successful
        """
        try:
            # Store the sweet type before state transition
            self._current_sweet = sweet_type

            search_state = RobotState.get_search_state(sweet_type)
            self.set_state(
                search_state,
                {
                    "sweet_type": sweet_type.value,
                    "display_name": sweet_type.display_name,
                },
            )
            return True
        except (KeyError, ValueError) as e:
            logger.error(f"Invalid sweet type: {e}")
            self._current_sweet = None
            self.set_state(RobotState.ERROR, {"error": str(e)})
            return False

    def get_current_sweet(self) -> Optional[SweetType]:
        """Get the sweet type currently being searched for or handled"""
        with self._lock:
            return self._current_sweet


class ChatStateManager(StateManagerBase):
    """Manager for chat/LLM states"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatStateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            super().__init__(ChatState, ChatState.IDLE)
            self._initialized = True

    def _define_allowed_transitions(self) -> Dict[ChatState, List[ChatState]]:
        return {
            ChatState.IDLE: [ChatState.BUSY, ChatState.ERROR],
            ChatState.BUSY: [ChatState.IDLE, ChatState.ERROR],
            ChatState.ERROR: [ChatState.IDLE],
        }


class STTStateManager(StateManagerBase):
    """Manager for speech-to-text states"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(STTStateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            super().__init__(STTState, STTState.INIT)
            self._initialized = True

    def _define_allowed_transitions(self) -> Dict[STTState, List[STTState]]:
        return {
            STTState.INIT: [STTState.IDLE, STTState.ERROR],
            STTState.IDLE: [STTState.LISTENING, STTState.ERROR],
            STTState.LISTENING: [STTState.TRANSCRIBING, STTState.ERROR],
            STTState.TRANSCRIBING: [STTState.IDLE, STTState.ERROR],
            STTState.ERROR: [STTState.IDLE],
        }


class TTSStateManager(StateManagerBase):
    """Manager for text-to-speech states"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TTSStateManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            super().__init__(TTSState, TTSState.INITIALIZING)
            self._initialized = True

    def _define_allowed_transitions(self) -> Dict[TTSState, List[TTSState]]:
        return {
            TTSState.INITIALIZING: [TTSState.LOADING_MODEL, TTSState.ERROR],
            TTSState.LOADING_MODEL: [TTSState.IDLE, TTSState.ERROR],
            TTSState.IDLE: [TTSState.GENERATING, TTSState.PLAYING, TTSState.ERROR],
            TTSState.GENERATING: [TTSState.PLAYING, TTSState.IDLE, TTSState.ERROR],
            TTSState.PLAYING: [TTSState.IDLE, TTSState.ERROR],
            TTSState.ERROR: [TTSState.IDLE],
        }
