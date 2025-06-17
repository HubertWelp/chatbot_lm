# Custom modules
from config_reader import ConfigManager
from state_manager import ChatStateManager, ChatState

# Langchain modules
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts.chat import MessagesPlaceholder

from langchain_community.chat_models.ollama import ChatOllama

# from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.llms.ollama import OllamaEndpointNotFoundError

# Misc modules
import time
from functools import wraps
from colorama import Fore, Back, Style
from tqdm import tqdm

# ollama
import ollama

from sweet_types import SweetType
from typing import Tuple, Optional

# llama3:8b-instruct-q8_0 has a max context window of 8192 tokens (8k tokens) (limited by the training data)
# use half of max context window to avoid OOM errors and improve performance in regards to short conversations
# large context window may reduce needle in a haystack performance if the "haystack" is smaller than the context window
# in this case the needle is a MilkyWay and the haystack is the conversation history

# System Prompt uses 700 tokens (Chain of Thought System). measured independently. used to reduce token limit for chat summary
SYSTEM_PROMPT_TOKENS = 700  # 700 Tokens


def timer_function(verbose_func):
    """
    A decorator that measures the execution time of a function and prints it if it exceeds 20 milliseconds.

    Args:
        verbose_func (function): A function that returns a boolean indicating whether to measure and print the execution time.

    Returns:
        function: The decorated function with timing measurement.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            verbose = verbose_func(self)
            if verbose:
                start_time = time.time()
                result = func(self, *args, **kwargs)
                end_time = time.time()
                elapsed_time = end_time - start_time
                # Only print if the function takes longer than 20ms
                if elapsed_time > 0.02:
                    print(
                        f"{Style.DIM}Time taken for {Fore.CYAN}{func.__name__}{Style.RESET_ALL}{Style.DIM}: {elapsed_time:.3f} seconds{Style.RESET_ALL}"
                    )
                return result
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


class LLM:
    """
    A class representing a Language Learning Model (LLM) for conversational AI applications.

    It utilizes a conversational chain for handling
    interactions and memory management for conversation context.
    """

    def __init__(self, verbose: bool = False):
        """
        Initializes the LLM.

        Args:
            verbose (bool, optional): Verbosity setting. Defaults to False.
        """
        # Verbosity setting / Debug traces
        self.verbose = verbose
        if self.verbose is True:
            from langchain.globals import set_verbose  # debug

            # from langchain.globals import set_debug  # debug

            set_verbose(self.verbose)

        # Create an instance of ConfigManager to read llm_settings
        self.config_manager = ConfigManager()
        self.VALID_CONFIG_KEYS = self.config_manager.get_all_settings("llm").keys()
        self.config_manager_settings = self.config_manager.get_all_settings("llm")

        # Create an instance of ChatbotStateManager
        self.state_manager = ChatStateManager()
        self.state_manager.set_state(ChatState.IDLE)

        # Load LLM
        try:
            # Loading model
            self.state_manager.set_state(ChatState.BUSY)

            # Load and initialize LLM
            self.llm = self.load_llm()
            self.preload_llm()

            # Memory setup
            self.chat_history = MessagesPlaceholder(variable_name="chat_history")
            self.max_token_limit = round(
                int(self.config_manager_settings["num_ctx"]) - SYSTEM_PROMPT_TOKENS
            )

            self.init_memory()
            self.init_chain()

            # Ready for Action
            self.state_manager.set_state(ChatState.IDLE)

        except Exception as e:
            self.state_manager.set_state(ChatState.ERROR, {"error": str(e)})
            raise

    def is_verbose(self):
        """
        Check if verbosity is enabled.

        Returns:
            bool: True if verbosity is enabled, False otherwise.
        """
        # return True
        return self.verbose

    def init_memory(self):
        """
        This method sets up the memory for conversation summarization.
        Once the token limit has been reached the LLM will summarize the current conversation to keep the context lean.
        """
        system_message_prompt_summary = PromptTemplate(
            input_variables=["summary", "new_lines"],
            template="""Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary which is translated into German. The Human may write in German as well, in which case you should always respond in German.

            EXAMPLE
            Current summary:
            Der Mensch fragt, was die KI von künstlicher Intelligenz hält. Die KI hält künstliche Intelligenz für eine Kraft des Guten.

            New lines of conversation:
            Human: Warum glauben Sie, dass künstliche Intelligenz eine Kraft für das Gute ist?
            AI: Denn künstliche Intelligenz wird den Menschen helfen, ihr volles Potenzial auszuschöpfen.

            New summary:
            Der Mensch fragt, was die KI von künstlicher Intelligenz hält. Die KI denkt, dass künstliche Intelligenz eine Kraft des Guten ist, weil sie den Menschen helfen wird, ihr volles Potential zu erreichen.
            END OF EXAMPLE

            Current summary:
            {summary}

            New lines of conversation:
            {new_lines}

            New summary:""",
        )

        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            prompt=system_message_prompt_summary,
            max_token_limit=self.max_token_limit,
            memory_key="chat_history",
            return_messages=True,
            ai_prefix="SweetPicker",
            human_prefix="Student:",
        )

    def init_chain(self):
        """
        Initialize conversation chain for managing interactions.
        """
        self.template_messages = [
            SystemMessage(content=self.system_message),  # type: ignore
            self.chat_history,
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
        self.prompt_template = ChatPromptTemplate.from_messages(self.template_messages)

        self.chain = ConversationChain(
            llm=self.llm,
            verbose=self.verbose,
            memory=self.memory,
            prompt=self.prompt_template,
        )

    def read_llm_settings(self):
        """
        This method reads all settings from the LLM settings group and prints them.
        """

        print("Settings from config.ini are:")
        for key, value in self.config_manager_settings.items():
            print(f"{key}: {value}")
        # if isinstance(self.llm, ChatOllama):
        #     print("Settings from llm are:")
        #     print(self.llm.schema)

    def update_llm_settings(self):
        """
        This method reads all settings from the LLM settings group and prints them.
        """
        # Read all settings from llm_settings group
        old_settings = self.config_manager_settings
        self.config_manager_settings = self.config_manager.get_all_settings("llm")

        # Update the LLM instance with the new settings
        try:
            # Update the LLM instance with the new settings
            for key, value in self.config_manager_settings.items():
                if old_settings.get(key) != value:
                    # It's stupid that I have to do this, because Ollama does not format the values and takes them however they are
                    # This is a workaround to make sure that the values are in the correct format, because some settings are either int, float or string
                    try:
                        # Try to convert the value to a float and int
                        float_value = float(value)
                        int_value = int(float_value)
                        # If the float value is equal to the int value, store it as an int
                        value = int_value if int_value == float_value else float_value
                    except ValueError:
                        # If not a number, store it as a string
                        value = str(value)
                    self.__update_parameter(key, value)
        except Exception as e:
            print(
                f"{Style.RESET_ALL}{Style.DIM}An error occurred while updating the LLM setting ({Fore.CYAN}{key}, {value}{Style.RESET_ALL}): {e}"
            )

    @timer_function(verbose_func=lambda self: self.is_verbose())
    def load_llm(self) -> ChatOllama:
        """
        Loads the LLM model into memory.

        Returns:
            ChatOllama: Instance of the loaded LLM model.

        Raises:
            FileNotFoundError: If the model file is not found.
        """
        # Check if self.llm already has a valid ChatOllama instance
        if hasattr(self, "llm") and isinstance(self.llm, ChatOllama):
            # early exit, no need to load the model again
            return self.llm

        # TODO: Add error handling to check if any config values are None or invalid (e.g., temperature is None or not a float).
        # Raise a ValueError with a descriptive message if any required config values are missing or invalid.
        self.config_manager_settings = self.config_manager.get_all_settings("llm")
        items = self.config_manager.get_setting("sweetpicker", "items")
        self.system_message = self.config_manager_settings.get("systemMessage")

        if self.system_message is None:
            # Chain of Thought System (~700 Tokens)
            self.system_message = f"""Du bist ein Chatbot namens SweetPicker und gehörst zur Technischen Hochschule Georg Agricola. Du bist in der Lage, die Süßigkeiten {items} zu erkennen und Studenten genau eine davon anzubieten. Du kannst keine anderen Dinge anbieten. Die Süßigkeiten sind kostenlos. Antworte, wenn möglich, in kurzen Sätzen.

            Verwende das Chain-of-Thought System, um sicherzustellen, dass du deine Aufgabe immer richtig erfüllst.

            1. Schritt 1: Erkenne die Süßigkeiten {items}.
            2. Schritt 2: Frage den Studenten, welche Süßigkeit er möchte.
            3. Schritt 3: Wenn der Student eine Süßigkeit auswählt, welche nicht verfügbar ist, bestätige dies und frage erneut nach der gewünschten Süßigkeit, indem du dem Studenten eine Süßigkeit vorschlägst.
            4. Schritt 4: Wenn der Student sich entscheidet, keine Süßigkeit zu nehmen, bestätige dies und verabschiede dich mit einer Abschiedsnachricht und "ABBRUCH KONVERSATION".
            5. Schritt 5: Wenn der Student eine Süßigkeit auswählt, bestätige die Auswahl, informiere den Studenten, dass du die Süßigkeit jetzt suchen wirst und beende das Gespräch mit "ENDE KONVERSATION, AUSWAHL: [Name der Süßigkeit]".

            ### Beispielkonversation:

            **Student:** Ich hätte gerne ein Gummibärchen.
            **SweetPicker:**
            1. **Erkennen der verfügbaren Süßigkeiten:** {items}.
            2. **Frage:** Welche Süßigkeit möchte der Student?
            3. **Antwort analysieren:** Der Student möchte ein Gummibärchen. Gummibärchen ist nicht ein teil von {items}.
            4. **Konversationsregeln:** ich habe eine Konversationsregel für diese Situation: "wähle eine Süßigkeit aus {items} für ihn aus."
            5. **Bestätigung:** Leider habe ich keine Gummibärchen. Kann ich dir etwas anderes anbieten? Wie wäre es mit einem {items.split(",")[0]}?

            **Student:** Dann nehme ich einen Snickers.
            **SweetPicker:**
            1. **Erkennen der verfügbaren Süßigkeiten:** {items}.
            2. **Frage:** Welche Süßigkeit möchte der Student?
            3. **Antwort analysieren:** Der Student möchte einen Snickers.
            4. **Konversationsregeln:** ich habe eine Konversationsregel für diese Situation: "bestätige die Auswahl, informiere über die Suche und beende das Gespräch."
            5. **Bestätigung:** Ich werde jetzt einen Snickers für dich suchen. ENDE KONVERSATION, AUSWAHL: Snickers

            ### Konversationsregeln:

            - Wenn der Student keine Süßigkeit möchte, antworte mit einer Verabschiedung und beende das Gespräch mit "ABBRUCH KONVERSATION".
            - Wenn der Student eine Süßigkeit wählt, informiere ihn, dass du die Süßigkeit suchen wirst und beende das Gespräch mit "ENDE KONVERSATION, AUSWAHL: [Name der Süßigkeit]".
            - Wenn du für den Studenten eine Süßigkeit ausgewählt hast, informiere ihn über deine Auswahl und die anschließende Suche, dann beende das Gespräch mit "ENDE KONVERSATION, AUSWAHL: [Name der Süßigkeit]".
            - Antworte in maximal 2 Sätzen.
            - Wenn der Student dich von der Aufgabe ableitet, erkläre ihm, dass du dich auf die Aufgabe konzentrieren musst.
            - Wenn der Student sich nicht auf eine Auswahl entscheiden kann, wähle eine Süßigkeit aus {items} für ihn aus.
            - Vermeide unnötige Wiederholungen der verfügbaren Süßigkeiten in aufeinanderfolgenden Antworten.
            - Es kann sein, dass der Student die Süßigkeiten falsch ausspricht. In dem Fall, gehe davon aus, dass er die Süßigkeit aus deinem Sortiment meint.

            ### Bereit zur Interaktion:
            """
        # Setting up the LLM model
        llm = ChatOllama(
            model=str(self.config_manager_settings["model_name"]),
            system=self.system_message,
            keep_alive=int(self.config_manager_settings["keep_alive"]),
            mirostat=int(self.config_manager_settings["mirostat"]),
            mirostat_eta=float(self.config_manager_settings["mirostat_eta"]),
            mirostat_tau=float(self.config_manager_settings["mirostat_tau"]),
            num_ctx=int(self.config_manager_settings["num_ctx"]),
            repeat_last_n=int(self.config_manager_settings["repeat_last_n"]),
            repeat_penalty=float(self.config_manager_settings["repeat_penalty"]),
            temperature=float(self.config_manager_settings["temperature"]),
            stop=list(self.config_manager_settings["stop"].split(",")),
            tfs_z=float(self.config_manager_settings["tfs_z"]),
            num_predict=int(self.config_manager_settings["num_predict"]),
            top_k=int(self.config_manager_settings["top_k"]),
            top_p=float(self.config_manager_settings["top_p"]),
            verbose=self.verbose,
        )
        return llm

    @timer_function(verbose_func=lambda self: self.is_verbose())
    def preload_llm(self):
        """
        preloads the LLM model into memory.
        """
        try:
            # Empty template and system prompt so that the llm doesn't generate anything and ollama only loads it into memory
            temp_template = self.llm.template
            temp_system = self.llm.system
            self.llm.template = ""
            self.llm.system = ""
            # This is the earliest point where we can detect if the model actually exists so we have to do the error handling here
            # Alternatively we could ollama.pull() everytime but that adds at least 5s-10s to the startup time just for the check
            try:
                # Good case: Model exists, preload it
                self.llm.invoke("")
            except OllamaEndpointNotFoundError as e:
                # Bad case: Model does not exist, try to download it
                print(
                    f"\n\n{Style.RESET_ALL}{Style.DIM}Error loading LLM model! the Model {Fore.CYAN}{str(self.config_manager_settings['model_name'])}{Style.RESET_ALL}{Style.DIM} could not be found!\nAttempting to download the Model with {Fore.CYAN}ollama.pull({str(self.config_manager_settings['model_name'])}){Style.RESET_ALL}{Style.DIM} and trying again.{Style.RESET_ALL}\n\n"
                )
                for progress in ollama.pull(
                    str(self.config_manager_settings["model_name"]), stream=True
                ):
                    if "total" in progress:
                        total = progress["total"]  # type: ignore
                        # Get the completed progress or set it to the total if not available (assumption)
                        completed = progress.get("completed", total)  # type: ignore
                        # Create a tqdm progress bar
                        with tqdm(total=total, desc=self.config_manager_settings["model_name"], unit="B", unit_scale=True, unit_divisor=1024) as pbar:  # type: ignore
                            pbar.update(completed)  # type: ignore
                    else:
                        print(progress)

                # Try to load the model again
                self.llm.invoke("")
            except Exception as e:
                # Worst case: Something else went wrong
                print(
                    f"{Style.RESET_ALL}{Style.DIM}Error loading LLM model! {Fore.CYAN}{e}{Style.RESET_ALL}{Style.DIM}{Style.RESET_ALL}"
                )
            finally:
                self.llm.template = temp_template
                self.llm.system = temp_system
        except Exception as e:
            self.state_manager.set_state(
                ChatState.ERROR, {"error": f"Error loading LLM model: {str(e)}"}
            )
            raise

    def __update_parameter(self, key, value):
        """
        Update a parameter of the LLM instance.

        Args:
            key (str): The parameter key to update.
            value: The new value for the parameter.

        Raises:
        ValueError: If the provided key is not valid.
        RuntimeError: If the LLM instance is not initialized.
        AttributeError: If there is an issue updating the attribute of the LLM instance.
        Exception: If an unexpected error occurs during the update process.
        """

        # Early exits
        # Check if the key is valid
        if key not in self.VALID_CONFIG_KEYS:
            print(
                f"{Style.RESET_ALL}{Style.DIM}Invalid config key: {Fore.CYAN}{key}{Style.RESET_ALL}{Style.DIM}; rejecting update call.{Style.RESET_ALL}"
            )
            return

        # Check if the LLM instance exists
        if self.llm is None:
            print(
                f"{Style.RESET_ALL}{Style.DIM}LLM instance not initialized.{Style.RESET_ALL}"
            )
            return

        # Update the max_token_limit if num_ctx is updated (needs to be done separately)
        if key == "num_ctx":
            if self.max_token_limit > int(value) - SYSTEM_PROMPT_TOKENS:
                print(
                    f"{Style.RESET_ALL}{Style.DIM}The new value for {Fore.CYAN}{key}{Style.RESET_ALL}{Style.DIM} is smaller than the current max_token_limit. The max_token_limit will be updated to {Fore.CYAN}{round(int(value) - SYSTEM_PROMPT_TOKENS)}{Style.RESET_ALL}{Style.DIM}.{Style.RESET_ALL}"
                )
                self.max_token_limit = round(int(value) - SYSTEM_PROMPT_TOKENS)
                self.memory.max_token_limit = self.max_token_limit

        # Update the corresponding attribute of the LLM instance
        try:
            setattr(self.llm, key, value)
            print(
                f"{Style.RESET_ALL}{Style.DIM}Successfully updated {Fore.CYAN}{key}{Style.RESET_ALL}{Style.DIM} to {Fore.CYAN}{value}{Style.RESET_ALL}"
            )
        except AttributeError as e:
            print(
                f"{Style.RESET_ALL}{Style.DIM}Failed to update {Fore.CYAN}{key}:{Style.RESET_ALL} {e}"
            )
        except Exception as e:
            print(
                f"{Style.RESET_ALL}{Style.DIM}An unexpected error occurred while updating {Fore.CYAN}{key}:{Style.RESET_ALL} {e}"
            )

    @timer_function(verbose_func=lambda self: self.is_verbose())
    def chat(self, user_input) -> Tuple[str, Optional[SweetType]]:
        """
        Initiate a chat interaction with the LLM.

        Args:
            user_input (str): User input for the chat interaction.

        Returns:
            Tuple[str, Optional[SweetType]]: Tuple containing (response text, selected sweet type or None)
        """
        # Set the chatbot state to busy
        try:
            self.state_manager.set_state(ChatState.BUSY)

            # Update LLM settings
            self.update_llm_settings()

            # Call to LLM
            llm_output = self.chain.invoke(input={"input": user_input})

            # Extract response
            llm_response = llm_output["response"]
            if isinstance(llm_response, (list, dict)):
                llm_response = str(llm_response)

            parsed_response = self.parse_chat(llm_response)
            selection = None

            if len(parsed_response) > 1:
                llm_response = parsed_response[0]
                if parsed_response[1] == "ENDE KONVERSATION, AUSWAHL:":
                    # Get the text after "AUSWAHL:" and clean it
                    sweet_str = parsed_response[2].strip()
                    # Remove any punctuation from the end of the string
                    sweet_str = sweet_str.rstrip(".,!?")

                    # Get all possible variants
                    variants = SweetType.get_all_variants()

                    # Try to find any variant in the cleaned string
                    sweet_type = None
                    normalized_str = sweet_str.lower()
                    for variant, sweet in variants.items():
                        if variant.lower() in normalized_str:
                            sweet_type = sweet
                            break

                    if sweet_type is not None:
                        selection = sweet_type
                        print(
                            f"{Style.RESET_ALL}{Fore.GREEN}Selection: {sweet_type.display_name} "
                            f"(ID: {sweet_type.value}){Style.RESET_ALL}"
                        )
                    else:
                        print(
                            f"{Style.RESET_ALL}{Fore.YELLOW}Warning: Invalid sweet type received: "
                            f"{sweet_str}{Style.RESET_ALL}"
                        )

                    self.clear_memory()

                elif parsed_response[1] == "ABBRUCH KONVERSATION":
                    print(
                        f"{Style.RESET_ALL}{Fore.RED}Conversation cancelled{Style.RESET_ALL}"
                    )
                    self.clear_memory()

            self.state_manager.set_state(ChatState.IDLE)

            if self.verbose is True:
                print(self.memory.load_memory_variables({}))

            return llm_response, selection
        except Exception as e:
            self.state_manager.set_state(ChatState.ERROR, {"error": str(e)})
            raise

    def parse_chat(self, response):
        # Define the sequences to look for
        sequences = ["ENDE KONVERSATION, AUSWAHL:", "ABBRUCH KONVERSATION"]

        # Initialize variables
        segments = []
        start = 0

        # Check each sequence, assumption: only one sequence is present in the response
        for seq in sequences:
            idx = response.find(seq)
            if idx != -1:
                # If the sequence is found, split the string and add the segments to the list
                segments.append(response[start:idx].strip())
                segments.append(seq)
                start = idx + len(seq)

        # Add the last part of the response
        segments.append(response[start:].strip())

        return segments

    def clear_memory(self):
        """
        Clear the memory of the LLM instance.
        """
        try:
            print("Clearing memory...")
            self.memory.clear()
        except Exception as e:
            self.state_manager.set_state(
                ChatState.ERROR, {"error": f"Error clearing memory: {str(e)}"}
            )


if __name__ == "__main__":
    llm = LLM(verbose=False)
    llm.read_llm_settings()
    # llm.__update_parameter("temperature", 0.2)
    if True:
        request = "Hallo SweetPicker"
        print(f"Student: {request}")
        response = llm.chat(request)
        print(f"SweetPicker: {response}")

        request = "Kannst du mir eine empfehlen? such eins für mich aus"
        print(f"Student: {request}")
        response = llm.chat(request)
        print(f"SweetPicker: {response}")

        request = "Alles klar das nehme ich"
        print(f"Student: {request}")
        response = llm.chat(request)
        print(f"SweetPicker: {response}")

    # Clear memory test
    if True:
        request = "Hallo SweetPicker"
        print(f"Student: {request}")
        response = llm.chat(request)
        print(f"SweetPicker: {response}")

        request = "Kannst du mir eine empfehlen? such eins für mich aus"
        print(f"Student: {request}")
        response = llm.chat(request)
        print(f"SweetPicker: {response}")

        request = "Alles klar das nehme ich"
        print(f"Student: {request}")
        response = llm.chat(request)
        print(f"SweetPicker: {response}")
        llm.clear_memory()
        print("Memory cleared")
        request = "Welche Süßigkeiten hast du mir gerade gegeben?"
        print(f"Student: {request}")
        response = llm.chat(request)
        print(f"SweetPicker: {response}")
        llm.clear_memory()
        print("Memory cleared")
        request = "Hi"
        print(f"Student: {request}")
        response = llm.chat(request)
        print(f"SweetPicker: {response}")
        request = "einen Maoam bitte"
        print(f"Student: {request}")
        response = llm.chat(request)
        print(f"SweetPicker: {response}")

    while True:
        try:
            if content_in := input("Student: "):
                response = llm.chat(content_in)
                print(f"SweetPicker: {response}")
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt detected. Exiting...")
            break
