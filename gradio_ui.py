import gradio as gr

# TODO: XTTS gets loaded twice??
if gr.NO_RELOAD:
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from threading import Thread
    import sys
    import time

    # Add the src/ directory to the sys.path so that we can import the LLM class
    sys.path.append("src/")
    from llm import LLM
    from config_reader import config_manager  # Import ConfigManager
    from speech_recognition import WhisperXProcessor
    from text_to_speech_generator import TTSHandler

USER_AVATAR = "images/thga_logo_head.png"
BOT_AVATAR = "images/robot_arm.png"
THEME = gr.themes.Soft()

DESCRIPTION = """
<div>
<h1 style="text-align: center;">SweetPicker Chatbot</h1>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/THGA-Logo.svg/2560px-THGA-Logo.svg.png" style="width: 70%; max-width: 250px; height: auto; opacity: 0.55; margin: auto;"> 
<p>🔎 For more details about the release and how to use the model <code>https://github.com/C-Sahin/chatbot_lm</code></p>
</div>
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/THGA-Logo.svg/2560px-THGA-Logo.svg.png" style="width: 70%; max-width: 550px; height: auto; opacity: 0.55; margin: auto;"> 
    <p style="font-size: 20px; margin-bottom: 2px; opacity: 0.65;">SP-Chatbot-V0.1</p>
</div>
"""

css = """
h1 {
  text-align: center;
  display: block;
}
"""

if gr.NO_RELOAD:
    llm = LLM(verbose=False)
    config_manager.set_setting("llm", "temperature", str(0.2))
    recognizer = WhisperXProcessor()
    recognizer.load_model()
    tts_handler = TTSHandler()
    tts_handler.load_model(
        xtts_checkpoint="/home/sahin/chatbot_lm/tts_train/run/training/GPT_XTTS_FT-May-31-2024_07+40PM-a682fa8d/best_model.pth",
        xtts_config="/home/sahin/chatbot_lm/tts_train/run/training/XTTS_v2.0_original_model_files/config.json",
        xtts_vocab="/home/sahin/chatbot_lm/tts_train/run/training/XTTS_v2.0_original_model_files/vocab.json",
    )
    # Fetch initial values from the config
    current_params = config_manager.get_all_settings("llm")
    mirostat_mapping = {
        "0": "0 (disabled)",
        "1": "1 (Mirostat)",
        "2": "2 (Mirostat 2.0)",
    }


def update_param_if_changed(param, value):
    # print(f"Checking parameter {param} with {value} for change")
    if current_params[param] != str(value):
        # print(f"Updating parameter {param} to {value}")
        current_params[param] = str(value)
        config_manager.set_setting("llm", param, str(value))
    else:
        # print(f"Parameter {param} already set to {value}")
        pass


def tts_interface_wrapper(text):
    print(f"speaking: {text}")
    out_path, _ = tts_handler.run_tts(
        lang="de",
        tts_text=text,
        speaker_audio_file="/home/sahin/chatbot_lm/tts_train/dataset/wavs/untitled enhanced_00000007.wav",
    )
    if out_path is None or not os.path.exists(out_path):
        print(f"Error: TTS output file does not exist at {out_path}")
    return out_path


def audio_interface_wrapper(audio):
    transcription, _ = recognizer.transcribe(audio)
    print(f"transcription: {transcription}")
    return transcription["segments"][0]["text"]


def chat_interface_wrapper(input_message, history=None):
    response = llm.chat(input_message)
    print(f"llm response: {response}")
    return response


# def chat_interface_wrapper(input_message):
#     response = llm.chat(input_message)
#     print(f"llm response: {response}")
#     return response


def chat_history_wrapper(input_message, chat_history):
    """
    Append the user input and LLM response to the chat history.
    """
    response = chat_interface_wrapper(input_message)
    chat_history.append((input_message, response))
    return chat_history, chat_history


# Gradio block for Text Chat
chatbot = gr.Chatbot(
    height=450,
    placeholder=PLACEHOLDER,
    label="SweetPicker Chatbot",
    avatar_images=[USER_AVATAR, BOT_AVATAR],  # type: ignore
    show_copy_button=False,
)

with gr.Blocks(fill_height=True, css=css) as chat:
    gr.Markdown(DESCRIPTION)
    output = gr.ChatInterface(
        fn=chat_interface_wrapper,
        theme=THEME,
        chatbot=chatbot,
        fill_height=True,
        examples=[
            ["Hallo SweetPicker."],
            ["Hallo, wer bist du?"],
            ["Was kannst du?"],
            ["Kannst du mir eine Süßigkeit empfehlen?"],
            ["Welche Süßigkeiten hast du zur Auswahl?"],
            ["Danke das nehme ich."],
        ],
        cache_examples=False,
        retry_btn=None,
        undo_btn=None,
        analytics_enabled=False,
        clear_btn="🗑️ Ansicht leeren",
        submit_btn="Senden",
    )
    reset_memory_btn = gr.Button("Verlauf löschen")
    reset_memory_btn.click(fn=llm.clear_memory).then(
        lambda: None, None, chatbot, queue=True
    )


# Gradio block for Voice Chat
voice_chatbot = gr.Chatbot(
    height=None,
    placeholder=PLACEHOLDER,
    label="SweetPicker Voice Chatbot history",
    avatar_images=[USER_AVATAR, BOT_AVATAR],  # type: ignore
    show_copy_button=False,
)
voice_chatbot_textbox = gr.Textbox(
    interactive=False,
    render=False,
    visible=False,
)
with gr.Blocks(theme=THEME) as voice:

    with gr.Row():
        input_audio = gr.Audio(
            label="User Request (Speech)",
            sources=["microphone"],
            type="filepath",
        )
        input_audio_transcribed = gr.Textbox(
            label="User Request (Read-only)",
            interactive=False,
            visible=False,
            render=False,
        )
        output_llm_readonly = gr.Textbox(
            label="LLM Response (Read-only)",
            interactive=False,
            visible=False,
            render=False,
        )
        output_tts = gr.Audio(
            label="LLM Response (Speech)",
            type="filepath",
            interactive=False,
            autoplay=True,
        )
    with gr.Column():
        voice_btn = gr.Button("Submit Voice")
    with gr.Column():
        chat_interface = gr.ChatInterface(
            fn=chat_history_wrapper,
            theme=THEME,
            chatbot=voice_chatbot,
            fill_height=True,
            cache_examples=False,
            retry_btn=None,
            undo_btn=None,
            analytics_enabled=False,
            clear_btn=None,
            submit_btn=None,
            textbox=voice_chatbot_textbox,
        )

    def process_voice(audio, history):
        transcription = audio_interface_wrapper(audio)
        llm_response = chat_interface_wrapper(transcription)
        tts_audio = tts_interface_wrapper(llm_response)
        history.append((transcription, llm_response))
        # print(f"history: {history}")
        # return transcription, llm_response, tts_audio, history
        return transcription, llm_response, tts_audio

    voice_btn.click(
        fn=process_voice,
        inputs=[input_audio, voice_chatbot],
        # outputs=[
        #     input_audio_transcribed,
        #     output_llm_readonly,
        #     output_tts,
        #     voice_chatbot,
        # ],
        outputs=[
            input_audio_transcribed,
            output_llm_readonly,
            output_tts,
            # voice_chatbot,
        ],
    )


# Gradio block for Streaming Voice Chat
voice_chatbot = gr.Chatbot(
    height=None,
    placeholder=PLACEHOLDER,
    label="SweetPicker Streaming Voice Chatbot history",
    avatar_images=[USER_AVATAR, BOT_AVATAR],  # type: ignore
    show_copy_button=False,
)
voice_chatbot_textbox = gr.Textbox(
    interactive=False,
    render=False,
    visible=False,
)
with gr.Blocks(theme=THEME) as voice_streaming:
    gr.Markdown("🎤 Voice Streaming Chatbot, WIP")
    # with gr.Row():
    #     input_audio = gr.Audio(
    #         label="User Request (Speech)",
    #         sources="microphone",
    #         type="filepath",
    #         streaming=True,
    #     )
    #     input_audio_transcribed = gr.Textbox(
    #         label="User Request (Read-only)",
    #         interactive=False,
    #         visible=False,
    #         render=False,
    #     )
    #     output_llm_readonly = gr.Textbox(
    #         label="LLM Response (Read-only)",
    #         interactive=False,
    #         visible=False,
    #         render=False,
    #     )
    #     output_tts = gr.Audio(
    #         label="LLM Response (Speech)",
    #         type="filepath",
    #         interactive=False,
    #         autoplay=True,
    #     )
    # with gr.Column():
    #     voice_btn = gr.Button("Submit Voice")
    # with gr.Column():
    #     chat_interface = gr.ChatInterface(
    #         fn=chat_history_wrapper,
    #         theme=THEME,
    #         chatbot=voice_chatbot,
    #         fill_height=True,
    #         cache_examples=False,
    #         retry_btn=None,
    #         undo_btn=None,
    #         analytics_enabled=False,
    #         clear_btn=None,
    #         submit_btn=None,
    #         textbox=voice_chatbot_textbox,
    #     )

    # def process_voice(audio, history):
    #     transcription = audio_interface_wrapper(audio)
    #     llm_response = chat_interface_wrapper(transcription)
    #     tts_audio = tts_interface_wrapper(llm_response)
    #     history.append((transcription, llm_response))
    #     return transcription, llm_response, tts_audio, history

    # voice_btn.click(
    #     fn=process_voice,
    #     inputs=[input_audio, voice_chatbot],
    #     outputs=[
    #         input_audio_transcribed,
    #         output_llm_readonly,
    #         output_tts,
    #         voice_chatbot,
    #     ],
    # )


with gr.Blocks(theme=THEME) as parameters:
    gr.Markdown("### Parameters")

    model = gr.Textbox(
        label="Model",
        value=current_params["model_name"],
        interactive=False,
    )
    mirostat = gr.Dropdown(
        label="Mirostat",
        choices=["0 (disabled)", "1 (Mirostat)", "2 (Mirostat 2.0)"],
        value=mirostat_mapping[current_params["mirostat"]],
        interactive=True,
        info="Enable Mirostat sampling for controlling perplexity. (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)",
    )
    mirostat_eta = gr.Slider(
        label="Mirostat Eta",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        value=float(current_params["mirostat_eta"]),
        interactive=True,
        info="Influences how quickly the algorithm responds to feedback from the generated text. (Default: 0.1)",
    )
    mirostat_tau = gr.Slider(
        label="Mirostat Tau",
        minimum=0.0,
        maximum=10.0,
        step=0.01,
        value=float(current_params["mirostat_tau"]),
        interactive=True,
        info="Controls the balance between coherence and diversity of the output. (Default: 5.0)",
    )
    num_ctx = gr.Slider(
        label="Context Window Size",
        minimum=2048,
        maximum=8192,
        step=1,
        value=int(current_params["num_ctx"]),
        interactive=True,
        info="Sets the size of the context window used to generate the next token. (Default: 4096)",
    )
    repeat_last_n = gr.Slider(
        label="Repeat Last N",
        minimum=-1,
        maximum=8192,
        step=1,
        value=int(current_params["repeat_last_n"]),
        interactive=True,
        info="Sets how far back for the model to look back to prevent repetition. (Default: 64, 0 = disabled, -1 = num_ctx)",
    )
    repeat_penalty = gr.Slider(
        label="Repeat Penalty",
        minimum=0.5,
        maximum=2.0,
        step=0.01,
        value=float(current_params["repeat_penalty"]),
        interactive=True,
        info="Sets how strongly to penalize repetitions. (Default: 1.1)",
    )
    temperature = gr.Slider(
        label="Temperature",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        value=float(current_params["temperature"]),
        interactive=True,
        info="The temperature of the model. (Default: 0.8)",
    )
    seed = gr.Number(
        label="Seed",
        value=int(current_params["seed"]),
        interactive=True,
        info="Sets the random number seed to use for generation. (Default: 0)",
    )
    stop = gr.Textbox(
        label="Stop Sequences",
        value=current_params["stop"],
        interactive=True,
        info="Sets the stop sequences to use. Separate with commas. e.g. 'Stop-word1, Stop-word2'",
    )
    tfs_z = gr.Slider(
        label="TFS Z",
        minimum=0.0,
        maximum=5.0,
        step=0.1,
        value=float(current_params["tfs_z"]),
        interactive=True,
        info="Tail free sampling is used to reduce the impact of less probable tokens. (Default: 1.0)",
    )
    num_predict = gr.Slider(
        label="Num Predict",
        minimum=0,
        maximum=2048,
        step=1,
        value=int(current_params["num_predict"]),
        interactive=True,
        info="Maximum number of tokens to predict. (Default: 128, -1 = infinite generation, -2 = fill context)",
    )
    top_k = gr.Slider(
        minimum=0,
        maximum=100,
        step=1,
        label="Top K",
        value=int(current_params["top_k"]),
        interactive=True,
        info="Reduces the probability of generating nonsense. (Default: 40)",
    )
    top_p = gr.Slider(
        label="Top P",
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        value=float(current_params["top_p"]),
        interactive=True,
        info="Works together with top-k. (Default: 0.9)",
    )

    mirostat.change(
        fn=lambda x: update_param_if_changed("mirostat", int(x.split()[0])),
        inputs=mirostat,
    )

    mirostat_eta.change(
        fn=lambda x: update_param_if_changed("mirostat_eta", x), inputs=mirostat_eta
    )
    mirostat_tau.change(
        fn=lambda x: update_param_if_changed("mirostat_tau", x), inputs=mirostat_tau
    )
    num_ctx.change(fn=lambda x: update_param_if_changed("num_ctx", x), inputs=num_ctx)
    repeat_last_n.change(
        fn=lambda x: update_param_if_changed("repeat_last_n", x), inputs=repeat_last_n
    )
    repeat_penalty.change(
        fn=lambda x: update_param_if_changed("repeat_penalty", x), inputs=repeat_penalty
    )
    temperature.change(
        fn=lambda x: update_param_if_changed("temperature", x), inputs=temperature
    )
    seed.change(fn=lambda x: update_param_if_changed("seed", x), inputs=seed)
    stop.change(fn=lambda x: update_param_if_changed("stop", x), inputs=stop)
    tfs_z.change(fn=lambda x: update_param_if_changed("tfs_z", x), inputs=tfs_z)
    num_predict.change(
        fn=lambda x: update_param_if_changed("num_predict", x), inputs=num_predict
    )
    top_k.change(fn=lambda x: update_param_if_changed("top_k", x), inputs=top_k)
    top_p.change(fn=lambda x: update_param_if_changed("top_p", x), inputs=top_p)


with gr.Blocks(theme=THEME, title="SweetPicker Chatbot") as demo:
    gr.TabbedInterface(
        [chat, voice, voice_streaming, parameters],  # type: ignore
        ["💬 Chat", "🗣️ Voice Chat", "🎤 Voice Streaming", "⚙️ Parameters"],
    )

if __name__ == "__main__":
    demo.launch(share=True)

# Reload mode with `gradio gradio_ui.py`
