# chatbot_lm

## Inhaltsverzeichnis

- [chatbot\_lm](#chatbot_lm)
  - [Inhaltsverzeichnis](#inhaltsverzeichnis)
  - [Überblick](#überblick)
  - [Features](#features)
  - [Installation](#installation)
  - [Verwendung](#verwendung)
  - [Anpassung](#anpassung)
  - [Lizenz](#lizenz)
  - [Kontakt](#kontakt)

## Überblick

Dieses Projekt befasst sich mit dem Entwurf eines modularen, ROS-integrierten Offline-Chatbots. Es ist speziell für den Einsatz mit dem SweetPicker konzipiert und nutzt Large Language Models (LLMs) in Verbindung mit Frameworks wie Langchain und Ollama, um eine natürliche menschliche Konversation zu führen.

## Features

- **ROS-Integration:** Die Integration mit dem Robot Operating System (ROS) ermöglicht die Kommunikation mit anderen Roboterkomponenten.
- **Modularität:** Durch die Verwendung von LLMs ist die Verwertung bis zu einem gewissen Grad modular (z.B. Use-Case: Süßigkeiten greifen -> andere Gegenstände greifen).
- **Offline-Fähigkeit:** Der Chatbot kann offline betrieben werden und benötigt lediglich zum Setup eine Internetverbindung.
- **LLM-Nutzung:** Die Verwendung von Large Language Models ermöglicht eine natürliche Interaktion mit dem Benutzer.
- **Unterstützung von Frameworks:** Das Projekt verwendet Frameworks wie Langchain und Ollama für eine effiziente Verarbeitung natürlicher Sprache. Alle in Ollama eingepflegten Modelle können problemlos eingesetzt werden.

## Installation

1. Klonen Sie das Repository: `git clone https://github.com/your_username/chatbot_lm.git`
2. Navigieren Sie in das Projektverzeichnis: `cd chatbot_lm`
3. Optional: Einrichten einer virtuellen Umgebung.
4. Installieren Sie ffmpeg `sudo apt update && sudo apt install ffmpeg`
5. Installieren Sie pyaudio `sudo apt-get install portaudio19-dev; sudo apt install python3-pyaudio`
6. Installieren Sie die erforderlichen Pakete: `pip install -r requirements.txt`
7. Installieren Sie die ROS-Version: [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
8. pip install git+https://github.com/m-bain/whisperx.git (v 1.0.0)
9. Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so} sudo apt-get -y install cudnn9-cuda-12 (solution: sudo apt-get install libcudnn8, pip install nvidia-cudnn-cu11)

## Verwendung

**Voraussetzungen für OOB experience:**

- Total VRAM Usage: 14~16GiB. <12GiB möglich mit stärker quantisierten Modellen
- NVIDIA GPU mit > 16GB VRAM (Oder kleinere Modelle wählen, siehe: [config.ini](config.ini))
  - ~8800MiB für [llama3:8b-instruct-q8_0](https://ollama.com/library/llama3:8b-instruct-q8_0) 2k Tokens context window
  - ~9100MiB für [llama3:8b-instruct-q8_0](https://ollama.com/library/llama3:8b-instruct-q8_0) 4k Tokens context window
  - ~9900MiB für [llama3:8b-instruct-q8_0](https://ollama.com/library/llama3:8b-instruct-q8_0) 8k Tokens context window
  - ~4300MiB für [whisperx, large-v3](https://github.com/m-bain/whisperX) Batch size of 16 and float16 computation
  - ~2100MiB für [whisperx, large-v3](https://github.com/m-bain/whisperX) Batch size of 4 and int8 computation
  - ~2700MiB für [xTTS-V2](https://huggingface.co/coqui/XTTS-v2)

- [CUDA installation V12](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu)
    Getestet mit
  - Cuda compilation tools, release 12.0, V12.0.140
  - Build cuda_12.0.r12.0/compiler.32267302_0
- [Ollama installation](https://ollama.com/download)
- [cuDNN installation](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- FFMPEG (ffmpeg version >= 6.1.1)

## Anpassung

Um den Chatbot an individuelle Anforderungen anzupassen, können Sie die Konfigurationsdateien bearbeiten (Siehe: [Config](config.ini), [Config Reader](config_reader.py), [State Manager](state_manager.py)) oder zusätzliche Module entwickeln. Eine Anleitung oder ein Leitfaden zur Anpassung des Chatbots kann hier bereitgestellt werden.

## Lizenz

Das Projekt ist unter der MIT-Lizenz lizenziert. Weitere Informationen finden Sie in der [LICENSE](LICENSE.md).

## Kontakt

Für Fragen, Anregungen oder Probleme können Sie mich unter [E-Mail](mailto:tnyy6j4i@duck.com) kontaktieren.
