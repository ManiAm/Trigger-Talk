
import sys
import asyncio
import websockets
import json

async def main():

    uri = "ws://localhost:5600/api/hotword/listen"

    async with websockets.connect(uri) as websocket:

        # example for Vosk (not very accurate)
        hotword_params = {
            "dev_index": None,
            "hotwords": ["hey jarvis", "hey agent"],
            "model_engine_hotword": "vosk",
            "model_name_hotword": "vosk-model-en-us-0.22",
            "model_engine_stt": "openai_whisper",
            "model_name_stt": "small.en",
            "target_latency": 100,
            "silence_duration": 2,
            "hotword_audio": "bell_1.wav",
            "silence_audio": "bell_1.wav"
        }

        # example for Pvporcupine (commercial)
        hotword_params = {
            "dev_index": None,
            "hotwords": ["hey agent", "bumblebee"],
            "model_engine_hotword": "pvporcupine",
            "model_name_hotword": None,
            "model_engine_stt": "openai_whisper",
            "model_name_stt": "small.en",
            "target_latency": 100,
            "silence_duration": 2,
            "hotword_audio": "bell_1.wav",
            "silence_audio": "bell_1.wav"
        }

        # example for Openwakeword (accurate and free)
        hotword_params = {
            "dev_index": None,
            "hotwords": ["hey_jarvis"],
            "model_engine_hotword": "openwakeword",
            "model_name_hotword": None,
            "model_engine_stt": "openai_whisper",
            "model_name_stt": "small.en",
            "target_latency": 80,
            "silence_duration": 2,
            "hotword_audio": "jarvis/at_your_service.wav",
            "silence_audio": "bell_1.wav"
        }

        print("Setting up hotword detection. Please wait...", flush=True)
        await websocket.send(json.dumps(hotword_params))

        while True:

            try:
                msg = await websocket.recv()
                print("SERVER:", msg, flush=True)
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.", flush=True)
                break


def run():

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nForced shutdown.")
        sys.exit(1)


if __name__ == "__main__":

    run()
