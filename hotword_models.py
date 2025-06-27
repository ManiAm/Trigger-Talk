
import os
import sys
import time
import tempfile
import wave
import webrtcvad
import sounddevice as sd
import numpy as np
from scipy.signal import resample

import utility
import config
from speech_to_text_api import STT_REST_API_Client

from engine_vosk import VoskEngine
from engine_openwakeword import OpenwakewordEngine
from engine_pvporcupine import PvporcupineEngine

MODELS = {
    "vosk": VoskEngine(),
    "openwakeword": OpenwakewordEngine(),
    "pvporcupine": PvporcupineEngine()
}


class HotwordModel():

    def __init__(self):

        self.stt_client = STT_REST_API_Client(url=config.speech_to_text_url)

        if not self.stt_client.check_health():
            print("SST service is not reachable")
            sys.exit(1)

        self.input_dev_index = None
        self.input_dev_sample_rate = None
        self.input_dev_channels = None

        self.model_handler = None
        self.script_state = {"interrupted": False}

        self.vad = webrtcvad.Vad(3)
        self.target_rate = 16000


    def init_hotword(
        self,
        dev_index=None,
        model_engine_hotword="vosq",
        model_name_hotword="vosk-model-en-us-0.22",
        model_engine_stt="openai_whisper",
        model_name_stt="small.en"):

        status, output = self.__init_input_device(dev_index)
        if not status:
            return False, output

        status, output = self.__init_engine_hotword(model_engine_hotword, model_name_hotword)
        if not status:
            return False, output

        status, output = self.__init_engine_stt(model_engine_stt, model_name_stt)
        if not status:
            return False, output

        return True, None


    def __init_input_device(self, dev_index):

        dev_info_default = utility.get_default_input_device()

        if dev_info_default:

            print(
                f"\nDefault input device: "
                f"[{dev_info_default['index']}] "
                f"{dev_info_default['name']} "
                f"(hostapi: {dev_info_default['hostapi_name']})"
            )

        if not dev_index:

            mic_only, _, _ = utility.get_audio_devices()

            dev_index = utility.select_best_microphone(mic_only)
            if dev_index is None:
                return False, "__init_input_device: No suitable input device found."

        dev_info = utility.get_device_info(dev_index)
        if not dev_info:
            return False, f"Cannot obtain device info with index {dev_index}."

        print(f'\nUsing input device: [{dev_info["index"]}] {dev_info["name"]} (hostapi: {dev_info["hostapi_name"]})')

        self.input_dev_index = dev_index
        self.input_dev_sample_rate = int(dev_info["rate"])
        self.input_dev_channels = dev_info["in_ch"]

        return True, None


    def __init_engine_hotword(self, model_engine_hotword, model_name_hotword):

        if model_engine_hotword not in MODELS:
            return False, f"Engine '{model_engine_hotword}' not supported"

        self.model_handler = MODELS[model_engine_hotword]

        try:
            return self.model_handler.init_model(
                model_name_hotword,
                self.input_dev_index,
                self.input_dev_sample_rate,
                self.input_dev_channels)
        except Exception as e:
            return False, f"Failed to init model: {str(e)}"


    def __init_engine_stt(self, model_engine_stt, model_name_stt):

        print(f"\n🔄 Loading {model_engine_stt} model '{model_name_stt}'...")

        return self.stt_client.load_model(model_engine_stt, model_name_stt)


    def stop_hotword_detection(self):

        print("Stopping hotword detection...")

        self.script_state["interrupted"] = True

        try:

            # wait for loops to terminate
            time.sleep(3)

            if self.model_handler:
                self.model_handler.stop_hotword_detection()

        finally:

            self.script_state["interrupted"] = False


    def detect_hotword_and_transcribe(
        self,
        hotword_list,
        on_hotword_callback=None,
        on_transcription_callback=None,
        target_latency_ms=100,
        silence_duration_s=3):

        if self.input_dev_index is None or self.model_handler is None:
            return False, "hotword detection is not initialized!"

        hotword_list = [x.lower() for x in hotword_list]

        while not self.script_state["interrupted"]:

            print(f"\nListening for hotwords '{hotword_list}'...")

            # blocking call until hotword is detected
            status, output = self.model_handler.start_hotword_detection(
                hotword_list,
                target_latency_ms,
                self.script_state,
                on_hotword_callback)

            if not status:
                return False, output

            if not self.script_state["interrupted"]:

                callback, audio_frames = self.__record_callback(silence_duration=silence_duration_s)

                blocksize = utility.choose_blocksize(
                    target_latency_ms,
                    self.input_dev_sample_rate)

                with sd.RawInputStream(
                    device=self.input_dev_index,
                    samplerate=self.input_dev_sample_rate,
                    blocksize=blocksize,
                    dtype='int16',
                    channels=self.input_dev_channels,
                    callback=callback) as stream:

                    print("Recording started...")
                    while stream.active:
                        sd.sleep(100)

                if audio_frames:

                    status, output = self.__recording_done_callback(audio_frames)
                    if not status:
                        return False, output
                    elif on_transcription_callback:
                        on_transcription_callback(output)

        return True, None


    def __record_callback(self, frame_duration_ms=30, silence_duration=3):

        frame_size = int(self.input_dev_sample_rate * frame_duration_ms / 1000) * 2  # in bytes

        buffer = bytearray()
        audio_frames = []
        silence_start = None

        def callback(indata, frames, time_info, status):

            nonlocal buffer, silence_start, audio_frames

            pcm = bytes(indata)
            buffer.extend(pcm)
            audio_frames.append(pcm)

            while not self.script_state["interrupted"] and len(buffer) >= frame_size:
                frame_bytes = bytes(buffer[:frame_size])
                buffer = buffer[frame_size:]

                if self.__is_silence(frame_bytes, self.input_dev_sample_rate, frame_duration_ms):
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        raise sd.CallbackStop()
                else:
                    silence_start = None

        return callback, audio_frames


    def __is_silence(self, pcm_bytes, original_rate, frame_duration_ms):

        # Convert bytes to int16 numpy array
        pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Compute how many samples are expected after resampling
        original_samples = len(pcm_data)
        target_samples = int(original_samples * self.target_rate / original_rate)

        # Resample to self.target_rate (16000 Hz)
        resampled = resample(pcm_data, target_samples).astype(np.int16)
        resampled_bytes = resampled.tobytes()

        # Ensure frame length matches allowed durations
        expected_bytes = int(self.target_rate * frame_duration_ms / 1000) * 2
        if len(resampled_bytes) != expected_bytes:
            return True  # Treat bad frames as silence

        return not self.vad.is_speech(resampled_bytes, self.target_rate)


    def __recording_done_callback(self, audio_frames):

        print("Recording stopped due to silence")

        temp_file = None

        try:

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.input_dev_channels)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(self.input_dev_sample_rate)
                wf.writeframes(b''.join(audio_frames))

            temp_file.close()

            print("Sending audio to backend for transcription...")

            status, output = self.stt_client.transcribe_file(temp_file.name, "openai_whisper", "small.en")
            if not status:
                return False, output

            return True, output.get("transcript", "")

        except Exception as e:
            return False, str(e)

        finally:

            if temp_file and os.path.exists(temp_file.name):
                os.remove(temp_file.name)
