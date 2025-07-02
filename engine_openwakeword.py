
import os
import sys
import queue
import gc
import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model

import utility


class OpenwakewordEngine:

    def __init__(self):

        self.q = queue.Queue(maxsize=50)

        self.openwakeword_model = None

        self.dev_index = None
        self.dev_sample_rate = None
        self.dev_channels = None

        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        custom_keyword_path_dir = os.path.join(script_dir, "openwakeword_keywords")

        my_keyword_path = {
            "hey agent": os.path.join(custom_keyword_path_dir, "my_model.tflite")
        }

        my_keyword_path = {
            k: v for k, v in my_keyword_path.items()
            if os.path.isfile(v)
        }

        model_paths = { k: v["model_path"] for k, v in openwakeword.MODELS.items() }
        self.keyword_path_all = {**my_keyword_path, **model_paths}


    def init_model(self, model_name, dev_index, dev_sample_rate, dev_channels):

        if model_name:
            return False, f"'{model_name}' is not a valid model in Openwakeword."

        self.dev_index = dev_index
        self.dev_sample_rate = dev_sample_rate
        self.dev_channels = dev_channels

        print(f"\n🔄 Loading Openwakeword models...")

        try:

            # One-time download of all pre-trained models
            openwakeword.utils.download_models()

        except Exception as e:
            return False, str(e)

        return True, None


    def start_hotword_detection(self, hotword_list, target_latency_ms, script_state, on_hotword_callback=None):

        keyword_paths = []
        for hotword in hotword_list:
            if hotword not in self.keyword_path_all:
                return False, f"invalid keyword '{hotword}'. Choose from {list(self.keyword_path_all.keys())}"
            keyword_paths.append(self.keyword_path_all[hotword])

        self.openwakeword_model = Model(wakeword_models=keyword_paths)

        sample_rate = 16000  # OpenWakeWord expects 16kHz audio
        blocksize = utility.choose_blocksize(target_latency_ms, sample_rate)

        self.__empty_queue()
        detected_hotword = None

        with sd.RawInputStream(
            device=self.dev_index,
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype='int16',
            channels=1,
            callback=self.__audio_callback):

            while not script_state["interrupted"]:

                data = self.q.get()

                if len(data) < blocksize * 2:  # 2 bytes per sample
                    continue

                audio_frame = np.frombuffer(data, dtype=np.int16)

                predictions  = self.openwakeword_model.predict(audio_frame)

                filtered = sorted(
                    [(name, score) for name, score in predictions.items() if score >= 0.5],
                    key=lambda x: x[1],
                    reverse=True
                )

                if filtered:

                    if len(filtered) > 1:
                        print(f"Multiple hotwords scored high: {filtered}")

                    name, score = filtered[0]
                    print(f"🔊 Hotword detected: {name} (score: {score:.2f})")

                    detected_hotword = name
                    break

        if detected_hotword:
            if on_hotword_callback:
                on_hotword_callback(detected_hotword)

        return True, None


    def stop_hotword_detection(self):

        if self.openwakeword_model:
            self.openwakeword_model = None

        gc.collect()


    def __audio_callback(self, indata, frames, time_info, status):

        if status:
            print(f"[STATUS] {status}", file=sys.stderr)

        try:
            self.q.put_nowait(bytes(indata))
        except queue.Full:
            print("[WARN] Audio queue full — dropping frame")


    def __empty_queue(self):

        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
