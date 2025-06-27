
import os
import gc
import struct
import sounddevice as sd
from dotenv import load_dotenv
import pvporcupine

import utility

load_dotenv()


class PvporcupineEngine:

    def __init__(self):

        self.pvporcupine_model = None

        self.dev_index = None
        self.dev_sample_rate = None
        self.dev_channels = None
        self.access_key = None

        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        sound_dir = os.path.join(script_dir, "sounds")
        self.hotword_sound = os.path.join(sound_dir, "bell_1.wav")

        custom_keyword_path_dir = os.path.join(script_dir, "pvporcupine_keywords")

        my_keyword_path = {
            "hey agent": os.path.join(custom_keyword_path_dir, "hey-agent_en_linux_v3_0_0.ppn")
        }

        my_keyword_path = {
            k: v for k, v in my_keyword_path.items()
            if os.path.isfile(v)
        }

        self.keyword_path_all = {**my_keyword_path, **pvporcupine.KEYWORD_PATHS}


    def init_model(self, model_name, dev_index, dev_sample_rate, dev_channels):

        if model_name:
            return False, f"'{model_name}' is not a valid model in Pvporcupine."

        self.dev_index = dev_index
        self.dev_sample_rate = dev_sample_rate
        self.dev_channels = dev_channels

        self.access_key = os.getenv('Pvporcupine_API_KEY', None)
        if not self.access_key:
            return False, "Environment varibale 'Pvporcupine_API_KEY' is not set."

        return True, None


    def start_hotword_detection(self, hotword_list, target_latency_ms, script_state, on_hotword_callback=None):

        _ = target_latency_ms

        keyword_paths = []
        for hotword in hotword_list:
            if hotword not in self.keyword_path_all:
                return False, f"invalid keyword '{hotword}'. Choose from {list(self.keyword_path_all.keys())}"
            keyword_paths.append(self.keyword_path_all[hotword])

        try:

            self.pvporcupine_model = pvporcupine.create(
                access_key=self.access_key,
                keyword_paths=keyword_paths
            )

        except Exception as e:
            return False, f"Pvporcupine Create failed: {e}"

        frame_length = self.pvporcupine_model.frame_length
        sample_rate = self.pvporcupine_model.sample_rate

        with sd.RawInputStream(
            device=self.dev_index,
            samplerate=sample_rate,
            blocksize=frame_length,
            dtype='int16',
            channels=1) as stream:

            while not script_state["interrupted"]:

                data, _ = stream.read(frame_length)

                # Convert raw bytes to a list of 16-bit samples
                audio_frame = struct.unpack_from("h" * frame_length, data)

                keyword_index = self.pvporcupine_model.process(audio_frame)

                if keyword_index >= 0:

                    detected_word = hotword_list[keyword_index]

                    if on_hotword_callback:
                        on_hotword_callback(detected_word)

                    utility.play_wav(self.hotword_sound)

                    return True, None

        return True, None


    def stop_hotword_detection(self):

        if self.pvporcupine_model:
            self.pvporcupine_model.delete()
            self.pvporcupine_model = None

        gc.collect()
