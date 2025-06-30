
import sys
import json
import queue
import gc
import sounddevice as sd
from vosk import Model, KaldiRecognizer

import utility


class VoskEngine:

    def __init__(self):

        self.q = queue.Queue(maxsize=50)

        self.vosk_model = None
        self.vosk_recognizer = None

        self.dev_index = None
        self.dev_sample_rate = None
        self.dev_channels = None


    def init_model(self, model_name, dev_index, dev_sample_rate, dev_channels):

        self.dev_index = dev_index
        self.dev_sample_rate = dev_sample_rate
        self.dev_channels = dev_channels

        print(f"\n🔄 Loading Vosk model '{model_name}'...")

        try:

            self.vosk_model = Model(model_name=model_name)
            self.vosk_recognizer = KaldiRecognizer(self.vosk_model, self.dev_sample_rate)
            self.vosk_recognizer.SetWords(True) # enable word-level recognition output

        except Exception as e:
            return False, str(e)

        return True, None


    def start_hotword_detection(self, hotword_list, target_latency_ms, script_state, hotword_audio, on_hotword_callback=None):

        blocksize = utility.choose_blocksize(target_latency_ms, self.dev_sample_rate)

        self.__empty_queue()

        with sd.RawInputStream(
            device=self.dev_index,
            samplerate=self.dev_sample_rate,
            blocksize=blocksize,
            dtype='int16',
            channels=self.dev_channels,
            callback=self.__audio_callback):

            while not script_state["interrupted"]:

                data = self.q.get()

                if self.vosk_recognizer.AcceptWaveform(data):

                    result = json.loads(self.vosk_recognizer.Result())
                    text = result.get("text", "").lower()

                    if not text:
                        continue

                    print(f"[VOICE] {text}")

                    for word in hotword_list:

                        if word in text:

                            print(f"🔊 Hotword detected: {word}")

                            if on_hotword_callback:
                                on_hotword_callback(word)

                            if hotword_audio:
                                utility.play_wav(hotword_audio)

                            return True, None

                else:
                    # partial results:
                    # partial = json.loads(recognizer.PartialResult())["partial"]
                    pass

        return True, None


    def stop_hotword_detection(self):

        self.vosk_recognizer = None
        self.vosk_model = None
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
