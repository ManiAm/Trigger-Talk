
import sounddevice as sd
import soundfile as sf
from collections import defaultdict
from scipy.signal import resample
import numpy as np


SKIP_PATTERNS = [
    "Steam Streaming",
    "Wave",
    "Mapper",
    "Primary Sound",
    "Virtual",
]

def get_default_input_device():

    default_input, _ = sd.default.device
    dev_info = get_device_info(default_input)
    return dev_info


def get_default_output_device():

    _, default_output = sd.default.device
    dev_info = get_device_info(default_output)
    return dev_info


def get_audio_devices():

    mic_only = defaultdict(list)
    speaker_only = defaultdict(list)
    input_output = defaultdict(list)

    devices = sd.query_devices()

    for idx, _ in enumerate(devices):

        dev_info = get_device_info(idx)
        if not dev_info:
            continue

        name = dev_info['name']
        hostapi_name = dev_info['hostapi_name']
        in_ch = dev_info['in_ch']
        out_ch = dev_info['out_ch']

        # Mic only
        if in_ch > 0 and out_ch == 0:
            mic_only[hostapi_name].append(dev_info)

        # Speaker only
        elif out_ch > 0 and in_ch == 0:
            speaker_only[hostapi_name].append(dev_info)

        # Both input & output
        elif in_ch > 0 and out_ch > 0:
            input_output[hostapi_name].append(dev_info)

    return mic_only, speaker_only, input_output


def get_device_info(device_index):

    try:

        dev = sd.query_devices(device_index)

        hostapi_names = get_hostapi_names()
        hostapi_name = hostapi_names.get(dev['hostapi'], f"Unknown ({dev['hostapi']})")

        return {
            "index": device_index,
            "name": dev['name'].strip(),
            "hostapi_name": hostapi_name,
            "in_ch": dev["max_input_channels"],
            "out_ch": dev["max_output_channels"],
            "rate": dev["default_samplerate"],
            "lat_in_low": dev["default_low_input_latency"],
            "lat_in_high": dev["default_high_input_latency"],
            "lat_out_low": dev["default_low_output_latency"],
            "lat_out_high": dev["default_high_output_latency"],
        }
    except Exception as e:
        print(f"❌ Failed to get device info: {e}")
        return None


def get_hostapi_names():

    hostapis = sd.query_hostapis()

    hostapi_names = {
        i: api['name'] for i, api in enumerate(hostapis)
    }

    return hostapi_names


def print_audio_devices():

    mic_only, speaker_only, input_output = get_audio_devices()

    print_audio_devices_group("🎤 Microphone-only Devices", mic_only)
    print_audio_devices_group("🔊 Speaker-only Devices", speaker_only)
    print_audio_devices_group("🎧 Devices with Both Input & Output", input_output)


def print_audio_devices_group(title, group):

    if not group:
        return

    print(f"\n{title}:\n")

    for hostapi, devices in group.items():

        print(f"    {hostapi}\n")

        for dev in devices:
            print(f"       [{dev['index']:>2}] {dev['name']}")
            print(f"            Input: {dev['in_ch']} ch | Output: {dev['out_ch']} ch")
            print(f"            Sample Rate: {dev['rate']} Hz")
            print(f"            Latency (in):  low={dev['lat_in_low']:.3f}s  high={dev['lat_in_high']:.3f}s")
            print(f"            Latency (out): low={dev['lat_out_low']:.3f}s  high={dev['lat_out_high']:.3f}s\n")


def select_best_microphone(mic_devices, input_output, preferred_hostapis=("Windows WASAPI", "Windows DirectSound")):

    if not mic_devices and not input_output:
        return None

    candidates = []

    # Flatten mic_devices dict into one list
    for _, devices in mic_devices.items():
        for dev in devices:
            in_ch = dev["in_ch"]
            if in_ch > 0:
                candidates.append(dev)

    # Flatten input_output dict into one list
    for _, devices in input_output.items():
        for dev in devices:
            in_ch = dev["in_ch"]
            if in_ch > 0:
                candidates.append(dev)

    # Prioritize preferred hostapis
    preferred = [d for d in candidates if d["hostapi_name"] in preferred_hostapis]
    fallback = [d for d in candidates if d["hostapi_name"] not in preferred_hostapis]

    def score(device):

        score = 0

        # Prefer more input channels
        score += device["in_ch"] * 10

        # Lower input latency = better
        score += max(0, 10 - device["lat_in_low"] * 100)

        # Prefer 44100 or 48000 Hz
        if device["rate"] in (44100.0, 48000.0):
            score += 5

        return score

    # Sort by score
    sorted_devices_preferred = sorted(preferred, key=score, reverse=True)
    sorted_devices_fallback = sorted(fallback, key=score, reverse=True)

    sorted_devices = sorted_devices_preferred + sorted_devices_fallback

    sorted_devices_phy = [d for d in sorted_devices if is_physical_mic(d)]

    for dev in sorted_devices_phy:

        try:
            with sd.InputStream(device=dev["index"], channels=1, samplerate=int(dev["rate"])):
                return dev["index"]
        except Exception as e:
            print(f"Skipping device [{dev['index']}] {dev['name']} — not usable: {e}")

    print("No available microphone devices found.")
    return None


def is_physical_mic(dev):

    return dev["in_ch"] > 0 and not any(skip in dev["name"] for skip in SKIP_PATTERNS)


def choose_blocksize(target_latency_ms, sample_rate):

    block_duration_sec = target_latency_ms / 1000
    blocksize = int(sample_rate * block_duration_sec)
    return blocksize


def resample_audio(audio_frames, input_rate=44100, target_rate=16000):

    num_samples = int(len(audio_frames) * target_rate / input_rate)
    return resample(audio_frames, num_samples).astype(np.int16)


def play_wav(filename):

    output_dev = sd.default.device[1]
    if output_dev is None or output_dev < 0:
        print("Error: invalid output device index")
        return

    data, samplerate = sf.read(filename, dtype='float32')

    # Normalize and limit volume
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.8  # cap at 80% volume to avoid clipping

    # Upmix mono to stereo
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)

    sd.play(data, samplerate, device=output_dev)
    sd.wait()  # Wait until audio is finished
