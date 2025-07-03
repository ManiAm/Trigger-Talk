
from enum import Enum


class MessageStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


class MessageType(str, Enum):
    NOTIFICATION = "Notification"
    HOST_INFO = "Host Info"
    DEV_INPUT = "Device Input"
    HOTWORD = "Hotword"
    SILENCE = "Silence"
    TRANSCRIBED = "Transcribed"
