
import json
import threading
import asyncio
import logging
import socket
import getpass
import platform
import uvicorn
from pydantic import BaseModel
from typing import Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter

from hotword_models import HotwordModel
from hotword_types import MessageStatus, MessageType

logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(
    title="Hotword API",
    description="Hotword Detection and Transcription API.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

hw_obj = HotwordModel()

lock = asyncio.Lock()
running_lock = threading.Lock()


class ListenParams(BaseModel):
    dev_index: Optional[int]
    hotwords: List[str]
    model_engine_hotword: str
    model_name_hotword: Optional[str]
    model_engine_stt: str
    model_name_stt: Optional[str]
    target_latency: Optional[int] = 100
    silence_duration: Optional[int] = 3


async def send_message(websocket, msg_status, msg_type, msg):

    message = {
        "status": msg_status,
        "type": msg_type,
        "text": msg
    }

    try:
        await websocket.send_text(json.dumps(message))
    except Exception:
        pass


async def safe_close(ws: WebSocket):

    try:
        await ws.close()
    except RuntimeError:
        pass


@router.get("/health")
def health_check():

    return {"status": "ok"}


@router.websocket("/listen")
async def websocket_listen(websocket: WebSocket):

    await websocket.accept()

    if lock.locked():

        await send_message(
            websocket,
            MessageStatus.ERROR,
            MessageType.NOTIFICATION,
            "Another session is already running.")

        await safe_close(websocket)
        return

    async with lock:

        try:

            params_raw = await websocket.receive_text()
            params = ListenParams(**json.loads(params_raw))
            print(f"Received parameters from client: {params.model_dump()}")

            loop = asyncio.get_event_loop()

            #######

            host_info = {
                "hostname": socket.gethostname(),
                "username": getpass.getuser(),
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.machine()
            }

            host_info_str = json.dumps(host_info)
            asyncio.run_coroutine_threadsafe(
                send_message(websocket, MessageStatus.OK, MessageType.HOST_INFO, host_info_str),
                loop)

            #######

            def dev_input_callback(dev_info):
                dev_info_str = json.dumps(dev_info)
                asyncio.run_coroutine_threadsafe(
                    send_message(websocket, MessageStatus.OK, MessageType.DEV_INPUT, dev_info_str),
                    loop)

            status, output = await loop.run_in_executor(
                None,
                lambda: hw_obj.init_audio_device(
                    dev_index=params.dev_index,
                    dev_input_callback=dev_input_callback
                )
            )

            if not status:
                await send_message(
                    websocket,
                    MessageStatus.ERROR,
                    MessageType.NOTIFICATION,
                    f"init_audio_device failed: {output}")
                await safe_close(websocket)
                return

            await send_message(
                websocket,
                MessageStatus.OK,
                MessageType.NOTIFICATION,
                "Audio device initialized.")

            #######

            status, output = await loop.run_in_executor(
                None,
                lambda: hw_obj.init_hotword(
                    model_engine_hotword=params.model_engine_hotword,
                    model_name_hotword=params.model_name_hotword,
                    model_engine_stt=params.model_engine_stt,
                    model_name_stt=params.model_name_stt
                )
            )

            if not status:
                await send_message(
                    websocket,
                    MessageStatus.ERROR,
                    MessageType.NOTIFICATION,
                    f"init_hotword failed: {output}")
                await safe_close(websocket)
                return

            await send_message(
                websocket,
                MessageStatus.OK,
                MessageType.NOTIFICATION,
                "Hotword detection initialized. Listening...")

        except Exception as e:
            await send_message(
                websocket,
                MessageStatus.ERROR,
                MessageType.NOTIFICATION,
                str(e))
            await safe_close(websocket)
            hw_obj.stop_hotword_detection()
            return

        try:

            loop = asyncio.get_event_loop()

            def on_hotword(text):
                asyncio.run_coroutine_threadsafe(
                    send_message(websocket, MessageStatus.OK, MessageType.HOTWORD, text),
                    loop)

            def on_silence(text):
                asyncio.run_coroutine_threadsafe(
                    send_message(websocket, MessageStatus.OK, MessageType.SILENCE, text),
                    loop)

            def on_transcription(text):
                asyncio.run_coroutine_threadsafe(
                    send_message(websocket, MessageStatus.OK, MessageType.TRANSCRIBED, text),
                    loop)

            future = loop.run_in_executor(
                None,
                hw_obj.detect_hotword_and_transcribe,
                params.hotwords,
                on_hotword,
                on_silence,
                on_transcription,
                params.target_latency,
                params.silence_duration)

            while not future.done():

                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect:
                    print("Client disconnected.")
                    hw_obj.stop_hotword_detection()
                    break

            if not future.cancelled():
                status, output = await future
                if not status:
                    await send_message(
                        websocket,
                        MessageStatus.ERROR,
                        MessageType.NOTIFICATION,
                        output)

        except Exception as e:
            await send_message(
                websocket,
                MessageStatus.ERROR,
                MessageType.NOTIFICATION,
                str(e))

        finally:
            await safe_close(websocket)
            hw_obj.stop_hotword_detection()


@router.post("/stop")
def stop():

    if not running_lock.acquire(blocking=False):
        return JSONResponse({"error": "Another session is in progress"}, status_code=423)

    try:

        print("Stoping hotword detection...")
        hw_obj.stop_hotword_detection()

        return {"text": "Hotword detection stopped."}

    finally:

        running_lock.release()


app.include_router(router, prefix="/api/hotword")


if __name__ == "__main__":

    print("Starting Hotword detection service on http://localhost:5600")
    uvicorn.run(app, host="0.0.0.0", port=5600)
