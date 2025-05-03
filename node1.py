


import os
import argparse
import asyncio
import base64
import hashlib
import hmac
import json
import wave
from datetime import datetime, timezone
from urllib.parse import urlencode

import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
from pydub import AudioSegment
from dotenv import load_dotenv
import ollama
from ollama._types import ResponseError

# === Load credentials from .env ===
load_dotenv()
APP_ID     = os.getenv("IFLY_APPID")
API_KEY    = os.getenv("IFLY_APIKEY")
API_SECRET = os.getenv("IFLY_APISECRET")

# === Endpoint & streaming config ===
HOST           = "ist-api-sg.xf-yun.com"
PATH           = "/v2/ist"
CHUNK_INTERVAL = 0.1  # seconds per slice (100 ms)

# === CLI args ===
parser = argparse.ArgumentParser(description="iFLYTEK Real-Time ASR with LLM Correction")
parser.add_argument("audio", help="Path to your audio file (mp3, wav, etc.)")
parser.add_argument(
    "--lang",
    choices=["mandarin", "cantonese", "english"],
    default="mandarin",
    help="Language of the audio"
)
args = parser.parse_args()

# === Language → iFLYTEK codes ===
LANG_MAP = {
    "mandarin":  ("zh_cn", "mandarin"),
    "cantonese": ("zh_cn", "cantonese"),
    "english":   ("en_us",  "general")
}
language, accent = LANG_MAP[args.lang]

def ensure_wav(path: str) -> str:
    base, ext = os.path.splitext(path)
    if ext.lower() != ".wav":
        out = base + "_16k_mono.wav"
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(out, format="wav")
        return out
    return path

def generate_ws_url() -> str:
    date   = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
    origin = f"host: {HOST}\ndate: {date}\nGET {PATH} HTTP/1.1"
    sha    = hmac.new(API_SECRET.encode(), origin.encode(), hashlib.sha256).digest()
    sig    = base64.b64encode(sha).decode()
    auth   = (
        f'api_key="{API_KEY}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{sig}"'
    )
    auth_b64 = base64.b64encode(auth.encode()).decode()
    qs = urlencode({"authorization": auth_b64, "date": date, "host": HOST})
    return f"wss://{HOST}{PATH}?{qs}"

async def live_transcription():
    wav_file = ensure_wav(args.audio)
    url = generate_ws_url()
    print(f"▶ Connecting to ASR:\n  {url}\n")

    corrected_chunks = []

    try:
        async with websockets.connect(url) as ws:
            # 1) Handshake
            await ws.send(json.dumps({
                "common":  {"app_id": APP_ID},
                "business": {"language": language, "domain": "ist_open", "accent": accent},
                "data":    {"status": 0, "format": "audio/L16;rate=16000", "encoding": "raw"}
            }))

            # 2) Stream audio
            with wave.open(wav_file, "rb") as wf:
                rate   = wf.getframerate()
                frames = int(rate * CHUNK_INTERVAL)
                while True:
                    raw = wf.readframes(frames)
                    if not raw:
                        break

                    await ws.send(json.dumps({
                        "data": {
                            "status":   1,
                            "format":   "audio/L16;rate=16000",
                            "encoding": "raw",
                            "audio":    base64.b64encode(raw).decode()
                        }
                    }))
                    await asyncio.sleep(CHUNK_INTERVAL)

                    # receive partial ASR
                    try:
                        resp = await asyncio.wait_for(ws.recv(), timeout=0.2)
                        msg  = json.loads(resp)
                        result = msg.get("data", {}).get("result")
                        if result:
                            chunk_text = "".join(
                                cw["w"]
                                for part in result["ws"]
                                for cw in part["cw"]
                            )
                            print(f"Detected text by ASR: {chunk_text}")

                            # LLM correction using the correct model name
                            try:
                                llm = ollama.chat(
                                    model="deepseek-r1:1.5b",
                                    messages=[
                                        {"role":"system","content":
                                            "You are a transcription correction assistant. "
                                            "Output ONLY the corrected text, no explanations."},
                                        {"role":"user","content":
                                            f"Correct this transcript chunk:\n\n{chunk_text}"}
                                    ]
                                )
                                corrected = llm["message"]["content"].strip()
                            except ResponseError as e:
                                print(f"⚠️ LLM error: {e}. Using ASR text.")
                                corrected = chunk_text

                            corrected_chunks.append(corrected)
                            print(f"Corrected text by LLM: {corrected}\n")

                    except asyncio.TimeoutError:
                        pass

            # 3) Signal end-of-stream
            await ws.send(json.dumps({"data": {"status": 2}}))

    except (ConnectionClosedOK, ConnectionClosedError) as e:
        print(f"WebSocket closed: {e}")

    # 4) Save corrected output only
    out_path = "corrected_transcript.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for line in corrected_chunks:
            f.write(line + "\n")

    print(f"Corrected transcript saved to {out_path}")

if __name__ == "__main__":
    asyncio.run(live_transcription())
    #####
    

