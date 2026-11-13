# Solution — Exercise 03: Streaming ASR and TTS Walkthrough

> Paired learning file: [`lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-03-streaming-asr-and-tts-walkthrough.md`](https://github.com/ai-infra-curriculum/ai-infra-inference-engineer-learning/blob/main/lessons/mod-112-multimodal-and-emerging-serving/exercises/exercise-03-streaming-asr-and-tts-walkthrough.md)
> Module: `mod-112-multimodal-and-emerging-serving`
> Type: implementation / walkthrough

## 1. Solution overview

The learner stands up two speech services end-to-end:

1. A **streaming ASR** endpoint that accepts microphone-shaped audio chunks and
   emits partial + final transcripts as speech arrives.
2. A **streaming TTS** endpoint that accepts text and emits audio frames as
   they are synthesized, not after the full utterance is rendered.

The deliverables show both halves: bidirectional streaming wire shapes, chunk
sizing choices, and a small latency report that names (a) end-of-utterance
transcription latency for ASR and (b) first-audio-frame latency for TTS.

Three artefacts:

| Artefact | Form | Pass bar |
|---|---|---|
| A — Streaming ASR server | Script + launch command | Accepts audio chunks (e.g. 20 ms Opus frames or 320-sample PCM chunks at 16 kHz) over WebSocket or gRPC; emits partials during speech and a final result at end-of-utterance |
| B — Streaming TTS server | Script + launch command | Accepts a text payload; emits PCM/Opus audio frames as they are synthesized; a client can start playback before the server has finished |
| C — Latency note | Short write-up | Reports ASR end-of-utterance latency and TTS time-to-first-audio-frame, with chunk size and model named; discusses the latency floor set by the chosen chunk size |

### Why streaming is architecturally different

A non-streaming speech service is a request/response blob exchange: client
sends a file, server returns a transcript or an audio blob. That shape is
**wrong for human-in-the-loop systems**. For ASR, the human expects to see
intermediate transcripts as they speak; for TTS, the human expects audio to
start playing before the sentence is fully synthesized. Both require the
server to produce output while input is still arriving.

Three properties change at the plumbing level:

1. **Transport.** HTTP request/response is replaced by WebSocket, gRPC
   bidirectional streams, or Server-Sent Events (SSE) for the server-to-client
   half. SSE is one-way (server → client), so it works for TTS but not for the
   client-sends-microphone-chunks half of ASR.
2. **Audio chunking.** Audio is not a single payload; it is a sequence of fixed-
   duration chunks (commonly 20 ms frames for VoIP/Opus, or 320-sample /
   640-sample windows at 16 kHz for ASR models). Chunk size sets the latency
   floor: a 100 ms window cannot possibly surface partial results in under
   100 ms.
3. **Model shape.** Streaming-capable ASR models are either (a) natively
   streaming encoders (RNN-T, CTC, Conformer-T with a limited lookahead),
   (b) chunk-attended encoders with a bounded right-context window, or (c)
   large non-streaming models (e.g. Whisper) wrapped by a **streaming runtime**
   (faster-whisper + VAD + sliding-window). The third is the common practitioner
   path for open-weights.

Streaming TTS is simpler in principle — the model emits frames in order — but
implementation varies by model family (autoregressive mel/token → vocoder,
diffusion-based, flow-matching). The common practitioner win is to start
streaming the vocoder output as mel frames become available, rather than
waiting for the full mel.

## 2. Implementation — worked walkthrough

The walkthrough uses `faster-whisper` for ASR (Whisper via CTranslate2, with a
VAD-wrapped streaming adapter) and `Coqui XTTS-v2` or `F5-TTS` for TTS. Both
are open-weights and widely deployed as practitioner references. Any
functionally equivalent stack (NVIDIA Riva, SeamlessStreaming, Parakeet, Piper,
Bark, Kokoro, StyleTTS2, OpenAI TTS-compatible backends) follows the same
pattern through its own config file.

### 2.1 Streaming ASR server (WebSocket, faster-whisper)

```python
# asr_server.py
import asyncio, io, json
import numpy as np
import soundfile as sf
import websockets
from faster_whisper import WhisperModel

MODEL = "Systran/faster-whisper-large-v3"   # pin
SAMPLE_RATE = 16000
CHUNK_MS = 500                               # one inference window
BUFFER_MS = 2000                             # sliding context
VAD_SILENCE_MS = 700                         # end-of-utterance trigger

model = WhisperModel(MODEL, device="cuda", compute_type="float16")

async def handle(ws):
    """Client sends 16 kHz, 16-bit, mono PCM frames (little-endian) as binary
    messages. Server emits JSON messages: {"type":"partial","text":...} and
    {"type":"final","text":...}."""
    pcm = bytearray()
    silent_ms = 0

    async for msg in ws:
        if isinstance(msg, (bytes, bytearray)):
            pcm.extend(msg)
        else:
            if msg == "flush":
                await _emit_final(ws, pcm)
                return
            continue

        # If we have at least CHUNK_MS of new audio, re-transcribe the last BUFFER_MS
        ms_held = (len(pcm) // 2) * 1000 // SAMPLE_RATE
        if ms_held < CHUNK_MS:
            continue

        window_bytes = pcm[-2 * SAMPLE_RATE * BUFFER_MS // 1000:]
        audio = np.frombuffer(bytes(window_bytes), dtype=np.int16).astype(np.float32) / 32768.0

        segments, info = model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            beam_size=1,
            without_timestamps=False,
        )
        text = " ".join(s.text.strip() for s in segments)
        await ws.send(json.dumps({"type": "partial", "text": text}))

        # Naive silence detection on the tail; replace with a real VAD for production.
        tail = audio[-SAMPLE_RATE * 500 // 1000:]
        if float(np.sqrt(np.mean(tail**2))) < 0.005:
            silent_ms += CHUNK_MS
            if silent_ms >= VAD_SILENCE_MS:
                await ws.send(json.dumps({"type": "final", "text": text}))
                pcm.clear()
                silent_ms = 0
        else:
            silent_ms = 0

async def _emit_final(ws, pcm):
    audio = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(audio, language="en", beam_size=1)
    text = " ".join(s.text.strip() for s in segments)
    await ws.send(json.dumps({"type": "final", "text": text}))

async def main():
    async with websockets.serve(handle, "0.0.0.0", 8030, max_size=2**20):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
```

Notes on the shape:

- The server re-transcribes a **sliding window** on each new chunk. For large
  Whisper this is expensive and imposes a floor on how often partials can fire;
  practitioner alternatives include natively-streaming models (Parakeet,
  SeamlessStreaming, FastConformer-RNNT via NVIDIA Riva) that process audio
  incrementally without re-reading the window.
- `vad_filter=True` uses faster-whisper's built-in VAD to suppress silence.
  Production deployments typically add a real VAD (Silero, WebRTC VAD) in
  front and use it to drive the final / end-of-utterance signal; the naive
  RMS check above is a placeholder, not a model.
- `beam_size=1` is a deliberate latency choice. Beam search improves accuracy
  at a latency cost that is often not worth it for streaming partials.

### 2.2 Streaming TTS server (SSE or WebSocket)

```python
# tts_server.py — FastAPI with Server-Sent Events for audio frames
import asyncio, io, json, base64
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from TTS.api import TTS                       # Coqui-TTS; pin to a release

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"   # pin
tts = TTS(MODEL).to("cuda")
SAMPLE_RATE = 24000
FRAME_MS = 20                                # 20 ms Opus-compatible frames
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000

app = FastAPI()

class TTSReq(BaseModel):
    text: str
    voice: str = "en"
    language: str = "en"

def frame_iter(audio_tensor_or_array):
    """Yield audio in SAMPLES_PER_FRAME chunks as it becomes available."""
    import numpy as np
    arr = np.asarray(audio_tensor_or_array, dtype="float32")
    for i in range(0, len(arr), SAMPLES_PER_FRAME):
        yield arr[i:i + SAMPLES_PER_FRAME].tobytes()

@app.post("/tts/stream")
def tts_stream(req: TTSReq):
    # XTTS-v2 exposes an incremental streaming generator; older models return
    # the full waveform and the server slices it into frames. Both paths yield
    # the same wire shape; the latency difference is in time-to-first-frame.
    def gen():
        for chunk_waveform in tts.tts_stream(
            text=req.text,
            language=req.language,
            speaker=req.voice,
        ):
            for frame in frame_iter(chunk_waveform):
                yield (b"event: audio\ndata: "
                       + base64.b64encode(frame) + b"\n\n")
        yield b"event: end\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
```

Notes on the shape:

- The server emits **fixed-size frames** at a cadence the downstream player can
  consume. Variable-size chunks force the player to re-buffer and introduce
  jitter.
- The wire format matters. PCM-float32 is easy to slice but 2× the bytes of
  int16 and 4× the bytes of Opus. For production, encode to Opus or 16-bit PCM
  at the server; the walkthrough uses float32 for illustrative simplicity.
- If the underlying model only returns a full waveform (`tts.tts(text)`),
  wrapping it in a frame iterator still produces a *streaming wire shape* but
  **not a streaming latency profile**. Time-to-first-frame will equal
  time-to-full-synthesis. Say this explicitly in the latency note.

### 2.3 Client — ASR

```python
# asr_client.py — stream a WAV file as if it were the microphone
import asyncio, json, sys
import soundfile as sf
import websockets

async def main(path):
    data, sr = sf.read(path, dtype="int16")
    assert sr == 16000, "ASR server expects 16 kHz"
    chunk_samples = 16000 * 500 // 1000   # 500 ms chunks

    async with websockets.connect("ws://localhost:8030") as ws:
        async def send():
            for i in range(0, len(data), chunk_samples):
                await ws.send(data[i:i + chunk_samples].tobytes())
                await asyncio.sleep(0.5)  # emulate real-time arrival
            await ws.send("flush")

        async def recv():
            async for m in ws:
                print(json.loads(m))

        await asyncio.gather(send(), recv())

asyncio.run(main(sys.argv[1]))
```

### 2.4 Client — TTS

```python
# tts_client.py
import base64, httpx, time

t0 = time.perf_counter()
first = None
with httpx.stream("POST", "http://localhost:8031/tts/stream",
                  json={"text": "Streaming TTS is working end to end.",
                        "voice": "en", "language": "en"},
                  timeout=60) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and first is None:
            first = (time.perf_counter() - t0) * 1000
            print(f"time-to-first-audio-frame: {first:.0f} ms")
        if line.startswith("event: end"):
            break
```

### 2.5 Latency note — what to report

The latency note is the artefact a reviewer reads first. It names, at minimum:

- **Chunk / frame size.** ASR window size (e.g. 500 ms transcribe window);
  TTS frame size (e.g. 20 ms).
- **Model and backend.** `Systran/faster-whisper-large-v3` + CTranslate2 fp16;
  `tts_models/multilingual/multi-dataset/xtts_v2`. Pin both.
- **ASR metrics.** Time-from-end-of-speech to final-transcript (`finalize
  latency`); mean partial-update latency (time from a 500 ms boundary to the
  partial being emitted).
- **TTS metrics.** Time-to-first-audio-frame (TTFF), and sustained real-time
  factor (synthesis duration / audio duration).
- **Floor acknowledgement.** The latency floor is structural: TTFF cannot be
  lower than the first synthesized frame's wall-clock cost; final-transcript
  latency cannot be lower than one window's worth of inference. Pushing below
  the floor requires model changes (natively-streaming ASR, lookahead-limited
  decoder) or hardware changes.

## 3. Validation steps

1. **ASR server accepts raw PCM chunks over WebSocket and emits JSON
   partials.** Running `asr_client.py` on a sample `.wav` produces at least one
   `{"type":"partial"}` message per 500 ms chunk and one final at the end.
2. **TTS server emits audio frames over SSE (or WebSocket).** Running
   `tts_client.py` prints a TTFF measurement and receives frames until an
   `event: end`.
3. **Models and backend versions pinned.** Both server files reference an
   explicit model identifier and the serving package version is pinned in
   `requirements.txt` or an image tag.
4. **Chunk / frame sizes documented and consistent.** ASR window size and TTS
   frame size are stated in the latency note and match the server configuration.
5. **Latency report includes floor acknowledgement.** The note names where the
   measured latency sits relative to the structural floor, not just a single
   number.
6. **Honest streaming for TTS.** If the underlying model does not stream
   internally, the note says so and TTFF is not reported as if streaming were
   real.
7. **Reproducibility.** A reviewer can start both servers from the pinned
   commands and reproduce the client-side numbers on the same hardware class.

## 4. Rubric / review checklist

Score each item 0 (missing), 1 (partial), 2 (fully correct). Passing: ≥ 10 / 14.

| # | Item | Points |
|---|---|---|
| 1 | Streaming ASR server accepts audio chunks and emits partials + final | 2 |
| 2 | Streaming TTS server emits audio frames progressively, not as one blob | 2 |
| 3 | Transport choice (WebSocket / gRPC / SSE) is justified for each direction | 2 |
| 4 | Chunk sizes stated; latency floor acknowledged | 2 |
| 5 | Models and backend versions pinned; sample rate / frame rate documented | 2 |
| 6 | Latency note reports ASR finalize latency and TTS TTFF with method | 2 |
| 7 | Reviewer can reproduce from scripts + launch commands without edits | 2 |

## 5. Common mistakes

- **Using Server-Sent Events for the ASR upload half.** SSE is one-way server →
  client; it cannot carry client-side audio chunks. For bidirectional streaming
  use WebSocket or gRPC. SSE is fine for the TTS download half.
- **Running Whisper per-chunk without a sliding context.** Each new chunk run
  in isolation produces fragmented, repeated, or missing-boundary transcripts.
  The window of past audio matters. Either keep a sliding buffer (as above) or
  use a natively-streaming model.
- **Claiming "streaming TTS" when the model is non-streaming.** Slicing a full
  waveform into 20 ms frames after synthesis is a streaming *wire shape* but
  not streaming latency. The note must say which.
- **Variable-size audio frames.** Some models emit output in arbitrary chunk
  lengths; passing those directly to a downstream player introduces jitter.
  Re-frame to fixed-size frames before emitting.
- **Using float32 PCM over the wire for production.** Fine for the walkthrough,
  but 4× the bandwidth of int16 and ~16× Opus. For production, encode to Opus
  at the server.
- **Ignoring the ASR end-of-utterance signal.** Without a VAD or
  silence-threshold rule, the client cannot know when to stop sending or when
  to use the final transcript. Fire an explicit `{"type":"final"}` at
  end-of-utterance.
- **Beam-searching streaming partials.** Beam search's accuracy gain is often
  dwarfed by its latency cost for intermediate transcripts. Use greedy or
  beam=1 for partials; beam search, if any, belongs on the final re-pass.
- **Comparing ASR WER on streaming vs. offline modes without naming the window
  size.** Streaming WER is strictly ≥ offline WER by construction (less
  right-context available). Report both and name the window.
- **Running the TTS and ASR servers on the same accelerator in a demo without
  acknowledging the contention.** Co-locating two models can fit on one GPU
  with margin, but the first demo load often shows surprisingly degraded
  latency because both models contend for SM time. Measure on isolated GPUs
  first, then co-locate.

## 6. References

- Radford, A. et al. *Robust Speech Recognition via Large-Scale Weak
  Supervision.* arXiv:2212.04356 — Whisper, the open-weights ASR used in the
  walkthrough.
- SYSTRAN. *faster-whisper.* https://github.com/SYSTRAN/faster-whisper —
  practitioner implementation (CTranslate2 backend) used in §2.1.
- Casanova, E. et al. *XTTS: A Massively Multilingual Zero-Shot Text-to-Speech
  Model.* arXiv:2406.04904 — the open TTS model referenced for streaming
  synthesis.
- Coqui-TTS project. https://github.com/coqui-ai/TTS — practitioner serving
  reference for the TTS half.
- NVIDIA Riva. *Streaming ASR and TTS reference architecture.*
  https://docs.nvidia.com/deeplearning/riva/ — alternative practitioner
  implementation at production scale; useful as a cross-reference for VAD and
  Opus wire framing.
- OpenAI Realtime API. https://platform.openai.com/docs/guides/realtime —
  reference wire shape for bidirectional streaming speech in production.
- RFC 6716 — Definition of the Opus Audio Codec.
  https://datatracker.ietf.org/doc/html/rfc6716 — 20 ms Opus frame structure
  referenced in §2.2.
- [owasp-ml-top-10](https://owasp.org/www-project-machine-learning-security-top-10/)
  — voice cloning and deepfake-audio risk sits adjacent to TTS deployments;
  governance covered under [[server-vs-edge-decision-drill]], not this
  walkthrough.
- [mitre-atlas](https://atlas.mitre.org/) — speech-based prompt injection /
  voice-liveness attacks appear under input-manipulation; not directly invoked
  here.
- [nist-ai-rmf](https://www.nist.gov/itl/ai-risk-management-framework) —
  governance baseline for speech services handling user audio.
