import os
import sys
import threading  # Keep threading for SpeechPipelineManager internals and AbortWorker
import time
import json
import struct
import asyncio
import uvicorn
import logging
from typing import Any, Dict
from contextlib import asynccontextmanager
from starlette.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from datetime import datetime
from queue import Empty
from .colors import Colors
from .logsetup import setup_logging
from .upsample_overlap import UpsampleOverlap

# Import modules that might affect logging first
from .speech_pipeline_manager import SpeechPipelineManager
from .audio_in import AudioInputProcessor

# Setup logging after all imports
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

# Logging is now properly configured with custom formatter
logger.info("🖥️👋 Welcome to local real-time voice chat")


class ServerConfig:
    """Centralized configuration with environment fallbacks."""

    def __init__(self) -> None:
        self.use_ssl: bool = False
        self.tts_engine: str = os.getenv("TTS_ENGINE", "kyutai")
        self.tts_orpheus_model: str = "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf"
        self.llm_model: str = os.getenv("LLM_MODEL", "qwen3-8b-abliterated")
        self.llm_provider: str = os.getenv("LLM_PROVIDER", "openai")
        self.openai_base_url: str = os.getenv(
            "OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
        self.no_think: bool = True

    @property
    def direct_stream(self) -> bool:
        return self.tts_engine in ["orpheus", "kyutai"]


CFG = ServerConfig()

USE_SSL = CFG.use_ssl
TTS_ENGINE = CFG.tts_engine
TTS_ORPHEUS_MODEL = CFG.tts_orpheus_model
LLM_MODEL = CFG.llm_model
LLM_PROVIDER = CFG.llm_provider
OPENAI_BASE_URL = CFG.openai_base_url
NO_THINK = CFG.no_think
DIRECT_STREAM = CFG.direct_stream

logger.info(
    f"🖥️⚙️ {Colors.apply('[PARAM]').blue} Starting engine: {Colors.apply(TTS_ENGINE).blue}")
logger.info(
    f"🖥️⚙️ {Colors.apply('[PARAM]').blue} Direct streaming: {Colors.apply('ON' if DIRECT_STREAM else 'OFF').blue}")

# Define the maximum allowed size for the incoming audio queue
try:
    MAX_AUDIO_QUEUE_SIZE = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", 50))
    logger.info(
        f"🖥️⚙️ {Colors.apply('[PARAM]').blue} Audio queue size limit set to: {Colors.apply(str(MAX_AUDIO_QUEUE_SIZE)).blue}")
except ValueError:
    logger.warning(
        "🖥️⚠️ Invalid MAX_AUDIO_QUEUE_SIZE env var. Using default: 50")
    MAX_AUDIO_QUEUE_SIZE = 50


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# from handlerequests import LanguageProcessor
# from audio_out import AudioOutProcessor

LANGUAGE = "en"
# TTS_FINAL_TIMEOUT = 0.5 # unsure if 1.0 is needed for stability
TTS_FINAL_TIMEOUT = 1.0  # unsure if 1.0 is needed for stability

# --------------------------------------------------------------------
# Custom no-cache StaticFiles
# --------------------------------------------------------------------


class NoCacheStaticFiles(StaticFiles):
    """
    Serves static files without allowing client-side caching.

    Overrides the default Starlette StaticFiles to add 'Cache-Control' headers
    that prevent browsers from caching static assets. Useful for development.
    """

    async def get_response(self, path: str, scope: Dict[str, Any]) -> Response:
        """
        Gets the response for a requested path, adding no-cache headers.

        Args:
            path: The path to the static file requested.
            scope: The ASGI scope dictionary for the request.

        Returns:
            A Starlette Response object with cache-control headers modified.
        """
        response: Response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        # These might not be strictly necessary with no-store, but belt and suspenders
        if "etag" in response.headers:
            response.headers.__delitem__("etag")
        if "last-modified" in response.headers:
            response.headers.__delitem__("last-modified")
        return response

# --------------------------------------------------------------------
# Lifespan management
# --------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan, initializing and shutting down resources.

    Initializes global components like SpeechPipelineManager, Upsampler, and
    AudioInputProcessor and stores them in `app.state`. Handles cleanup on shutdown.

    Args:
        app: The FastAPI application instance.
    """
    logger.info("🖥️▶️ Server starting up")
    # Initialize global components, not connection-specific state
    app.state.SpeechPipelineManager = SpeechPipelineManager(
        tts_engine=TTS_ENGINE,
        llm_provider=LLM_PROVIDER,
        llm_model=LLM_MODEL,
        no_think=NO_THINK,
        orpheus_model=TTS_ORPHEUS_MODEL,
        llm_base_url=OPENAI_BASE_URL,
    )

    app.state.Upsampler = UpsampleOverlap()
    app.state.AudioInputProcessor = AudioInputProcessor(
        LANGUAGE,
        is_orpheus=TTS_ENGINE in ["orpheus", "kyutai"],
        pipeline_latency=app.state.SpeechPipelineManager.full_output_pipeline_latency / 1000,  # seconds
    )
    # Keep this? Its usage isn't clear in the provided snippet. Minimizing changes.
    app.state.Aborting = False

    yield

    logger.info("🖥️⏹️ Server shutting down")
    app.state.AudioInputProcessor.shutdown()

# --------------------------------------------------------------------
# FastAPI app instance
# --------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with no cache
app.mount("/static", NoCacheStaticFiles(directory="src/static"), name="static")


@app.get("/favicon.ico")
async def favicon():
    """
    Serves the favicon.ico file.

    Returns:
        A FileResponse containing the favicon.
    """
    return FileResponse("src/static/favicon.ico")


@app.get("/")
async def get_index() -> HTMLResponse:
    """
    Serves the main index.html page.

    Reads the content of static/index.html and returns it as an HTML response.

    Returns:
        An HTMLResponse containing the content of index.html.
    """
    with open("src/static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------


def parse_json_message(text: str) -> dict:
    """
    Safely parses a JSON string into a dictionary.

    Logs a warning if the JSON is invalid and returns an empty dictionary.

    Args:
        text: The JSON string to parse.

    Returns:
        A dictionary representing the parsed JSON, or an empty dictionary on error.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("🖥️⚠️ Ignoring client message with invalid JSON")
        return {}


def format_timestamp_ns(timestamp_ns: int) -> str:
    """
    Formats a nanosecond timestamp into a human-readable HH:MM:SS.fff string.

    Args:
        timestamp_ns: The timestamp in nanoseconds since the epoch.

    Returns:
        A string formatted as hours:minutes:seconds.milliseconds.
    """
    # Split into whole seconds and the nanosecond remainder
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000

    # Convert seconds part into a datetime object (local time)
    dt = datetime.fromtimestamp(seconds)

    # Format the main time as HH:MM:SS
    time_str = dt.strftime("%H:%M:%S")

    # For instance, if you want milliseconds, divide the remainder by 1e6 and format as 3-digit
    milliseconds = remainder_ns // 1_000_000
    formatted_timestamp = f"{time_str}.{milliseconds:03d}"

    return formatted_timestamp

# --------------------------------------------------------------------
# WebSocket data processing
# --------------------------------------------------------------------


async def process_incoming_data(ws: WebSocket, app: FastAPI, incoming_chunks: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    Receives messages via WebSocket, processes audio and text messages.

    Handles binary audio chunks, extracting metadata (timestamp, flags) and
    putting the audio PCM data with metadata into the `incoming_chunks` queue.
    Applies back-pressure if the queue is full.
    Parses text messages (assumed JSON) and triggers actions based on message type
    (e.g., updates client TTS state via `callbacks`, clears history, sets speed).

    Args:
        ws: The WebSocket connection instance.
        app: The FastAPI application instance (for accessing global state if needed).
        incoming_chunks: An asyncio queue to put processed audio metadata dictionaries into.
        callbacks: The TranscriptionCallbacks instance for this connection to manage state.
    """
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]

                # Ensure we have at least an 8‑byte header: 4 bytes timestamp_ms + 4 bytes flags
                if len(raw) < 8:
                    logger.warning(
                        "🖥️⚠️ Received packet too short for 8‑byte header.")
                    continue

                # Unpack big‑endian uint32 timestamp (ms) and uint32 flags
                timestamp_ms, flags = struct.unpack("!II", raw[:8])
                client_sent_ns = timestamp_ms * 1_000_000

                # Build metadata using fixed fields
                metadata = {
                    "client_sent_ms":           timestamp_ms,
                    "client_sent":              client_sent_ns,
                    "client_sent_formatted":    format_timestamp_ns(client_sent_ns),
                    "isTTSPlaying":             bool(flags & 1),
                }

                # Record server receive time
                server_ns = time.time_ns()
                metadata["server_received"] = server_ns
                metadata["server_received_formatted"] = format_timestamp_ns(
                    server_ns)

                # The rest of the payload is raw PCM bytes
                metadata["pcm"] = raw[8:]

                # Check queue size before putting data
                current_qsize = incoming_chunks.qsize()
                if current_qsize < MAX_AUDIO_QUEUE_SIZE:
                    # Now put only the metadata dict (containing PCM audio) into the processing queue.
                    await incoming_chunks.put(metadata)
                else:
                    # Queue is full, drop the chunk and log a warning
                    logger.warning(
                        f"🖥️⚠️ Audio queue full ({current_qsize}/{MAX_AUDIO_QUEUE_SIZE}); dropping chunk. Possible lag."
                    )

            elif "text" in msg and msg["text"]:
                # Text-based message: parse JSON
                data = parse_json_message(msg["text"])
                msg_type = data.get("type")
                logger.info(Colors.apply(f"🖥️📥 ←←Client: {data}").orange)

                if msg_type == "tts_start":
                    logger.info("🖥️ℹ️ Received tts_start from client.")
                    # Update connection-specific state via callbacks
                    callbacks.tts_client_playing = True
                elif msg_type == "tts_stop":
                    logger.info("🖥️ℹ️ Received tts_stop from client.")
                    # Update connection-specific state via callbacks
                    callbacks.tts_client_playing = False
                # Add to the handleJSONMessage function in server.py
                elif msg_type == "clear_history":
                    logger.info("🖥️ℹ️ Received clear_history from client.")
                    app.state.SpeechPipelineManager.reset()
                elif msg_type == "set_speed":
                    speed_value = data.get("speed", 0)
                    speed_factor = speed_value / 100.0  # Convert 0-100 to 0.0-1.0
                    if app.state.AudioInputProcessor.transcriber:
                        turn_detection = app.state.AudioInputProcessor.transcriber.turn_detection
                        if turn_detection:
                            turn_detection.update_settings(speed_factor)
                            logger.info(
                                f"🖥️⚙️ Updated turn detection settings to factor: {speed_factor:.2f}")
                    else:
                        logger.info(
                            "🖥️⚙️ Cannot update speed: speech recognition disabled")
                elif msg_type == "text_input":
                    # Handle text-only input without transcription
                    text_content = data.get("content", "").strip()
                    if text_content:
                        logger.info(
                            f"🖥️💬 Received text input: '{text_content}'")
                        # Process text input directly through the callbacks
                        await process_text_input(app, callbacks, text_content)
                elif msg_type == "toggle_speech":
                    # Handle speech recognition toggle
                    enabled = data.get("enabled", False)
                    logger.info(
                        f"🖥️🎙️ Speech recognition {'enabled' if enabled else 'disabled'}")
                    await handle_speech_toggle(app, callbacks, enabled)

    except asyncio.CancelledError:
        pass  # Task cancellation is expected on disconnect
    except WebSocketDisconnect as e:
        logger.warning(
            f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in process_incoming_data: {repr(e)}")
    except RuntimeError as e:  # Often raised on closed transports
        logger.error(
            f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in process_incoming_data: {repr(e)}")
    except Exception as e:
        logger.exception(
            f"🖥️💥 {Colors.apply('EXCEPTION').red} in process_incoming_data: {repr(e)}")


async def process_text_input(app: FastAPI, callbacks: 'TranscriptionCallbacks', text_content: str) -> None:
    """
    Processes text input directly without transcription, mimicking the voice input flow.

    Simulates the transcription pipeline by directly calling the appropriate callbacks
    for partial transcription, potential sentence detection, and final transcription.

    Args:
        app: The FastAPI application instance.
        callbacks: The TranscriptionCallbacks instance for this connection.
        text_content: The text content to process.
    """
    try:
        # Send as partial transcription first
        callbacks.on_partial(text_content)

        # Simulate a small delay to mimic real transcription timing
        await asyncio.sleep(0.1)

        # Trigger potential sentence detection (this may start generation)
        callbacks.on_potential_sentence(text_content)

        # Another small delay
        await asyncio.sleep(0.1)

        # Mark as potential final
        callbacks.on_potential_final(text_content)

        # Small delay before final processing
        await asyncio.sleep(0.1)

        # Process as final transcription (this triggers the full response flow)
        # Empty audio bytes since this is text input
        callbacks.on_before_final(b"", text_content)
        callbacks.on_final(text_content)

        logger.info(f"🖥️✅ Text input processed successfully: '{text_content}'")

    except Exception as e:
        logger.exception(f"🖥️💥 Error processing text input: {repr(e)}")


async def handle_speech_toggle(app: FastAPI, callbacks: 'TranscriptionCallbacks', enabled: bool) -> None:
    """
    Handles speech recognition toggle by enabling/disabling audio processing.

    When speech is disabled, the AudioInputProcessor stops processing audio chunks
    and STT models can be unloaded to save resources.

    Args:
        app: The FastAPI application instance.
        callbacks: The TranscriptionCallbacks instance for this connection.
        enabled: Whether speech recognition should be enabled.
    """
    try:
        # Update connection-specific speech state
        callbacks.speech_enabled = enabled

        if enabled:
            logger.info("🖥️🎙️ Enabling speech recognition for this connection")
            # Enable speech recognition
            app.state.AudioInputProcessor.enable_speech_recognition()
            # Apply pending callbacks if transcriber was just created
            app.state.AudioInputProcessor._apply_pending_callbacks()
        else:
            logger.info(
                "🖥️🎙️ Disabling speech recognition for this connection")
            # Disable speech recognition
            app.state.AudioInputProcessor.disable_speech_recognition()

        logger.info(
            f"🖥️✅ Speech toggle processed successfully: {'enabled' if enabled else 'disabled'}")

    except Exception as e:
        logger.exception(f"🖥️💥 Error handling speech toggle: {repr(e)}")


async def send_text_messages(ws: WebSocket, message_queue: asyncio.Queue) -> None:
    """
    Continuously sends text messages from a queue to the client via WebSocket.

    Waits for messages on the `message_queue`, formats them as JSON, and sends
    them to the connected WebSocket client. Logs non-TTS messages.

    Args:
        ws: The WebSocket connection instance.
        message_queue: An asyncio queue yielding dictionaries to be sent as JSON.
    """
    try:
        while True:
            await asyncio.sleep(0.001)  # Yield control
            data = await message_queue.get()
            msg_type = data.get("type")
            if msg_type != "tts_chunk":
                logger.info(Colors.apply(f"🖥️📤 →→Client: {data}").orange)
            await ws.send_json(data)
    except asyncio.CancelledError:
        pass  # Task cancellation is expected on disconnect
    except WebSocketDisconnect as e:
        logger.warning(
            f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in send_text_messages: {repr(e)}")
    except RuntimeError as e:  # Often raised on closed transports
        logger.error(
            f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in send_text_messages: {repr(e)}")
    except Exception as e:
        logger.exception(
            f"🖥️💥 {Colors.apply('EXCEPTION').red} in send_text_messages: {repr(e)}")


async def _reset_interrupt_flag_async(app: FastAPI, callbacks: 'TranscriptionCallbacks'):
    """
    Resets the microphone interruption flag after a delay (async version).

    Waits for 1 second, then checks if the AudioInputProcessor is still marked
    as interrupted. If so, resets the flag on both the processor and the
    connection-specific callbacks instance.

    Args:
        app: The FastAPI application instance (to access AudioInputProcessor).
        callbacks: The TranscriptionCallbacks instance for the connection.
    """
    await asyncio.sleep(1)
    # Check the AudioInputProcessor's own interrupted state
    if app.state.AudioInputProcessor.interrupted:
        logger.info(
            f"{Colors.apply('🖥️🎙️ ▶️ Microphone continued (async reset)').cyan}")
        app.state.AudioInputProcessor.interrupted = False
        # Reset connection-specific interruption time via callbacks
        callbacks.interruption_time = 0
        logger.info(Colors.apply(
            "🖥️🎙️ interruption flag reset after TTS chunk (async)").cyan)


async def send_tts_chunks(app: FastAPI, message_queue: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    Continuously sends TTS audio chunks from the SpeechPipelineManager to the client.

    Monitors the state of the current speech generation (if any) and the client
    connection (via `callbacks`). Retrieves audio chunks from the active generation's
    queue, upsamples/encodes them, and puts them onto the outgoing `message_queue`
    for the client. Handles the end-of-generation logic and state resets.

    Args:
        app: The FastAPI application instance (to access global components).
        message_queue: An asyncio queue to put outgoing TTS chunk messages onto.
        callbacks: The TranscriptionCallbacks instance managing this connection's state.
    """
    try:
        logger.info("🖥️🔊 Starting TTS chunk sender")
        last_quick_answer_chunk = 0
        last_chunk_sent = 0
        prev_status = None

        # --- Pacing config (simple realtime queue) ---
        SR = 24000  # Kyutai sample rate (mono, 16-bit)
        PREBUFFER_SEC = 0.10  # small prebuffer to avoid initial stutter
        pacing_started = False
        pacing_start_monotonic = 0.0
        samples_emitted = 0
        prebuffer_chunks: list[str] = []
        prebuffer_samples: int = 0

        while True:
            await asyncio.sleep(0.001)  # Yield control

            # Use connection-specific interruption_time via callbacks
            if app.state.AudioInputProcessor.interrupted and callbacks.interruption_time and time.time() - callbacks.interruption_time > 2.0:
                app.state.AudioInputProcessor.interrupted = False
                callbacks.interruption_time = 0  # Reset via callbacks
                logger.info(Colors.apply(
                    "🖥️🎙️ interruption flag reset after 2 seconds").cyan)

            is_tts_finished = app.state.SpeechPipelineManager.is_valid_gen(
            ) and app.state.SpeechPipelineManager.running_generation.audio_quick_finished

            def log_status():
                nonlocal prev_status
                last_quick_answer_chunk_decayed = (
                    last_quick_answer_chunk
                    and time.time() - last_quick_answer_chunk > TTS_FINAL_TIMEOUT
                    and time.time() - last_chunk_sent > TTS_FINAL_TIMEOUT
                )

                curr_status = (
                    # Access connection-specific state via callbacks
                    int(callbacks.tts_to_client),
                    int(callbacks.tts_client_playing),
                    int(callbacks.tts_chunk_sent),
                    1,  # Placeholder?
                    int(callbacks.is_hot),  # from callbacks
                    int(callbacks.synthesis_started),  # from callbacks
                    # Global manager state
                    int(app.state.SpeechPipelineManager.running_generation is not None),
                    # Global manager state
                    int(app.state.SpeechPipelineManager.is_valid_gen()),
                    int(is_tts_finished),  # Calculated local variable
                    # Input processor state
                    int(app.state.AudioInputProcessor.interrupted)
                )

                if curr_status != prev_status:
                    status = Colors.apply("🖥️🚦 State ").red
                    logger.info(
                        f"{status} ToClient {curr_status[0]}, "
                        # Renamed slightly for clarity
                        f"ttsClientON {curr_status[1]}, "
                        f"ChunkSent {curr_status[2]}, "
                        f"hot {curr_status[4]}, synth {curr_status[5]}"
                        f" gen {curr_status[6]}"
                        f" valid {curr_status[7]}"
                        f" tts_q_fin {curr_status[8]}"
                        f" mic_inter {curr_status[9]}"
                    )
                    prev_status = curr_status

            # Use connection-specific state via callbacks
            if not callbacks.tts_to_client:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if not app.state.SpeechPipelineManager.running_generation:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if app.state.SpeechPipelineManager.running_generation.abortion_started:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if not app.state.SpeechPipelineManager.running_generation.quick_answer_first_chunk_ready:
                await asyncio.sleep(0.001)
                log_status()
                continue

            chunk = None
            try:
                chunk = app.state.SpeechPipelineManager.running_generation.audio_chunks.get_nowait()
                if chunk:
                    last_quick_answer_chunk = time.time()
            except Empty:
                final_expected = app.state.SpeechPipelineManager.running_generation.quick_answer_provided
                audio_final_finished = app.state.SpeechPipelineManager.running_generation.audio_final_finished

                if not final_expected or audio_final_finished:
                    # Flush any remaining upsampled audio tail before finishing
                    try:
                        tail_chunk = app.state.Upsampler.flush_base64_chunk()
                        if tail_chunk:
                            message_queue.put_nowait({
                                "type": "tts_chunk",
                                "content": tail_chunk
                            })
                    except Exception as e:
                        logger.warning(
                            f"🖥️⚠️ Upsampler flush failed: {repr(e)}")
                    logger.info(
                        "🖥️🏁 Sending of TTS chunks and 'user request/assistant answer' cycle finished.")
                    callbacks.send_final_assistant_answer()  # Callbacks method

                    assistant_answer = app.state.SpeechPipelineManager.running_generation.quick_answer + \
                        app.state.SpeechPipelineManager.running_generation.final_answer
                    app.state.SpeechPipelineManager.running_generation = None

                    callbacks.tts_chunk_sent = False  # Reset via callbacks
                    callbacks.reset_state()  # Reset connection state via callbacks

                await asyncio.sleep(0.001)
                log_status()
                continue

            # --- Pacing / Prebuffer ---
            base64_chunk = app.state.Upsampler.get_base64_chunk(chunk)

            # Estimate duration by PCM size: bytes -> samples (16-bit mono)
            try:
                pcm_bytes = len(chunk)
                chunk_samples = pcm_bytes // 2
            except Exception:
                chunk_samples = 0

            if not pacing_started:
                # Prebuffer until threshold reached
                prebuffer_chunks.append(base64_chunk)
                prebuffer_samples += chunk_samples
                # Send the very first chunk immediately to minimize TTFA, then prebuffer
                if len(prebuffer_chunks) == 1:
                    message_queue.put_nowait({
                        "type": "tts_chunk",
                        "content": prebuffer_chunks[0]
                    })
                    samples_emitted += chunk_samples
                if prebuffer_samples / SR >= PREBUFFER_SEC:
                    pacing_started = True
                    pacing_start_monotonic = time.monotonic()
                    # Flush remaining prebuffered chunks at paced rate (skip the first which we already sent)
                    for b64 in prebuffer_chunks[1:]:
                        target_time = pacing_start_monotonic + \
                            (samples_emitted / SR)
                        delay = target_time - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        message_queue.put_nowait({
                            "type": "tts_chunk",
                            "content": b64
                        })
                        # Track samples precisely using prebuffer_samples
                        try:
                            # Estimate samples for this base64 frame using total prebuffer
                            # As a simple approximation, distribute evenly
                            samples_emitted += max(0, prebuffer_samples //
                                                   max(1, len(prebuffer_chunks)))
                        except Exception:
                            pass
                    prebuffer_chunks.clear()
                    prebuffer_samples = 0
                # Do not send further while building prebuffer
                last_chunk_sent = time.time()
                continue
            else:
                # Pace outgoing chunks relative to stream start
                target_time = pacing_start_monotonic + (samples_emitted / SR)
                delay = target_time - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                message_queue.put_nowait({
                    "type": "tts_chunk",
                    "content": base64_chunk
                })
                samples_emitted += chunk_samples
                last_chunk_sent = time.time()

            # Use connection-specific state via callbacks
            if not callbacks.tts_chunk_sent:
                # Use the async helper function instead of a thread
                asyncio.create_task(
                    _reset_interrupt_flag_async(app, callbacks))

            callbacks.tts_chunk_sent = True  # Set via callbacks

    except asyncio.CancelledError:
        pass  # Task cancellation is expected on disconnect
    except WebSocketDisconnect as e:
        logger.warning(
            f"🖥️⚠️ {Colors.apply('WARNING').red} disconnect in send_tts_chunks: {repr(e)}")
    except RuntimeError as e:
        logger.error(
            f"🖥️💥 {Colors.apply('RUNTIME_ERROR').red} in send_tts_chunks: {repr(e)}")
    except Exception as e:
        logger.exception(
            f"🖥️💥 {Colors.apply('EXCEPTION').red} in send_tts_chunks: {repr(e)}")


# --------------------------------------------------------------------
# Callback class to handle transcription events
# --------------------------------------------------------------------
class TranscriptionCallbacks:
    """
    Manages state and callbacks for a single WebSocket connection's transcription lifecycle.

    This class holds connection-specific state flags (like TTS status, user interruption)
    and implements callback methods triggered by the `AudioInputProcessor` and
    `SpeechPipelineManager`. It sends messages back to the client via the provided
    `message_queue` and manages interaction logic like interruptions and final answer delivery.
    It also includes a threaded worker to handle abort checks based on partial transcription.
    """

    def __init__(self, app: FastAPI, message_queue: asyncio.Queue):
        """
        Initializes the TranscriptionCallbacks instance for a WebSocket connection.

        Args:
            app: The FastAPI application instance (to access global components).
            message_queue: An asyncio queue for sending messages back to the client.
        """
        self.app = app
        self.message_queue = message_queue
        self.final_transcription = ""
        self.abort_text = ""
        self.last_abort_text = ""

        # Initialize connection-specific state flags here
        self.tts_to_client: bool = False
        self.user_interrupted: bool = False
        self.tts_chunk_sent: bool = False
        self.tts_client_playing: bool = False
        self.interruption_time: float = 0.0
        self.speech_enabled: bool = False  # Default to speech disabled

        # These were already effectively instance variables or reset logic existed
        self.silence_active: bool = True
        self.is_hot: bool = False
        self.user_finished_turn: bool = False
        self.synthesis_started: bool = False
        self.assistant_answer: str = ""
        self.final_assistant_answer: str = ""
        self.is_processing_potential: bool = False
        self.is_processing_final: bool = False
        self.last_inferred_transcription: str = ""
        self.final_assistant_answer_sent: bool = False
        self.partial_transcription: str = ""  # Added for clarity

        self.reset_state()  # Call reset to ensure consistency

        self.abort_request_event = threading.Event()
        self.abort_worker_thread = threading.Thread(
            target=self._abort_worker, name="AbortWorker", daemon=True)
        self.abort_worker_thread.start()

    def reset_state(self):
        """Resets connection-specific state flags and variables to their initial values."""
        # Reset all connection-specific state flags
        self.tts_to_client = False
        self.user_interrupted = False
        self.tts_chunk_sent = False
        # Don't reset tts_client_playing here, it reflects client state reports
        self.interruption_time = 0.0

        # Reset other state variables
        self.silence_active = True
        self.is_hot = False
        self.user_finished_turn = False
        self.synthesis_started = False
        self.assistant_answer = ""
        self.final_assistant_answer = ""
        self.is_processing_potential = False
        self.is_processing_final = False
        self.last_inferred_transcription = ""
        self.final_assistant_answer_sent = False
        self.partial_transcription = ""

        # Keep the abort call related to the audio processor/pipeline manager
        self.app.state.AudioInputProcessor.abort_generation()

    def _abort_worker(self):
        """Background thread worker to check for abort conditions based on partial text."""
        while True:
            was_set = self.abort_request_event.wait(
                timeout=0.1)  # Check every 100ms
            if was_set:
                self.abort_request_event.clear()
                # Only trigger abort check if the text actually changed
                if self.last_abort_text != self.abort_text:
                    self.last_abort_text = self.abort_text
                    logger.debug(
                        f"🖥️🧠 Abort check triggered by partial: '{self.abort_text}'")
                    self.app.state.SpeechPipelineManager.check_abort(
                        self.abort_text, False, "on_partial")

    def on_partial(self, txt: str):
        """
        Callback invoked when a partial transcription result is available.

        Updates internal state, sends the partial result to the client,
        and signals the abort worker thread to check for potential interruptions.

        Args:
            txt: The partial transcription text.
        """
        self.final_assistant_answer_sent = False  # New user speech invalidates previous final answer sending state
        self.final_transcription = ""  # Clear final transcription as this is partial
        self.partial_transcription = txt
        self.message_queue.put_nowait(
            {"type": "partial_user_request", "content": txt})
        self.abort_text = txt  # Update text used for abort check
        self.abort_request_event.set()  # Signal the abort worker

    def safe_abort_running_syntheses(self, reason: str):
        """Placeholder for safely aborting syntheses (currently does nothing)."""
        # TODO: Implement actual abort logic if needed, potentially interacting with SpeechPipelineManager
        pass

    def on_tts_allowed_to_synthesize(self):
        """Callback invoked when the system determines TTS synthesis can proceed."""
        # Access global manager state
        if self.app.state.SpeechPipelineManager.running_generation and not self.app.state.SpeechPipelineManager.running_generation.abortion_started:
            logger.info(f"{Colors.apply('🖥️🔊 TTS ALLOWED').blue}")

    def on_potential_sentence(self, txt: str):
        """
        Callback invoked when a potentially complete sentence is detected by the STT.

        Triggers the preparation of a speech generation based on this potential sentence.

        Args:
            txt: The potential sentence text.
        """
        logger.debug(f"🖥️🧠 Potential sentence: '{txt}'")
        # Access global manager state
        self.app.state.SpeechPipelineManager.prepare_generation(txt)

    def on_potential_final(self, txt: str):
        """
        Callback invoked when a potential *final* transcription is detected (hot state).

        Logs the potential final transcription.

        Args:
            txt: The potential final transcription text.
        """
        logger.info(f"{Colors.apply('🖥️🧠 HOT: ').magenta}{txt}")

    def on_potential_abort(self):
        """Callback invoked if the STT detects a potential need to abort based on user speech."""
        # Placeholder: Currently logs nothing, could trigger abort logic.
        pass

    def on_before_final(self, audio: bytes, txt: str):
        """
        Callback invoked just before the final STT result for a user turn is confirmed.

        Sets flags indicating user finished, allows TTS if pending, interrupts microphone input,
        releases TTS stream to client, sends final user request and any pending partial
        assistant answer to the client, and adds user request to history.

        Args:
            audio: The raw audio bytes corresponding to the final transcription. (Currently unused)
            txt: The transcription text (might be slightly refined in on_final).
        """
        logger.info(Colors.apply(
            '🖥️🏁 =================== USER TURN END ===================').light_gray)
        self.user_finished_turn = True
        # Reset connection-specific flag (user finished, not interrupted)
        self.user_interrupted = False
        # Access global manager state
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            logger.info(
                f"{Colors.apply('🖥️🔊 TTS ALLOWED (before final)').blue}")

        # first block further incoming audio (Audio processor's state)
        if not self.app.state.AudioInputProcessor.interrupted:
            logger.info(
                f"{Colors.apply('🖥️🎙️ ⏸️ Microphone interrupted (end of turn)').cyan}")
            self.app.state.AudioInputProcessor.interrupted = True
            self.interruption_time = time.time()  # Set connection-specific flag

        logger.info(f"{Colors.apply('🖥️🔊 TTS STREAM RELEASED').blue}")
        self.tts_to_client = True  # Set connection-specific flag

        # Send final user request (using the reliable final_transcription OR current partial if final isn't set yet)
        user_request_content = self.final_transcription if self.final_transcription else self.partial_transcription
        self.message_queue.put_nowait({
            "type": "final_user_request",
            "content": user_request_content
        })

        # Access global manager state
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            # Send partial assistant answer (if available) to the client
            # Use connection-specific user_interrupted flag
            if self.app.state.SpeechPipelineManager.running_generation.quick_answer and not self.user_interrupted:
                self.assistant_answer = self.app.state.SpeechPipelineManager.running_generation.quick_answer
                self.message_queue.put_nowait({
                    "type": "partial_assistant_answer",
                    "content": self.assistant_answer
                })

        logger.info(
            f"🖥️🧠 Adding user request to history: '{user_request_content}'")
        # Access global manager state
        self.app.state.SpeechPipelineManager.history.append(
            {"role": "user", "content": user_request_content})

    def on_final(self, txt: str):
        """
        Callback invoked when the final transcription result for a user turn is available.

        Logs the final transcription and stores it.

        Args:
            txt: The final transcription text.
        """
        logger.info(
            f"\n{Colors.apply('🖥️✅ FINAL USER REQUEST (STT Callback): ').green}{txt}")
        if not self.final_transcription:  # Store it if not already set by on_before_final logic
            self.final_transcription = txt

    def abort_generations(self, reason: str):
        """
        Triggers the abortion of any ongoing speech generation process.

        Logs the reason and calls the SpeechPipelineManager's abort method.

        Args:
            reason: A string describing why the abortion is triggered.
        """
        logger.info(
            f"{Colors.apply('🖥️🛑 Aborting generation:').blue} {reason}")
        # Access global manager state
        self.app.state.SpeechPipelineManager.abort_generation(
            reason=f"server.py abort_generations: {reason}")

    def on_silence_active(self, silence_active: bool):
        """
        Callback invoked when the silence detection state changes.

        Updates the internal silence_active flag.

        Args:
            silence_active: True if silence is currently detected, False otherwise.
        """
        # logger.debug(f"🖥️🎙️ Silence active: {silence_active}") # Optional: Can be noisy
        self.silence_active = silence_active

    def on_partial_assistant_text(self, txt: str):
        """
        Callback invoked when a partial text result from the assistant (LLM) is available.

        Updates the internal assistant answer state and sends the partial answer to the client,
        unless the user has interrupted.

        Args:
            txt: The partial assistant text.
        """
        logger.info(
            f"{Colors.apply('🖥️💬 PARTIAL ASSISTANT ANSWER: ').green}{txt}")
        # Use connection-specific user_interrupted flag
        if not self.user_interrupted:
            self.assistant_answer = txt
            # Use connection-specific tts_to_client flag
            if self.tts_to_client:
                self.message_queue.put_nowait({
                    "type": "partial_assistant_answer",
                    "content": txt
                })

    def on_recording_start(self):
        """
        Callback invoked when the audio input processor starts recording user speech.

        If client-side TTS is playing, it triggers an interruption: stops server-side
        TTS streaming, sends stop/interruption messages to the client, aborts ongoing
        generation, sends any final assistant answer generated so far, and resets relevant state.
        """
        logger.info(
            f"{Colors.ORANGE}🖥️🎙️ Recording started.{Colors.RESET} TTS Client Playing: {self.tts_client_playing}")
        # Use connection-specific tts_client_playing flag
        if self.tts_client_playing:
            self.tts_to_client = False  # Stop server sending TTS
            self.user_interrupted = True  # Mark connection as user interrupted
            logger.info(
                f"{Colors.apply('🖥️❗ INTERRUPTING TTS due to recording start').blue}")

            # Send final assistant answer *if* one was generated and not sent
            logger.info(Colors.apply(
                "🖥️✅ Sending final assistant answer (forced on interruption)").pink)
            self.send_final_assistant_answer(forced=True)

            # Minimal reset for interruption:
            self.tts_chunk_sent = False  # Reset chunk sending flag
            # self.assistant_answer = "" # Optional: Clear partial answer if needed

            logger.info("🖥️🛑 Sending stop_tts to client.")
            self.message_queue.put_nowait({
                "type": "stop_tts",  # Client handles this to mute/ignore
                "content": ""
            })

            logger.info(
                f"{Colors.apply('🖥️🛑 RECORDING START ABORTING GENERATION').red}")
            self.abort_generations(
                "on_recording_start, user interrupts, TTS Playing")

            logger.info("🖥️❗ Sending tts_interruption to client.")
            self.message_queue.put_nowait({  # Tell client to stop playback and clear buffer
                "type": "tts_interruption",
                "content": ""
            })

            # Reset state *after* performing actions based on the old state
            # Be careful what exactly needs reset vs persists (like tts_client_playing)
            # self.reset_state() # Might clear too much, like user_interrupted prematurely

    def send_final_assistant_answer(self, forced=False):
        """
        Sends the final (or best available) assistant answer to the client.

        Constructs the full answer from quick and final parts if available.
        If `forced` and no full answer exists, uses the last partial answer.
        Cleans the text and sends it as 'final_assistant_answer' if not already sent.

        Args:
            forced: If True, attempts to send the last partial answer if no complete
                    final answer is available. Defaults to False.
        """
        final_answer = ""
        # Access global manager state
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            final_answer = self.app.state.SpeechPipelineManager.running_generation.quick_answer + \
                self.app.state.SpeechPipelineManager.running_generation.final_answer

        if not final_answer:  # Check if constructed answer is empty
            # If forced, try using the last known partial answer from this connection
            if forced and self.assistant_answer:
                final_answer = self.assistant_answer
                logger.warning(
                    f"🖥️⚠️ Using partial answer as final (forced): '{final_answer}'")
            else:
                logger.warning(
                    f"🖥️⚠️ Final assistant answer was empty, not sending.")
                return  # Nothing to send

        logger.debug(
            f"🖥️✅ Attempting to send final answer: '{final_answer}' (Sent previously: {self.final_assistant_answer_sent})")

        if not self.final_assistant_answer_sent and final_answer:
            import re
            # Clean up the final answer text
            cleaned_answer = re.sub(r'[\r\n]+', ' ', final_answer)
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()
            cleaned_answer = cleaned_answer.replace('\\n', ' ')
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()

            if cleaned_answer:  # Ensure it's not empty after cleaning
                logger.info(
                    f"\n{Colors.apply('🖥️✅ FINAL ASSISTANT ANSWER (Sending): ').green}{cleaned_answer}")
                self.message_queue.put_nowait({
                    "type": "final_assistant_answer",
                    "content": cleaned_answer
                })
                self.app.state.SpeechPipelineManager.history.append(
                    {"role": "assistant", "content": cleaned_answer})
                self.final_assistant_answer_sent = True
                self.final_assistant_answer = cleaned_answer  # Store the sent answer
            else:
                logger.warning(
                    f"🖥️⚠️ {Colors.YELLOW}Final assistant answer was empty after cleaning.{Colors.RESET}")
                self.final_assistant_answer_sent = False  # Don't mark as sent
                self.final_assistant_answer = ""  # Clear the stored answer
        elif forced and not final_answer:  # Should not happen due to earlier check, but safety
            logger.warning(
                f"🖥️⚠️ {Colors.YELLOW}Forced send of final assistant answer, but it was empty.{Colors.RESET}")
            self.final_assistant_answer = ""  # Clear the stored answer


# --------------------------------------------------------------------
# Main WebSocket endpoint
# --------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Handles the main WebSocket connection for real-time voice chat.

    Accepts a connection, sets up connection-specific state via `TranscriptionCallbacks`,
    initializes audio/message queues, and creates asyncio tasks for handling
    incoming data, audio processing, outgoing text messages, and outgoing TTS chunks.
    Manages the lifecycle of these tasks and cleans up on disconnect.

    Args:
        ws: The WebSocket connection instance provided by FastAPI.
    """
    await ws.accept()
    logger.info("🖥️✅ Client connected via WebSocket.")

    message_queue = asyncio.Queue()
    audio_chunks = asyncio.Queue()

    # Set up callback manager - THIS NOW HOLDS THE CONNECTION-SPECIFIC STATE
    callbacks = TranscriptionCallbacks(app, message_queue)

    # Assign callbacks to the AudioInputProcessor (global component)
    # These methods within callbacks will now operate on its *instance* state
    app.state.AudioInputProcessor.realtime_callback = callbacks.on_partial
    app.state.AudioInputProcessor.recording_start_callback = callbacks.on_recording_start
    app.state.AudioInputProcessor.silence_active_callback = callbacks.on_silence_active

    # Set transcriber callbacks if transcriber exists
    if app.state.AudioInputProcessor.transcriber:
        app.state.AudioInputProcessor.transcriber.potential_sentence_end = callbacks.on_potential_sentence
        app.state.AudioInputProcessor.transcriber.on_tts_allowed_to_synthesize = callbacks.on_tts_allowed_to_synthesize
        app.state.AudioInputProcessor.transcriber.potential_full_transcription_callback = callbacks.on_potential_final
        app.state.AudioInputProcessor.transcriber.potential_full_transcription_abort_callback = callbacks.on_potential_abort
        app.state.AudioInputProcessor.transcriber.full_transcription_callback = callbacks.on_final
        app.state.AudioInputProcessor.transcriber.before_final_sentence = callbacks.on_before_final

    # Store callbacks in AudioInputProcessor for later assignment when transcriber is created
    app.state.AudioInputProcessor._pending_callbacks = callbacks

    # Assign callback to the SpeechPipelineManager (global component)
    app.state.SpeechPipelineManager.on_partial_assistant_text = callbacks.on_partial_assistant_text

    # Create tasks for handling different responsibilities
    # Pass the 'callbacks' instance to tasks that need connection-specific state
    tasks = [
        asyncio.create_task(process_incoming_data(
            ws, app, audio_chunks, callbacks)),  # Pass callbacks
        asyncio.create_task(
            app.state.AudioInputProcessor.process_chunk_queue(audio_chunks)),
        asyncio.create_task(send_text_messages(ws, message_queue)),
        asyncio.create_task(send_tts_chunks(
            app, message_queue, callbacks)),  # Pass callbacks
    ]

    try:
        # Wait for any task to complete (e.g., client disconnect)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if not task.done():
                task.cancel()
        # Await cancelled tasks to let them clean up if needed
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception as e:
        logger.error(
            f"🖥️💥 {Colors.apply('ERROR').red} in WebSocket session: {repr(e)}")
    finally:
        logger.info("🖥️🧹 Cleaning up WebSocket tasks...")
        for task in tasks:
            if not task.done():
                task.cancel()
        # Ensure all tasks are awaited after cancellation
        # Use return_exceptions=True to prevent gather from stopping on first error during cleanup
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("🖥️❌ WebSocket session ended.")

# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
if __name__ == "__main__":

    # Run the server without SSL
    if not USE_SSL:
        logger.info("🖥️▶️ Starting server without SSL.")
        uvicorn.run("server:app", host="127.0.0.1", port=8000, log_config=None)

    else:
        logger.info("🖥️🔒 Attempting to start server with SSL.")
        # Check if cert files exist
        cert_file = "127.0.0.1+1.pem"
        key_file = "127.0.0.1+1-key.pem"
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            logger.error(
                f"🖥️💥 SSL cert file ({cert_file}) or key file ({key_file}) not found.")
            logger.error("🖥️💥 Please generate them using mkcert:")
            # Assuming Windows based on earlier check, adjust if needed
            logger.error("🖥️💥   choco install mkcert")
            logger.error("🖥️💥   mkcert -install")
            # Remind user to replace with actual IP if needed
            logger.error("🖥️💥   mkcert 127.0.0.1 YOUR_LOCAL_IP")
            logger.error("🖥️💥 Exiting.")
            sys.exit(1)

        # Run the server with SSL
        logger.info(
            f"🖥️▶️ Starting server with SSL (cert: {cert_file}, key: {key_file}).")
        uvicorn.run(
            "server:app",
            host="127.0.0.1",
            port=8000,
            log_config=None,
            ssl_certfile=cert_file,
            ssl_keyfile=key_file,
        )
