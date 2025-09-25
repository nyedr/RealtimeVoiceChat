import torch
import numpy as np
import pyaudio
import logging
import traceback
import time
from typing import Optional, Union, List, Callable
from queue import Queue
from dataclasses import dataclass

from moshi.models.loaders import CheckpointInfo
from moshi.conditioners import dropout_all_conditions
from moshi.models.lm import LMGen
from moshi.models.tts import (
    Entry,
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
    ConditionAttributes,
    script_to_entries,
)


from RealtimeTTS import BaseEngine

DEFAULT_VOICE = "expresso/ex01-ex02_default_001_channel2_198s.wav"


class KyutaiVoice:
    """
    Represents a voice configuration for Kyutai TTS.

    Attributes:
        voice_path (str): Path to the voice embedding file or voice name.
        name (str): Human-readable name for the voice.
    """

    def __init__(self, voice_path: str, name: str = None):
        self.voice_path = voice_path
        self.name = name or voice_path

    def __repr__(self):
        return f"KyutaiVoice(voice_path='{self.voice_path}', name='{self.name}')"


@dataclass
class KyutaiTTSGen:
    """Internal TTS generator class based on the streaming example."""
    tts_model: 'TTSModel'
    attributes: List[ConditionAttributes]
    on_frame: Optional[Callable[[np.ndarray], None]] = None

    def __post_init__(self):
        tts_model = self.tts_model
        attributes = self.attributes
        self.offset = 0
        self.state = self.tts_model.machine.new_state([])
        # Logging/behavior flags
        # Set to True to force an immediate first step after first entry
        self.force_first_step = False

        if tts_model.cfg_coef != 1.0:
            if tts_model.valid_cfg_conditionings:
                raise ValueError(
                    "This model does not support direct CFG, but was trained with "
                    "CFG distillation. Pass instead `cfg_coef` to `make_condition_attributes`."
                )
            nulled = self._make_null(attributes)
            attributes = list(attributes) + nulled

        assert tts_model.lm.condition_provider is not None
        prepared = tts_model.lm.condition_provider.prepare(attributes)
        condition_tensors = tts_model.lm.condition_provider(prepared)

        def _on_text_logits_hook(text_logits):
            if tts_model.padding_bonus:
                text_logits[..., tts_model.machine.token_ids.pad] += (
                    tts_model.padding_bonus
                )
            return text_logits

        def _on_audio_hook(audio_tokens):
            audio_offset = tts_model.lm.audio_offset
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[1]):
                delay = delays[q + audio_offset]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[:, q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(
                    self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = torch.tensor(
                out_tokens, dtype=torch.long, device=text_tokens.device
            )

        tts_model.lm.dep_q = tts_model.n_q
        self.lm_gen = LMGen(
            tts_model.lm,
            temp=tts_model.temp,
            temp_text=tts_model.temp,
            cfg_coef=tts_model.cfg_coef,
            condition_tensors=condition_tensors,
            on_text_logits_hook=_on_text_logits_hook,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            cfg_is_masked_until=None,
            cfg_is_no_text=True,
        )
        # Don't call streaming_forever here - managed at engine level
        # self.lm_gen.streaming_forever(1)

    def _make_null(self, all_attributes: List[ConditionAttributes]) -> List[ConditionAttributes]:
        """When using CFG, returns the null conditions."""
        return dropout_all_conditions(all_attributes)

    def process_last(self):
        """Process remaining entries in the state."""
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            self._step()
        additional_steps = (
            self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        )
        for _ in range(additional_steps):
            self._step()

    def process(self):
        """Process entries while maintaining streaming threshold."""
        queue_size = len(self.state.entries)
        threshold = self.tts_model.machine.second_stream_ahead
        logging.info(
            f"KyutaiTTSGen: process() called at {time.time():.6f}, queue_size={queue_size}, threshold={threshold}")
        steps_taken = 0
        # Allow processing even with small queue sizes if force_first_step is enabled and we haven't started yet
        min_queue_size = 0 if (self.force_first_step and self.offset ==
                               0) else self.tts_model.machine.second_stream_ahead
        while len(self.state.entries) > min_queue_size:
            if steps_taken == 0:
                logging.info(
                    f"KyutaiTTSGen: Starting to step at {time.time():.6f} (queue_size={len(self.state.entries)} > min_queue_size={min_queue_size})")
            self._step()
            steps_taken += 1
            # After first step with force_first_step, revert to normal threshold
            if self.force_first_step and steps_taken == 1:
                min_queue_size = self.tts_model.machine.second_stream_ahead
        if steps_taken > 0:
            logging.info(
                f"KyutaiTTSGen: Completed {steps_taken} steps, queue_size now={len(self.state.entries)}")

    def _step(self):
        """Process a single step of generation."""
        if self.offset == 0:
            logging.info(
                f"KyutaiTTSGen: First _step() called at {time.time():.6f}")

        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = torch.full(
            (1, missing, 1),
            self.tts_model.machine.token_ids.zero,
            dtype=torch.long,
            device=self.tts_model.lm.device,
        )
        frame = self.lm_gen.step(input_tokens)
        self.offset += 1
        if frame is not None and self.on_frame is not None:
            self.on_frame(frame)

    def append_entry(self, entry):
        """Add an entry to the generation queue."""
        self.state.entries.append(entry)
        try:
            queue_size = len(self.state.entries)
            threshold = self.tts_model.machine.second_stream_ahead
            logging.info(
                f"KyutaiEngine: Appended entry at {time.time():.6f}, queue size: {queue_size}, threshold: {threshold}")
            if self.force_first_step and self.offset == 0 and queue_size > 0:
                logging.info(
                    f"KyutaiTTSGen: Forcing immediate first step (queue_size={queue_size})")
                self._step()
        except Exception:
            pass


class KyutaiEngine(BaseEngine):
    """
    Real-time Text-to-Speech engine using Kyutai's Moshi TTS model.

    This engine provides streaming TTS capabilities using Kyutai's state-of-the-art
    neural TTS model with real-time audio generation.
    """

    def __init__(
        self,
        hf_repo: str = DEFAULT_DSM_TTS_REPO,
        voice_repo: str = DEFAULT_DSM_TTS_VOICE_REPO,
        voice: Union[str, KyutaiVoice] = DEFAULT_VOICE,
        device: str = "cuda",
        n_q: int = 24,
        temp: float = 0.6,
        cfg_coef: float = 2.0,
        debug: bool = False
    ):
        """
        Initialize the Kyutai TTS engine.

        Args:
            hf_repo (str): HuggingFace repository for the TTS model.
            voice_repo (str): HuggingFace repository for voice embeddings.
            voice (Union[str, KyutaiVoice]): Voice configuration or path.
            device (str): PyTorch device ('cuda', 'cpu', etc.).
            n_q (int): Number of quantization levels.
            temp (float): Temperature for generation.
            cfg_coef (float): Classifier-free guidance coefficient.
            debug (bool): Enable debug logging.
        """

        super().__init__()
        self.hf_repo = hf_repo
        self.voice_repo = voice_repo
        self.device = device
        self.n_q = n_q
        self.temp = temp
        self.cfg_coef = cfg_coef
        self.debug = debug

        # Initialize voice
        if isinstance(voice, str):
            self.voice = KyutaiVoice(voice)
        else:
            self.voice = voice

        # Model initialization
        self.tts_model = None
        self.condition_attributes = None
        self.queue = Queue()
        self.tts_gen = None
        self.mimi_streaming_context = None
        self.lm_streaming_context = None

        # Streaming/voice coordination
        self.streaming_active = False
        self._pending_voice: Optional[KyutaiVoice] = None

        self.post_init()

    def post_init(self):
        """Initialize the engine after BaseEngine setup."""
        self.engine_name = "kyutai"
        self.can_consume_generators = True
        self._load_model()

    def _load_model(self):
        """Load the Kyutai TTS model and prepare voice conditions."""
        try:
            # Avoid re-loading if already initialized (post_init may be called twice)
            if getattr(self, "tts_model", None) is not None:
                logging.debug(
                    "Kyutai TTS model already loaded; skipping reload.")
                return
            logging.info("Loading Kyutai TTS model...")
            checkpoint_info = CheckpointInfo.from_hf_repo(self.hf_repo)
            self.tts_model = TTSModel.from_checkpoint_info(
                checkpoint_info,
                n_q=self.n_q,
                temp=self.temp,
                device=self.device
            )

            # Prepare voice conditions
            self._update_voice_conditions()

            # Initialize streaming context once
            self._init_streaming_context()

            logging.info("Kyutai TTS model loaded successfully")

        except Exception as e:
            logging.error(f"Failed to load Kyutai TTS model: {e}")
            raise

    def _init_streaming_context(self):
        """Initialize the streaming context and TTS generator."""
        try:
            # Initialize mimi streaming context
            self.mimi_streaming_context = self.tts_model.mimi.streaming(1)
            self.mimi_streaming_context.__enter__()

            # Create TTS generator once
            self.tts_gen = KyutaiTTSGen(
                self.tts_model,
                [self.condition_attributes],
                on_frame=None  # Will be set per synthesis call
            )

            # Initialize LM generator streaming context
            self.lm_streaming_context = self.tts_gen.lm_gen.streaming(1)
            self.lm_streaming_context.__enter__()

        except Exception as e:
            logging.error(f"Failed to initialize streaming context: {e}")
            raise

    def _reinitialize_tts_generator(self):
        """Reinitialize TTS generator with new conditions and manage streaming contexts."""
        try:
            # Clean up existing LM streaming context
            if hasattr(self, 'lm_streaming_context') and self.lm_streaming_context is not None:
                try:
                    self.lm_streaming_context.__exit__(None, None, None)
                except:
                    pass
                self.lm_streaming_context = None

            # Create new TTS generator
            self.tts_gen = KyutaiTTSGen(
                self.tts_model,
                [self.condition_attributes],
                on_frame=None
            )

            # Initialize new LM streaming context
            self.lm_streaming_context = self.tts_gen.lm_gen.streaming(1)
            self.lm_streaming_context.__enter__()

        except Exception as e:
            logging.error(f"Failed to reinitialize TTS generator: {e}")
            raise

    def _update_voice_conditions(self):
        """Update voice condition attributes."""
        if self.tts_model is None:
            return

        voice_path = self.voice.voice_path
        if not voice_path.endswith(".safetensors"):
            voice_path = self.tts_model.get_voice_path(voice_path)

        self.condition_attributes = self.tts_model.make_condition_attributes(
            [voice_path], cfg_coef=self.cfg_coef
        )

    def get_stream_info(self):
        """
        Returns the audio stream configuration for PyAudio.

        Returns:
            tuple: (format, channels, sample_rate)
        """
        if self.tts_model is None:
            # Return default values if model not loaded yet
            return pyaudio.paInt16, 1, 24000
        return pyaudio.paInt16, 1, self.tts_model.mimi.sample_rate

    def synthesize_generator(self, generator, audio_queue, stop_event):
        """
        Synthesize audio from a text generator using Kyutai TTS with proper streaming context management.

        Args:
            generator: A generator yielding text chunks.
            audio_queue: Queue to put audio chunks into.
            stop_event: Event to signal stopping.

        Returns:
            bool: True if synthesis was successful.
        """
        if self.tts_model is None:
            logging.error("TTS model not loaded")
            return False

        try:
            # Log TTS model configuration values for TTFA analysis
            logging.info(
                f"KyutaiEngine: TTS Config - second_stream_ahead={self.tts_model.machine.second_stream_ahead}, "
                f"initial_padding={self.tts_model.machine.initial_padding}, "
                f"max_padding={self.tts_model.machine.max_padding}, "
                f"delay_steps={self.tts_model.delay_steps}, "
                f"mimi_frame_rate={self.tts_model.mimi.frame_rate:.1f}, "
                f"audio_delay_est={self.tts_model.delay_steps / self.tts_model.mimi.frame_rate:.3f}s"
            )

            # Apply any pending voice update before opening streaming contexts
            if self._pending_voice is not None:
                try:
                    self.voice = self._pending_voice
                    self._pending_voice = None
                    self._update_voice_conditions()
                    logging.info(
                        "KyutaiEngine: applied pending voice before synthesis start")
                except Exception as e:
                    logging.warning(
                        f"KyutaiEngine: failed to apply pending voice: {e}")

            # Set up frame callback to put audio in queue
            first_audio_frame_logged = False

            def on_frame(frame):
                nonlocal first_audio_frame_logged
                if stop_event.is_set():
                    return

                if (frame != -1).all():
                    # Detach gradients before converting to numpy
                    pcm = self.tts_model.mimi.decode(
                        frame[:, 1:, :]).detach().cpu().numpy()
                    audio_data = np.clip(pcm[0, 0], -1, 1)

                    # Convert to int16 for PyAudio
                    audio_data = (audio_data * 32767).astype(np.int16)
                    audio_queue.put(audio_data.tobytes())

                    # Log first audio frame decoding timing
                    if not first_audio_frame_logged:
                        logging.info(
                            f"KyutaiEngine: First audio frame decoded at {time.time():.6f}, "
                            f"{len(audio_data)} samples -> queue")
                        first_audio_frame_logged = True

                    if self.debug:
                        logging.debug(
                            f"Generated audio chunk: {len(audio_data)} samples")

            # Ensure any previously-open streaming contexts are closed to avoid conflicts
            if hasattr(self, 'lm_streaming_context') and self.lm_streaming_context is not None:
                try:
                    self.lm_streaming_context.__exit__(None, None, None)
                except Exception:
                    pass
                self.lm_streaming_context = None

            if hasattr(self, 'mimi_streaming_context') and self.mimi_streaming_context is not None:
                try:
                    self.mimi_streaming_context.__exit__(None, None, None)
                except Exception:
                    pass
                self.mimi_streaming_context = None

            # Open fresh streaming contexts for this synthesis session
            self.streaming_active = True
            with self.tts_model.mimi.streaming(1):
                tts_gen = KyutaiTTSGen(
                    self.tts_model,
                    [self.condition_attributes],
                    on_frame=on_frame
                )
                tts_gen.force_first_step = True  # Force immediate processing of the first entry

                with tts_gen.lm_gen.streaming(1):
                    logging.info(
                        f"KyutaiEngine: Synthesis loop starting at {time.time():.6f} (streaming contexts opened)")
                    # Process text chunks from generator
                    for text_chunk in generator:
                        if stop_event.is_set():
                            logging.info(
                                "KyutaiEngine: synthesis stopped by user")
                            return False

                        if not text_chunk or not text_chunk.strip():
                            logging.debug(
                                f"KyutaiEngine: Skipping empty text chunk")
                            continue

                        logging.debug(
                            f"KyutaiEngine: Processing text chunk: '{text_chunk[:30]}...'")

                        # Prepare script entries for this chunk
                        stripped_text = text_chunk.strip()
                        if not stripped_text:
                            logging.debug(
                                "KyutaiEngine: Text chunk is empty after stripping, skipping")
                            continue
                        entries = self._prepare_script(
                            stripped_text, first_turn=False)
                        logging.debug(
                            f"KyutaiEngine: Prepared {len(entries)} entries for text: '{stripped_text[:30]}...'")

                        # Process entries
                        for entry in entries:
                            if stop_event.is_set():
                                return False
                            tts_gen.append_entry(entry)
                            tts_gen.process()

                    # Process any remaining entries
                    tts_gen.process_last()

            return True

        except Exception as e:
            logging.error(f"Kyutai synthesis error: {e}")
            if self.debug:
                traceback.print_exc()
            return False
        finally:
            self.streaming_active = False

    def synthesize(self, text) -> bool:
        """
        Synthesize text to audio using Kyutai TTS.

        Args:
            text: Text to synthesize (can be str or CharIterator).

        Returns:
            bool: True if synthesis was successful.
        """
        super().synthesize(text)

        if self.tts_model is None:
            logging.error("TTS model not loaded")
            return False

        try:
            # Convert text to string if it's a CharIterator or other iterator
            if hasattr(text, '__iter__') and not isinstance(text, str):
                # It's an iterator (like CharIterator), convert to string
                text_str = ''.join(char for char in text)
                if self.debug:
                    logging.debug(
                        f"Converted iterator to string: {len(text_str)} characters")
            else:
                # It's already a string
                text_str = str(text)

            if self.debug:
                logging.debug(
                    f"Synthesizing text: '{text_str[:50]}{'...' if len(text_str) > 50 else ''}'")

            # Prepare the script entries
            # Use consistent conditioning across quick/final phases to avoid voice shifts
            entries = self._prepare_script(text_str.strip(), first_turn=False)

            # Set up frame callback to put audio in queue
            def on_frame(frame):
                if self.stop_synthesis_event.is_set():
                    return

                if (frame != -1).all():
                    # Detach gradients before converting to numpy
                    pcm = self.tts_model.mimi.decode(
                        frame[:, 1:, :]).detach().cpu().numpy()
                    audio_data = np.clip(pcm[0, 0], -1, 1)

                    # Convert to int16 for PyAudio
                    audio_data = (audio_data * 32767).astype(np.int16)
                    self.queue.put(audio_data.tobytes())

                    if self.debug:
                        logging.debug(
                            f"Generated audio chunk: {len(audio_data)} samples")

            # Reuse existing TTS generator with new frame callback
            if self.tts_gen is None:
                logging.error("TTS generator not initialized")
                return False

            # Update the frame callback
            self.tts_gen.on_frame = on_frame

            # Reset generator state for new synthesis
            self.tts_gen.offset = 0
            self.tts_gen.state = self.tts_model.machine.new_state([])

            # Process entries (streaming context already active)
            for entry in entries:
                if self.stop_synthesis_event.is_set():
                    logging.info("KyutaiEngine: synthesis stopped by user")
                    return False

                self.tts_gen.append_entry(entry)
                self.tts_gen.process()

            self.tts_gen.process_last()

            return True

        except Exception as e:
            logging.error(f"Kyutai synthesis error: {e}")
            if self.debug:
                traceback.print_exc()
            return False

    def _prepare_script(self, script: str, first_turn: bool) -> List[Entry]:
        """Prepare script entries for TTS processing."""
        multi_speaker = first_turn and self.tts_model.multi_speaker
        return script_to_entries(
            self.tts_model.tokenizer,
            self.tts_model.machine.token_ids,
            self.tts_model.mimi.frame_rate,
            [script],
            multi_speaker=multi_speaker,
            padding_between=1,
        )

    def get_voices(self):
        """
        Get available voices (currently returns common voice configurations).

        Returns:
            list: List of KyutaiVoice instances.
        """
        # Default voice configurations - in practice these would be discovered
        # from the voice repository
        default_voices = [
            KyutaiVoice(
                "expresso/ex01-ex02_default_001_channel2_198s.wav", "Expresso Default"),
            KyutaiVoice(
                "expresso/ex03-ex01_happy_001_channel1_334s.wav", "Expresso Neu"),
            KyutaiVoice(
                "expresso/ex03-ex01_sad_001_channel1_334s.wav", "Expresso Sad"),
            KyutaiVoice(
                "expresso/ex03-ex01_neutral_001_channel1_334s.wav", "Expresso Neutral"),
            KyutaiVoice(
                "expresso/ex01-ex02_enunciated_001_channel2_354s.wav", "Expresso Enunciated"),
            KyutaiVoice(
                "expresso/ex01-ex02_fast_001_channel2_73s.wav", "Expresso Fast"),
            KyutaiVoice(
                "expresso/ex01-ex02_projected_002_channel2_248s.wav", "Expresso Projected"),
            KyutaiVoice(
                "expresso/ex01-ex02_whisper_001_channel2_717s.wav", "Expresso Whisper"),
            KyutaiVoice(
                "expresso/ex04-ex01_disgusted_001_channel1_130s.wav", "Expresso Disgusted"),
            KyutaiVoice(
                "expresso/ex04-ex01_laughing_001_channel1_306s.wav", "Expresso Laughing"),
            KyutaiVoice(
                "expresso/ex04-ex02_awe_001_channel2_1013s.wav", "Expresso Awe"),
            KyutaiVoice(
                "expresso/ex04-ex02_bored_001_channel2_232s.wav", "Expresso Bored"),
            KyutaiVoice(
                "expresso/ex04-ex02_confused_001_channel2_488s.wav", "Expresso Confused"),
            KyutaiVoice(
                "expresso/ex04-ex02_desire_001_channel2_694s.wav", "Expresso Desire"),
            KyutaiVoice(
                "expresso/ex04-ex02_fearful_001_channel2_266s.wav", "Expresso Fearful"),
            KyutaiVoice(
                "expresso/ex04-ex02_happy_001_channel2_140s.wav", "Expresso Happy"),
            KyutaiVoice(
                "expresso/ex04-ex02_sarcastic_001_channel2_466s.wav", "Expresso Sarcastic"),
        ]
        return default_voices

    def set_voice(self, voice: Union[str, KyutaiVoice]):
        """
        Set the voice for synthesis.

        Args:
            voice (Union[str, KyutaiVoice]): Voice path or KyutaiVoice instance.
        """
        if isinstance(voice, str):
            self.voice = KyutaiVoice(voice)
        elif isinstance(voice, KyutaiVoice):
            self.voice = voice

        # Update voice conditions if model is loaded
        if self.tts_model is not None:
            # If currently streaming, defer voice update until the next synthesis start
            if getattr(self, 'streaming_active', False):
                try:
                    self._pending_voice = self.voice
                    logging.warning(
                        "KyutaiEngine: set_voice deferred (streaming active)")
                except Exception as e:
                    logging.warning(
                        f"KyutaiEngine: failed to defer voice: {e}")
                return
            try:
                self._update_voice_conditions()
                # Reinitialize TTS generator with new voice conditions (if pre-initialized)
                if self.tts_gen is not None:
                    self._reinitialize_tts_generator()
            except Exception as e:
                logging.warning(f"Delaying set_voice due to error: {e}")

    def set_voice_parameters(self, **voice_parameters):
        """
        Set voice parameters for synthesis.

        Args:
            **voice_parameters: Parameters like temp, cfg_coef, etc.
        """
        valid_params = ['temp', 'cfg_coef', 'n_q']
        for param, value in voice_parameters.items():
            if param in valid_params:
                setattr(self, param, value)
                if param == 'temp' and self.tts_model is not None:
                    self.tts_model.temp = value
                elif param == 'cfg_coef':
                    self.cfg_coef = value
                    if self.tts_model is not None:
                        self._update_voice_conditions()
                        # Reinitialize TTS generator with new conditions
                        if self.tts_gen is not None:
                            self._reinitialize_tts_generator()
            elif self.debug:
                logging.warning(f"Ignoring invalid parameter: {param}")

    def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        super().shutdown()

        # Clean up LM streaming context
        if hasattr(self, 'lm_streaming_context') and self.lm_streaming_context is not None:
            try:
                self.lm_streaming_context.__exit__(None, None, None)
            except:
                pass
            self.lm_streaming_context = None

        # Clean up mimi streaming context
        if hasattr(self, 'mimi_streaming_context') and self.mimi_streaming_context is not None:
            try:
                self.mimi_streaming_context.__exit__(None, None, None)
            except:
                pass
            self.mimi_streaming_context = None

        # Clean up TTS generator
        if hasattr(self, 'tts_gen') and self.tts_gen is not None:
            self.tts_gen = None

        # Clean up model
        if hasattr(self, 'tts_model') and self.tts_model is not None:
            del self.tts_model
            self.tts_model = None
