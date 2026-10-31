"""Voice features for Bhasha AI.

* Speech-to-text: the recording is sent to Gemini, which transcribes it. This
  needs no extra service and understands English and the Indian languages
  listed in ``SPEECH_LANGUAGES``.
* Text-to-speech: Google Text-to-Speech (``gTTS``) reads the result aloud in
  Telugu, Hindi, Tamil, Kannada, Malayalam and Marathi. It needs internet.
"""

from __future__ import annotations

import io
import logging
import re

from google.genai import types

from translator import (
    LANGUAGES,
    APIConnectionError,
    BhashaError,
    InvalidInputError,
    UnsupportedLanguageError,
    generate_text,
)

logger = logging.getLogger("bhasha.speech")

# Languages a user may speak in. English comes first because the app's input
# is English; the others help people who prefer to speak their own language.
SPEECH_LANGUAGES: tuple[str, ...] = ("English", *LANGUAGES.keys())

MAX_AUDIO_BYTES = 15 * 1024 * 1024  # stays inside Gemini's inline-upload limit
MAX_TTS_CHARS = 4000
_SUPPORTED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3",
    "audio/ogg", "audio/flac", "audio/aac", "audio/aiff",
}
_NO_SPEECH_MARKER = "NO_SPEECH"


# --------------------------------------------------------------------------- #
# Speech to text
# --------------------------------------------------------------------------- #


def transcribe_audio(audio_bytes: bytes, mime_type: str | None, spoken_language: str = "English") -> str:
    """Convert a recording to text.

    Raises:
        UnsupportedLanguageError: the spoken language is not on our list.
        InvalidInputError: nothing recorded, too large, or no speech heard.
        BhashaError subclasses: for API problems.
    """
    if spoken_language not in SPEECH_LANGUAGES:
        raise UnsupportedLanguageError(
            f"Speaking in “{spoken_language}” is not supported yet. "
            f"Please choose one of: {', '.join(SPEECH_LANGUAGES)}."
        )
    if not audio_bytes:
        raise InvalidInputError("No recording was received. Tap the microphone and try again.")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise InvalidInputError("The recording is too long. Please speak for under a few minutes.")

    mime = (mime_type or "audio/wav").split(";")[0].strip().lower()
    if mime not in _SUPPORTED_AUDIO_TYPES:
        raise InvalidInputError(
            f"This audio format ({mime}) is not supported. Please record again or type your text."
        )

    prompt = (
        f"Transcribe this recording. The speaker is speaking {spoken_language}. "
        "Write exactly what was said, with normal punctuation, in the script of "
        "the language spoken (Roman letters for English). Do not translate, "
        "summarise or answer it. Output only the transcript. "
        f"If there is no clear human speech, output exactly {_NO_SPEECH_MARKER}."
    )
    transcript = generate_text(
        [prompt, types.Part.from_bytes(data=audio_bytes, mime_type=mime)],
        temperature=0.0,
    ).strip()

    if not transcript or _NO_SPEECH_MARKER in transcript:
        raise InvalidInputError(
            "I couldn't hear any speech. Move closer to the microphone, speak clearly and try again."
        )
    return transcript


# --------------------------------------------------------------------------- #
# Text to speech
# --------------------------------------------------------------------------- #


def clean_for_speech(text: str) -> str:
    """Strip symbols that sound odd when read aloud."""
    text = re.sub(r"[*_`#•]+", " ", text)
    text = re.sub(r"\s*\n\s*", ". ", text)
    text = re.sub(r"\.\s*\.", ".", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def synthesize_speech(text: str, language: str) -> bytes:
    """Return MP3 bytes that read ``text`` aloud in ``language``.

    Raises:
        UnsupportedLanguageError: no voice for that language.
        InvalidInputError: nothing to read.
        APIConnectionError: the voice service could not be reached.
    """
    lang = LANGUAGES.get(language)
    if lang is None:
        raise UnsupportedLanguageError(
            f"A voice for “{language}” is not available. Supported voices: {', '.join(LANGUAGES)}."
        )
    speakable = clean_for_speech(text or "")
    if not speakable:
        raise InvalidInputError("There is no text to read aloud yet. Translate something first.")
    if len(speakable) > MAX_TTS_CHARS:
        speakable = speakable[:MAX_TTS_CHARS].rsplit(" ", 1)[0]

    try:
        from gtts import gTTS
    except ImportError as exc:  # pragma: no cover - listed in requirements.txt
        raise APIConnectionError("The voice package (gTTS) is not installed. Run: pip install gTTS") from exc

    try:
        buffer = io.BytesIO()
        gTTS(text=speakable, lang=lang.code, slow=False).write_to_fp(buffer)
        audio = buffer.getvalue()
    except ValueError as exc:  # gTTS: "Language not supported"
        raise UnsupportedLanguageError(
            f"A voice for {language} is not available right now."
        ) from exc
    except BhashaError:
        raise
    except Exception as exc:  # gTTSError, network errors, timeouts
        logger.warning("Text-to-speech failed: %s", type(exc).__name__)
        raise APIConnectionError(
            "Couldn't create the voice. Check your internet connection and try again."
        ) from exc

    if not audio:
        raise APIConnectionError("The voice service returned no audio. Please try again.")
    return audio
