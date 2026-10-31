"""Offline tests: Gemini and gTTS are mocked, so no API key or internet is needed.

Run from the project folder:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import speech  # noqa: E402
import translator as t  # noqa: E402


class ValidationTests(unittest.TestCase):
    def test_empty_input(self):
        for bad in ("", "   \n ", None):
            with self.assertRaises(t.InvalidInputError):
                t.validate_text(bad)

    def test_symbols_only(self):
        with self.assertRaises(t.InvalidInputError):
            t.validate_text("12345 !!! ???")

    def test_too_long(self):
        with self.assertRaises(t.InvalidInputError):
            t.validate_text("a" * (t.MAX_INPUT_CHARS + 1))

    def test_valid(self):
        self.assertEqual(t.validate_text("  Hello world \r\n"), "Hello world")


class PromptTests(unittest.TestCase):
    def test_every_language_and_mode_builds(self):
        for lang in t.LANGUAGES.values():
            for mode in t.MODES:
                prompt = t.build_system_prompt(lang, mode)
                self.assertIn(lang.name, prompt)
                self.assertIn("Do not invent", prompt)

    def test_simple_mode_has_example_label(self):
        te = t.LANGUAGES["Telugu"]
        self.assertIn(te.example_label, t.build_system_prompt(te, t.MODE_SIMPLE))

    def test_telugu_is_first(self):
        self.assertEqual(next(iter(t.LANGUAGES)), "Telugu")


class TranslateTests(unittest.TestCase):
    def setUp(self):
        p = mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
        p.start()
        self.addCleanup(p.stop)

    def test_success_and_markdown_cleanup(self):
        with mock.patch.object(t, "generate_text", return_value="**మీ** దరఖాస్తు\n\n\n\n* పరిశీలన") as g:
            out = t.translate_text("Your application is received.", "Telugu", t.MODE_DIRECT)
        self.assertEqual(out, "మీ దరఖాస్తు\n\n• పరిశీలన")
        self.assertIn("<source>", g.call_args.args[0])

    def test_all_languages_route_through_ai(self):
        for name in t.LANGUAGES:
            with mock.patch.object(t, "generate_text", return_value="ok"):
                self.assertEqual(t.translate_text("Hello", name, t.MODE_SIMPLE), "ok")

    def test_missing_key(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
            with self.assertRaises(t.MissingAPIKeyError):
                t.translate_text("Hello", "Telugu")

    def test_bad_language_and_mode(self):
        with self.assertRaises(t.UnsupportedLanguageError):
            t.translate_text("Hello", "Klingon")
        with self.assertRaises(t.InvalidInputError):
            t.translate_text("Hello", "Telugu", "Shout")

    def test_connection_failure_is_friendly(self):
        fake_client = mock.Mock()
        fake_client.models.generate_content.side_effect = ConnectionError("down")
        with mock.patch.object(t, "get_client", return_value=fake_client), \
             mock.patch.object(t.time, "sleep"):
            with self.assertRaises(t.APIConnectionError) as ctx:
                t.translate_text("Hello", "Telugu")
        self.assertIn("internet", str(ctx.exception))
        self.assertEqual(fake_client.models.generate_content.call_count, t.MAX_ATTEMPTS)

    def test_empty_ai_response(self):
        fake_client = mock.Mock()
        fake_client.models.generate_content.return_value = mock.Mock(text=None)
        with mock.patch.object(t, "get_client", return_value=fake_client):
            with self.assertRaises(t.AIResponseError):
                t.translate_text("Hello", "Hindi")


class SpeechTests(unittest.TestCase):
    def test_unsupported_spoken_language(self):
        with self.assertRaises(t.UnsupportedLanguageError):
            speech.transcribe_audio(b"x" * 2000, "audio/wav", "French")

    def test_empty_audio(self):
        with self.assertRaises(t.InvalidInputError):
            speech.transcribe_audio(b"", "audio/wav", "English")

    def test_no_speech_detected(self):
        with mock.patch.object(speech, "generate_text", return_value="NO_SPEECH"):
            with self.assertRaises(t.InvalidInputError):
                speech.transcribe_audio(b"x" * 2000, "audio/wav", "English")

    def test_transcription_success(self):
        with mock.patch.object(speech, "generate_text", return_value=" Hello there "):
            self.assertEqual(speech.transcribe_audio(b"x" * 2000, "audio/wav", "English"), "Hello there")

    def test_tts_validation(self):
        with self.assertRaises(t.UnsupportedLanguageError):
            speech.synthesize_speech("hello", "French")
        with self.assertRaises(t.InvalidInputError):
            speech.synthesize_speech("  **  ", "Telugu")

    def test_clean_for_speech(self):
        self.assertEqual(speech.clean_for_speech("**Hi**\n• there"), "Hi. there")


if __name__ == "__main__":
    unittest.main()
