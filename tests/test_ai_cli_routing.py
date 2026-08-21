"""Regression tests for configured CLI routing in the AI agent."""

import unittest

from src.ai import ai_agent, ai_provider


class CliRoutingTest(unittest.TestCase):
    def test_codex_cli_bypasses_api_agent_provider(self):
        """Choosing Codex CLI must call the configured CLI, not an API SDK."""
        original_method = ai_provider.get_ai_method
        original_generate = ai_provider.generate
        original_key = ai_provider.get_ai_key
        calls = []
        try:
            ai_provider.get_ai_method = lambda: 'codex'
            ai_provider.get_ai_key = lambda _provider: (_ for _ in ()).throw(
                AssertionError('CLI routing must not request an API key'))

            def fake_generate(system, conversation, **kwargs):
                calls.append((system, conversation, kwargs))
                return 'Codex reply'

            ai_provider.generate = fake_generate
            replies = []
            result = ai_agent.run_agent(
                'Inspect this project', [], system='system instructions',
                on_text=replies.append, remember=False)
        finally:
            ai_provider.get_ai_method = original_method
            ai_provider.generate = original_generate
            ai_provider.get_ai_key = original_key

        self.assertEqual(result, 'Codex reply')
        self.assertEqual(replies, ['Codex reply'])
        self.assertEqual(calls[0][1], 'Inspect this project')
        self.assertEqual(calls[0][2]['method'], 'codex')


class SpeechRecognitionTranscriptionTest(unittest.TestCase):
    def test_transcription_uses_speech_recognition_without_api_key(self):
        """Voice input must not initialize Gemini or require an API key."""
        import sys
        import types
        from unittest.mock import patch
        from src.ai.assistant import voice_io

        calls = []

        class FakeAudioFile:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class FakeRecognizer:
            def record(self, source):
                return 'audio'

            def recognize_google(self, audio, language):
                calls.append((audio, language))
                return 'recognized text'

        fake_sr = types.SimpleNamespace(
            Recognizer=FakeRecognizer,
            AudioFile=FakeAudioFile,
            UnknownValueError=type('UnknownValueError', (Exception,), {}),
            RequestError=type('RequestError', (Exception,), {}),
        )
        with patch.dict(sys.modules, {'speech_recognition': fake_sr}):
            result = voice_io.transcribe(b'not-a-real-wav', language_hint='pl')

        self.assertEqual(result, 'recognized text')
        self.assertEqual(calls, [('audio', 'pl-PL')])

if __name__ == '__main__':
    unittest.main(verbosity=2)