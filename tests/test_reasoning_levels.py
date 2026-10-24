import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from importlib import import_module

run = import_module("03_run")


class FakeOpenAIClient:
    def __init__(self):
        self.request = None
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            model="gpt-5.6-luna-test",
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
        )


class FakeAnthropicClient:
    def __init__(self):
        self.request = None
        self.messages = self

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="{}")],
            usage=SimpleNamespace(input_tokens=12, output_tokens=4),
            model="claude-opus-5-test",
        )


class ReasoningLevelTests(unittest.TestCase):
    def test_openai_reasoning_effort_is_explicit(self):
        client = FakeOpenAIClient()
        run.chiama_openai(client, "gpt-5.6-luna", [], None, "medium")
        self.assertEqual(client.request["reasoning_effort"], "medium")

    def test_anthropic_effort_is_explicit(self):
        client = FakeAnthropicClient()
        run.chiama_anthropic(client, "claude-opus-5", [], None, "high")
        self.assertEqual(client.request["output_config"]["effort"], "high")


if __name__ == "__main__":
    unittest.main()
