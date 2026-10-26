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


class OutputAlignmentTests(unittest.TestCase):
    def test_same_schema_and_token_limit_for_both_providers(self):
        openai, anthropic = FakeOpenAIClient(), FakeAnthropicClient()
        run.chiama_openai(openai, "gpt-5.6-luna", [], None, "medium")
        run.chiama_anthropic(anthropic, "claude-haiku-5-5", [], None, "medium")

        formato_openai = openai.request["response_format"]
        self.assertEqual(formato_openai["type"], "json_schema")
        self.assertTrue(formato_openai["json_schema"]["strict"])
        self.assertEqual(
            formato_openai["json_schema"]["schema"],
            anthropic.request["output_config"]["format"]["schema"],
        )
        self.assertEqual(openai.request["max_completion_tokens"], anthropic.request["max_tokens"])


if __name__ == "__main__":
    unittest.main()
