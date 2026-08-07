"""The Groq wrapper, with the SDK client itself faked out."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from agent.llm import LLMClient


class Shape(BaseModel):
    name: str
    count: int


def client_returning(content):
    """
    Builds a fake Groq client whose one completion returns `content`.

    Args:
        content (str): What the model should appear to have said.

    Returns:
        MagicMock: Stands in for `groq.Groq`.
    """
    fake = MagicMock()
    fake.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )
    return fake


@pytest.fixture
def llm():
    """
    An LLMClient with the SDK patched out.

    Returns:
        LLMClient: Configured the way `build_agent` configures it.
    """
    with patch("agent.llm.Groq") as groq:
        groq.return_value = client_returning("  hello  ")
        yield LLMClient(api_key="k", model="m", temperature=0.1, max_tokens=64)


class TestText:
    def test_returns_the_reply_stripped(self, llm):
        assert llm.text("be brief", "hi") == "hello"

    def test_sends_the_configured_model_and_settings(self, llm):
        llm.text("be brief", "hi")
        kw = llm.client.chat.completions.create.call_args.kwargs
        assert kw["model"] == "m"
        assert kw["temperature"] == 0.1
        assert kw["max_tokens"] == 64

    def test_sends_instructions_as_system_and_question_as_user(self, llm):
        llm.text("be brief", "hi")
        messages = llm.client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "be brief"}
        assert messages[1] == {"role": "user", "content": "hi"}


class TestStructured:
    def test_parses_a_valid_reply_into_the_schema(self, llm):
        llm.client = client_returning('{"name": "a", "count": 2}')
        out = llm.structured("fill it in", "go", Shape)
        assert (out.name, out.count) == ("a", 2)

    def test_asks_for_json_mode(self, llm):
        llm.client = client_returning('{"name": "a", "count": 2}')
        llm.structured("fill it in", "go", Shape)
        kw = llm.client.chat.completions.create.call_args.kwargs
        assert kw["response_format"] == {"type": "json_object"}

    def test_puts_the_schema_in_the_system_message(self, llm):
        llm.client = client_returning('{"name": "a", "count": 2}')
        llm.structured("fill it in", "go", Shape)
        system = llm.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "count" in system and "JSON only" in system

    def test_a_reply_that_does_not_match_raises_ValueError(self, llm):
        llm.client = client_returning('{"name": "a"}')      # count missing
        with pytest.raises(ValueError, match="Shape"):
            llm.structured("fill it in", "go", Shape)

    def test_junk_raises_ValueError_too(self, llm):
        llm.client = client_returning("not json at all")
        with pytest.raises(ValueError):
            llm.structured("fill it in", "go", Shape)
