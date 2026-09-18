"""Tests for LLM retry behaviour."""

import anthropic
import httpx
import openai
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ax_prover.config import LLMConfig
from ax_prover.utils.llm import LLMClient, _is_transient_error, create_llm


def _status_error(cls, status: int) -> Exception:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


class TestIsTransientError:
    """Only failures that can succeed on a retry are classified as transient."""

    @pytest.mark.parametrize(
        "exc",
        [
            _status_error(anthropic.RateLimitError, 429),
            _status_error(anthropic.InternalServerError, 500),
            _status_error(anthropic.APIStatusError, 529),
            _status_error(openai.RateLimitError, 429),
            anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")),
        ],
    )
    def test_transient(self, exc):
        assert _is_transient_error(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            _status_error(anthropic.BadRequestError, 400),
            _status_error(anthropic.AuthenticationError, 401),
            _status_error(anthropic.NotFoundError, 404),
            # Unsupported provider_config is rejected client-side before any request
            ValueError("`thinking={'budget_tokens': ...}` is not supported for claude-opus-5"),
        ],
    )
    def test_permanent(self, exc):
        assert _is_transient_error(exc) is False


class TestClientRetry:
    """LLMClient retries transient failures and fails fast on permanent ones."""

    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        config = LLMConfig(model="anthropic:claude-opus-5", provider_config={})
        config.retry_config = {
            "stop_after_attempt": 5,
            "exponential_jitter_params": {"initial": 0.01, "max": 0.02, "exp_base": 2.0},
        }
        return LLMClient(config)

    async def test_retries_transient_until_success(self, client):
        attempts = []

        def flaky(_):
            attempts.append(1)
            if len(attempts) < 3:
                raise _status_error(anthropic.RateLimitError, 429)
            return AIMessage(content="recovered")

        client._get_runnable = lambda **kwargs: RunnableLambda(flaky)

        response = await client.ainvoke([])

        assert response.content == "recovered"
        assert len(attempts) == 3

    async def test_permanent_error_is_not_retried(self, client):
        """A permanent error must surface immediately rather than retry for hours."""
        attempts = []

        def always_invalid(_):
            attempts.append(1)
            raise ValueError("budget_tokens is not supported for claude-opus-5")

        client._get_runnable = lambda **kwargs: RunnableLambda(always_invalid)

        with pytest.raises(ValueError, match="budget_tokens"):
            await client.ainvoke([])

        assert len(attempts) == 1


class TestCreateLlmProviderKwargs:
    """Anthropic-only provider kwargs never reach a non-Anthropic SDK.

    `betas` and `thinking` are Anthropic Messages-API concepts. LangChain's `init_chat_model`
    forwards unknown kwargs into `model_kwargs`, and the OpenAI SDK then rejects them at
    call time (`AsyncCompletions.parse() got an unexpected keyword argument 'betas'`) — which is
    how the packaged defaults (`betas: [structured-outputs-…]`, `thinking: {…}`) broke every
    `openai:` model, because a `--config` merge cannot remove a default key (#45).
    """

    ANTHROPIC_ONLY = {
        "betas": ["structured-outputs-2025-11-13"],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }

    def test_openai_model_drops_anthropic_only_kwargs(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = LLMConfig(
            model="openai:deepseek-v4-pro",
            provider_config={
                **self.ANTHROPIC_ONLY,
                "base_url": "https://example.invalid/v1",
                "max_tokens": 131072,
                "temperature": 1.0,
                "reasoning_effort": "max",
            },
        )
        llm = create_llm(config)
        assert "betas" not in llm.model_kwargs
        assert "thinking" not in llm.model_kwargs
        assert llm.max_tokens == 131072
        assert llm.temperature == 1.0
        assert llm.reasoning_effort == "max"

    def test_google_model_drops_anthropic_only_kwargs(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        config = LLMConfig(
            model="google_genai:gemini-2.5-pro", provider_config=dict(self.ANTHROPIC_ONLY)
        )
        llm = create_llm(config)
        dumped = llm.model_dump()
        assert "betas" not in str(dumped.get("model_kwargs", {}))
        assert "thinking" not in str(dumped.get("model_kwargs", {}))

    def test_anthropic_model_keeps_betas_and_thinking(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        config = LLMConfig(
            model="anthropic:claude-opus-5", provider_config=dict(self.ANTHROPIC_ONLY)
        )
        llm = create_llm(config)
        assert llm.betas == ["structured-outputs-2025-11-13"]
        assert llm.thinking == {"type": "enabled", "budget_tokens": 1024}
