"""Tests for KAgentXAILlm."""

from unittest import mock

import pytest
from openai import AsyncOpenAI

from kagent.adk.models._xai import DEFAULT_XAI_BASE_URL, KAgentXAILlm
from kagent.adk.types import ModelConfig


class TestKAgentXAILlm:
    def test_default_construction(self):
        llm = KAgentXAILlm(type="xai", model="grok-4.7", api_key="sk-test")
        assert llm.model == "grok-4.7"
        assert llm.base_url is None

    def test_supported_models_regex_covers_grok(self):
        patterns = KAgentXAILlm.supported_models()
        assert r"grok-.*" in patterns

    def test_client_uses_default_base_url_when_unset(self, monkeypatch):
        monkeypatch.delenv("XAI_API_BASE", raising=False)
        llm = KAgentXAILlm(type="xai", model="grok-4.7", api_key="sk-test")
        with mock.patch("kagent.adk.models._xai.AsyncOpenAI") as mock_client:
            mock_client.return_value = mock.MagicMock(spec=AsyncOpenAI)
            _ = llm._client
            assert mock_client.call_args.kwargs["base_url"] == DEFAULT_XAI_BASE_URL
            assert mock_client.call_args.kwargs["api_key"] == "sk-test"

    def test_client_uses_env_base_url_when_config_unset(self, monkeypatch):
        monkeypatch.setenv("XAI_API_BASE", "https://gateway.example.com/xai/v1")
        llm = KAgentXAILlm(type="xai", model="grok-4.7", api_key="sk-test")
        with mock.patch("kagent.adk.models._xai.AsyncOpenAI") as mock_client:
            mock_client.return_value = mock.MagicMock(spec=AsyncOpenAI)
            _ = llm._client
            assert mock_client.call_args.kwargs["base_url"] == "https://gateway.example.com/xai/v1"

    def test_config_base_url_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("XAI_API_BASE", "https://from-env.example.com/v1")
        llm = KAgentXAILlm(
            type="xai",
            model="grok-4.7",
            api_key="sk-test",
            base_url="https://from-config.example.com/v1",
        )
        with mock.patch("kagent.adk.models._xai.AsyncOpenAI") as mock_client:
            mock_client.return_value = mock.MagicMock(spec=AsyncOpenAI)
            _ = llm._client
            assert mock_client.call_args.kwargs["base_url"] == "https://from-config.example.com/v1"

    def test_client_reads_env_api_key(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "env-key")
        llm = KAgentXAILlm(type="xai", model="grok-4.3")
        with mock.patch("kagent.adk.models._xai.AsyncOpenAI") as mock_client:
            mock_client.return_value = mock.MagicMock(spec=AsyncOpenAI)
            _ = llm._client
            assert mock_client.call_args.kwargs["api_key"] == "env-key"

    def test_client_raises_when_no_key_and_no_passthrough(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        llm = KAgentXAILlm(type="xai", model="grok-4.7")
        with pytest.raises(ValueError, match="xAI API key must be provided"):
            _ = llm._client

    def test_client_allows_passthrough_without_key(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        llm = KAgentXAILlm(type="xai", model="grok-4.7", api_key_passthrough=True)
        with mock.patch("kagent.adk.models._xai.AsyncOpenAI") as mock_client:
            mock_client.return_value = mock.MagicMock(spec=AsyncOpenAI)
            _ = llm._client
            assert mock_client.call_args.kwargs.get("api_key") is None


class TestCreateLLMFromXAIConfig:
    def test_create_llm_from_xai_model_config(self):
        """Integration: _create_llm_from_model_config returns KAgentXAILlm for xai type."""
        from kagent.adk.types import XAI, _create_llm_from_model_config

        config = ModelConfig(
            model="grok-4.7",
            xai=XAI(base_url="https://api.x.ai/v1", temperature=0.5, max_tokens=1024),
        )
        result = _create_llm_from_model_config(config)
        assert isinstance(result, KAgentXAILlm)
        assert result.model == "grok-4.7"
        assert result.base_url == "https://api.x.ai/v1"
        assert result.temperature == 0.5
        assert result.max_tokens == 1024
