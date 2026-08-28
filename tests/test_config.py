"""Credential resolution order and error quality."""

import pytest

from livesearchbench import config


class TestResolution:
    def test_explicit_override_wins(self, monkeypatch):
        monkeypatch.setenv("LSB_TEST_VAR", "from-env")
        assert config.get("LSB_TEST_VAR", override="explicit") == "explicit"

    def test_environment_beats_default(self, monkeypatch):
        monkeypatch.setenv("LSB_TEST_VAR", "from-env")
        assert config.get("LSB_TEST_VAR", default="fallback") == "from-env"

    def test_default_used_when_unset(self, monkeypatch):
        monkeypatch.delenv("LSB_TEST_VAR", raising=False)
        assert config.get("LSB_TEST_VAR", default="fallback") == "fallback"

    def test_missing_returns_none(self, monkeypatch):
        monkeypatch.delenv("LSB_TEST_VAR", raising=False)
        assert config.get("LSB_TEST_VAR") is None


class TestRequire:
    def test_returns_the_value(self, monkeypatch):
        monkeypatch.setenv("LSB_TEST_VAR", "v")
        assert config.require("LSB_TEST_VAR") == "v"

    def test_error_names_every_way_to_set_it(self, monkeypatch):
        monkeypatch.delenv("LSB_MISSING", raising=False)
        with pytest.raises(config.MissingCredential) as exc:
            config.require("LSB_MISSING", purpose="testing")
        message = str(exc.value)
        assert "LSB_MISSING" in message
        assert "export" in message and ".env" in message and "testing" in message


class TestDefaults:
    def test_serper_endpoint_is_not_empty(self):
        # The previous release shipped "" here, making RAG.py unrunnable.
        assert config.DEFAULT_SERPER_ENDPOINT.startswith("https://")

    def test_serper_credentials_supply_the_endpoint(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "k")
        monkeypatch.delenv("SERPER_ENDPOINT", raising=False)
        endpoint, key = config.serper_credentials()
        assert endpoint == config.DEFAULT_SERPER_ENDPOINT and key == "k"

    def test_serper_without_a_key_raises(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        monkeypatch.setattr(config, "_DOTENV_CACHE", {})
        with pytest.raises(config.MissingCredential):
            config.serper_credentials()

    def test_openai_base_url_has_a_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        base, _ = config.openai_credentials()
        assert base.startswith("https://")


class TestUserAgent:
    def test_includes_a_contact_and_the_component(self):
        ua = config.user_agent("LiveSearchBench-Test")
        assert "LiveSearchBench-Test" in ua
        assert "http" in ua or "@" in ua
        assert "python-requests" in ua

    def test_contact_is_overridable(self, monkeypatch):
        monkeypatch.setenv("LSB_CONTACT", "maintainer@example.org")
        assert "maintainer@example.org" in config.user_agent()
