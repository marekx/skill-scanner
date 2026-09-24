# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for first-party Anthropic workload identity federation (keyless auth)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm")

from skill_scanner.core.analyzers.llm_provider_config import ProviderConfig

FEDERATION_ENV = {
    "ANTHROPIC_FEDERATION_RULE_ID": "rule_test",
    "ANTHROPIC_ORGANIZATION_ID": "org_test",
    "ANTHROPIC_SERVICE_ACCOUNT_ID": "sa_test",
    "ANTHROPIC_IDENTITY_TOKEN": "fake.jwt.token",
}


def _fake_credentials(token: str) -> MagicMock:
    access_token = MagicMock()
    access_token.token = token
    provider = MagicMock(return_value=access_token)
    result = MagicMock()
    result.provider = provider
    return result


def test_is_anthropic_true_for_claude_direct() -> None:
    with patch.dict(os.environ, {}, clear=True):
        cfg = ProviderConfig(model="claude-sonnet-5")
    assert cfg.is_anthropic is True


@pytest.mark.parametrize(
    "model,provider",
    [
        ("bedrock/anthropic.claude-sonnet-5", None),
        ("vertex_ai/claude-sonnet-5", None),
        ("azure/gpt-4o", None),
        ("claude-sonnet-5", "openai-compatible"),
    ],
)
def test_is_anthropic_false_for_gateways(model: str, provider: str | None) -> None:
    with patch.dict(os.environ, {"SKILL_SCANNER_LLM_API_KEY": "k"}, clear=True):
        cfg = ProviderConfig(model=model, provider=provider)
    assert cfg.is_anthropic is False


def test_federation_mints_bearer_token() -> None:
    with patch.dict(os.environ, FEDERATION_ENV, clear=True):
        with patch(
            "anthropic.lib.credentials.default_credentials",
            return_value=_fake_credentials("oat-token"),
        ):
            cfg = ProviderConfig(model="claude-sonnet-5")

    assert cfg._using_anthropic_oauth is True
    assert cfg.api_key == "oat-token"

    params = cfg.get_request_params()
    assert params.get("auth_token") == "oat-token"
    assert "api_key" not in params
    assert params["extra_headers"]["anthropic-beta"] == "oauth-2025-04-20"
    cfg.validate()  # must not raise


def test_explicit_key_takes_precedence_over_federation() -> None:
    env = dict(FEDERATION_ENV, SKILL_SCANNER_LLM_API_KEY="explicit-key")
    with patch.dict(os.environ, env, clear=True):
        with patch("anthropic.lib.credentials.default_credentials") as mocked:
            cfg = ProviderConfig(model="claude-sonnet-5")
            mocked.assert_not_called()

    assert cfg._using_anthropic_oauth is False
    assert cfg.api_key == "explicit-key"
    assert cfg.get_request_params().get("api_key") == "explicit-key"


def test_no_credentials_raises_with_federation_hint() -> None:
    with patch.dict(os.environ, {}, clear=True):
        cfg = ProviderConfig(model="claude-sonnet-5")
    assert cfg.api_key is None
    with pytest.raises(ValueError, match="workload-identity"):
        cfg.validate()


def test_federation_env_without_sdk_falls_through() -> None:
    with patch.dict(os.environ, FEDERATION_ENV, clear=True):
        # Simulate the anthropic SDK not being installed.
        with patch.dict("sys.modules", {"anthropic.lib.credentials": None}):
            cfg = ProviderConfig(model="claude-sonnet-5")
    assert cfg._using_anthropic_oauth is False
    assert cfg.api_key is None
