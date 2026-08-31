"""AReaL workflow that routes Modal LLM calls through Ogma tunnels."""

from __future__ import annotations

from typing import Any

from areal.api import InferenceEngine
from platoon.train.areal.proxy import ArealProxySession
from platoon.train.areal.workflows import GroupRolloutWorkflow


class OgmaRoutedGroupRolloutWorkflow(GroupRolloutWorkflow):
    """Keep session traffic internal while exposing the LLM endpoint via Ogma."""

    def __init__(
        self,
        *args,
        proxy_endpoint_map: dict[str, str],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.proxy_endpoint_map = dict(proxy_endpoint_map)

    def to_workflow_kwargs(self) -> dict[str, Any]:
        kwargs = super().to_workflow_kwargs()
        kwargs["proxy_endpoint_map"] = self.proxy_endpoint_map
        return kwargs

    def _build_rollout_config(
        self,
        engine: InferenceEngine,
        session: ArealProxySession,
    ):
        config = super()._build_rollout_config(engine, session)
        internal_url = self._require_proxy_base_url().rstrip("/")
        try:
            public_url = self.proxy_endpoint_map[internal_url]
        except KeyError as error:
            raise RuntimeError(
                f"No Ogma tunnel registered for AReaL proxy {internal_url}; "
                f"known proxies: {sorted(self.proxy_endpoint_map)}"
            ) from error
        config.rollout_config.model_endpoint = public_url
        return config
