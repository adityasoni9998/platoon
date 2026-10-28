"""Workspace configuration for SWE-rebench rollouts."""

from pathlib import Path
from typing import Literal

from platoon.openhands.apptainer import DirectoryOverlayWorkspace
from pydantic import BaseModel, ConfigDict, model_validator


class ApptainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sif_dir: str
    cache_dir: str
    scratch_dir: str

    def create_workspace(self, instance: dict) -> DirectoryOverlayWorkspace:
        sif_path = Path(self.sif_dir) / f"{instance['instance_id']}.sif"
        if not sif_path.is_file():
            raise FileNotFoundError(f"Prebuilt Apptainer SIF not found: {sif_path}")
        scratch_dir = Path(self.scratch_dir)
        if not scratch_dir.is_dir():
            raise ValueError(
                f"Apptainer scratch_dir must be an existing local directory: {scratch_dir}"
            )
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return DirectoryOverlayWorkspace(
            sif_file=str(sif_path),
            cache_dir=str(cache_dir),
            scratch_dir=str(scratch_dir),
            working_dir="/testbed",
            detach_logs=False,
            health_check_timeout=600,
        )


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["modal", "apptainer"] = "modal"
    apptainer: ApptainerConfig | None = None

    @model_validator(mode="after")
    def require_apptainer_config(self):
        if self.type == "apptainer" and self.apptainer is None:
            raise ValueError("workspace.apptainer is required when workspace.type is 'apptainer'")
        return self
