"""Configuration loading/merging for pipeline.yaml + toolchains.yaml + strategies.yaml."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Toolchain:
    id: str
    root: str
    nvcc: str
    ptxas: str
    nvdisasm: str
    cuobjdump: str
    arch: str
    ccbin: str
    runnable: bool
    nvcc_version: str = ""
    ptxas_version: str = ""
    has_ptxas_o4: bool = False          # accepted but may be an alias
    o4_is_alias_of: Optional[str] = None  # e.g. "O3" when -O4 == -O3 SASS

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "root": self.root, "nvcc": self.nvcc, "ptxas": self.ptxas,
            "nvdisasm": self.nvdisasm, "cuobjdump": self.cuobjdump,
            "arch": self.arch, "ccbin": self.ccbin, "runnable": self.runnable,
            "nvcc_version": self.nvcc_version, "ptxas_version": self.ptxas_version,
            "has_ptxas_o4": self.has_ptxas_o4, "o4_is_alias_of": self.o4_is_alias_of,
        }


@dataclass
class OptimizationProfile:
    name: str
    ptxas_opt: int


@dataclass
class Config:
    project_root: Path
    toolchains: List[Toolchain]
    profiles: List[OptimizationProfile]
    generation: Dict[str, Any]
    compilation: Dict[str, Any]
    execution: Dict[str, Any]
    oracle: Dict[str, Any]
    stress_gate: Dict[str, Any]
    novelty: Dict[str, Any]
    storage: Dict[str, Any]
    scheduling: Dict[str, Any]
    reporting: Dict[str, Any]
    strategies: Dict[str, Any] = field(default_factory=dict)

    def runnable_toolchains(self) -> List[Toolchain]:
        return [t for t in self.toolchains if t.runnable]

    def profile_by_name(self, name: str) -> OptimizationProfile:
        for p in self.profiles:
            if p.name == name:
                return p
        raise KeyError(f"unknown profile {name}")

    def find_toolchain(self, toolchain_id: str) -> Toolchain:
        for t in self.toolchains:
            if t.id == toolchain_id:
                return t
        raise KeyError(f"unknown toolchain {toolchain_id}")


def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Optional[str] = None,
                toolchains_path: Optional[str] = None,
                strategies_path: Optional[str] = None) -> Config:
    root = PROJECT_ROOT
    cfg_path = Path(config_path) if config_path else root / "config" / "pipeline.yaml"
    tc_path = Path(toolchains_path) if toolchains_path else root / "config" / "toolchains.yaml"
    st_path = Path(strategies_path) if strategies_path else root / "config" / "strategies.yaml"

    pipeline_cfg = yaml.safe_load(cfg_path.read_text()) or {}
    tc_cfg = yaml.safe_load(tc_path.read_text()) or {}
    st_cfg = yaml.safe_load(st_path.read_text()) or {}

    if pipeline_cfg.get("project", {}).get("root"):
        root = Path(pipeline_cfg["project"]["root"]).resolve()

    toolchains = []
    for tc in tc_cfg.get("toolchains", []):
        toolchains.append(Toolchain(
            id=tc["id"], root=tc["root"], nvcc=tc["nvcc"], ptxas=tc["ptxas"],
            nvdisasm=tc["nvdisasm"], cuobjdump=tc["cuobjdump"],
            arch=tc["arch"], ccbin=tc["ccbin"], runnable=bool(tc.get("runnable", True)),
        ))
    profiles = [OptimizationProfile(p["name"], int(p["ptxas_opt"]))
                for p in tc_cfg.get("optimization_profiles", [])]

    return Config(
        project_root=root,
        toolchains=toolchains,
        profiles=profiles,
        generation=pipeline_cfg.get("generation", {}),
        compilation=pipeline_cfg.get("compilation", {}),
        execution=pipeline_cfg.get("execution", {}),
        oracle=pipeline_cfg.get("oracle", {}),
        stress_gate=pipeline_cfg.get("stress_gate", {}),
        novelty=pipeline_cfg.get("novelty", {}),
        storage=pipeline_cfg.get("storage", {}),
        scheduling=pipeline_cfg.get("scheduling", {}),
        reporting=pipeline_cfg.get("reporting", {}),
        strategies=st_cfg.get("strategies", {}),
    )


def isolated_env(toolchain: Toolchain, extra_path: Optional[str] = None) -> Dict[str, str]:
    """Clean, isolated environment for one nvcc/ptxas invocation.

    Pins the toolkit as CUDA_HOME/CUDA_PATH, drops unrelated LD_LIBRARY_PATH pollution and
    forces the host compiler so cross-toolkit runs are reproducible.
    """
    env = {
        "PATH": f"{Path(toolchain.root) / 'bin'}:{os.environ.get('PATH', '')}",
        "CUDA_HOME": toolchain.root,
        "CUDA_PATH": toolchain.root,
        "LD_LIBRARY_PATH": f"{Path(toolchain.root) / 'lib64'}:{Path(toolchain.root) / 'lib'}",
        "HOME": os.environ.get("HOME", "/home/linux"),
        "NVCC_PREPEND_INCLUDE": os.environ.get("NVCC_PREPEND_INCLUDE", ""),
        "LANG": "C",
    }
    if extra_path:
        env["PATH"] = f"{extra_path}:{env['PATH']}"
    return env


def run_tool(argv, timeout: int = 120, cwd=None) -> subprocess.CompletedProcess:
    """Run a tool with isolated stderr capture; returns CompletedProcess."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=cwd)
