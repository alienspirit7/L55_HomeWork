from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataCfg:
    window: int
    n_features: int
    normalization_split: str
    volume_norm_window: int
    split_train: float
    split_val: float
    split_test: float
    cache_ttl_hours: int


@dataclass(frozen=True)
class GatekeeperCfg:
    rate_limit_per_min: int
    rate_limit_per_hour: int
    rate_limit_concurrent: int
    rate_limit_burst: int
    rate_limit_burst_window_sec: int


@dataclass(frozen=True)
class EnvCfg:
    n_actions: int
    init_cash: float
    fee_bps: int


@dataclass(frozen=True)
class ModelCfg:
    gamma: float
    huber_delta: float


@dataclass(frozen=True)
class TrainCfg:
    lr: float
    batch: int
    buffer: int
    target_sync_steps: int
    eps_start: float
    eps_end: float
    eps_decay_steps: int
    train_steps: int
    eval_every: int
    grad_clip: float


@dataclass(frozen=True)
class EvalCfg:
    seeds: list[int]
    sharpe_annualization: int
    benchmark: str


@dataclass(frozen=True)
class Config:
    data: DataCfg
    gatekeeper: GatekeeperCfg
    env: EnvCfg
    model: ModelCfg
    train: TrainCfg
    eval: EvalCfg


_SECTION_TYPES: dict[str, type] = {
    "data": DataCfg,
    "gatekeeper": GatekeeperCfg,
    "env": EnvCfg,
    "model": ModelCfg,
    "train": TrainCfg,
    "eval": EvalCfg,
}


def _build(cls: type, raw: dict[str, Any]) -> Any:
    expected = {f.name for f in fields(cls)}
    unknown = set(raw) - expected
    if unknown:
        raise ValueError(f"unknown keys in {cls.__name__}: {sorted(unknown)}")
    missing = expected - set(raw)
    if missing:
        raise ValueError(f"missing keys in {cls.__name__}: {sorted(missing)}")
    return cls(**raw)


def load_config(path: str | Path) -> Config:
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    unknown = set(raw) - set(_SECTION_TYPES)
    if unknown:
        raise ValueError(f"unknown top-level config sections: {sorted(unknown)}")
    sections = {name: _build(cls, raw.get(name, {})) for name, cls in _SECTION_TYPES.items()}
    return Config(**sections)
