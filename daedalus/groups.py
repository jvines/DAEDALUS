"""Toggleable model components on the MoMS state space.

Each `Group` declares a subset of the continuous parameter vector beta whose
inclusion in the model is governed by a single binary indicator gamma_k. When
gamma_k = 0, the parameters in the group are held at their off-values; when
gamma_k = 1, they are sampled from their continuous prior.

Optional fields:
    birth_proposal:
        A `BirthProposal` implementation invoked whenever the trans-dim
        kernel attempts to flip gamma_k from 0 to 1. None (the default)
        selects the uniform-u birth: draw u ~ Uniform(0, 1) for the
        group's u-coords and push through the user's prior_transform.
        Under uniform-u the proposal density and continuous prior cancel
        in the M-H ratio, so the chain only needs the inclusion-prior
        ratio.

    log_prior_continuous:
        Required when `birth_proposal` is supplied. A callable
        ``log_prior_continuous(beta_for_group) -> float`` that returns
        the user's continuous prior density on the group's parameters
        in physical beta-space. Needed because a custom proposal
        density no longer cancels with the prior, so the M-H ratio
        becomes ``inclusion_ratio * pi_beta(beta) / q_beta(beta)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from .birth_proposals import BirthProposal


@dataclass
class Group:
    name: str
    params: Sequence[int]
    off_values: np.ndarray
    inclusion_prior: float = 0.5
    init_scale: float = 1.0
    birth_proposal: "BirthProposal | None" = None
    log_prior_continuous: Callable[[np.ndarray], float] | None = None

    def __post_init__(self) -> None:
        self.off_values = np.asarray(self.off_values, dtype=float)
        if len(self.params) != self.off_values.size:
            raise ValueError(
                f"Group {self.name!r}: len(params)={len(self.params)} does not match "
                f"len(off_values)={self.off_values.size}"
            )
        if not 0.0 < self.inclusion_prior < 1.0:
            raise ValueError(
                f"Group {self.name!r}: inclusion_prior must be in (0, 1), "
                f"got {self.inclusion_prior}"
            )
        if self.birth_proposal is not None and self.log_prior_continuous is None:
            raise ValueError(
                f"Group {self.name!r}: when birth_proposal is supplied, "
                f"log_prior_continuous must also be supplied (the "
                f"continuous prior no longer cancels with the proposal "
                f"density in the M-H ratio)."
            )

    @property
    def ndim(self) -> int:
        return len(self.params)
