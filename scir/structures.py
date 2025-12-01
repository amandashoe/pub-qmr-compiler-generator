from dataclasses import dataclass
from enum import Enum
from typing import Optional
from fractions import Fraction

class BoundaryType(Enum):
    Z = 1
    X = 2


class PatchType(Enum):
    ALGORITHM = 1
    ANCILLA = 2
    MAGIC_T = 3


@dataclass(frozen=True)
class Location:
    x: int
    y: int


@dataclass(frozen=True)
class Patch:
    name: str  # TODO can we disallow this from being replaced?
    location: Location
    top_bottom: Optional[BoundaryType]
    left_right: Optional[BoundaryType]
    patch_type: PatchType


@dataclass(frozen=True)
class Operation:
    name: str
    qubits: list[str]
    routing_qubits: list[str]
    magic_state_qubits: list[str]


@dataclass(frozen=True)
class Step:
    start_time: Fraction
    operations: list[Operation]


@dataclass(frozen=True)
class Architecture:
    width: int
    height: int
    initial_patches: list[Patch]


@dataclass(frozen=True)
class Schedule:
    arch: Architecture
    steps: list[Step]
    total_time: Optional[Fraction]