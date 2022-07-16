from __future__ import annotations
from typing import Callable, TypeVar


S = TypeVar('S')  # Generic type for the state
F = TypeVar('F')  # Generic type for the fitness function


def parallel_score(score_f: Callable[[S], F], state: S, plist: list):
    plist.append((state, score_f(state)))
