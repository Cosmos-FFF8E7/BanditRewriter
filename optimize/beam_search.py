

import numpy as np
from typing import List, Callable, TypeVar, Any

T = TypeVar("T")


def beam_search(
    candidates: List[T], scoring_function: Callable[[T], float], beam_width: int
) -> List[T]:

    scores: List[float] = [scoring_function(candidate) for candidate in candidates]

    if len(candidates) <= beam_width:
        return candidates

    sorted_indices: np.ndarray = np.argsort(scores)

    selected_indices: np.ndarray = sorted_indices[-beam_width:]

    return [candidates[i] for i in selected_indices]
