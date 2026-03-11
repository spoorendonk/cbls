"""Solution pool with restart strategies + parallel search."""

from __future__ import annotations

import threading
import time
import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cbls.dag import full_evaluate
from cbls.violation import ViolationManager
from cbls.search import (
    SearchResult, initialize_random, fj_nl_initialize, solve,
)

if TYPE_CHECKING:
    from cbls.model import Model


@dataclass
class Solution:
    """A stored solution."""
    state: dict
    objective: float
    feasible: bool


class SolutionPool:
    """Thread-safe pool of best solutions found across all threads."""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.solutions: list[Solution] = []
        self.lock = threading.Lock()

    def submit(self, solution: Solution) -> bool:
        """Submit a solution. Returns True if it was added to pool."""
        with self.lock:
            self.solutions.append(solution)
            # Sort: feasible first, then by objective
            self.solutions.sort(
                key=lambda s: (not s.feasible, s.objective)
            )
            if len(self.solutions) > self.capacity:
                self.solutions = self.solutions[:self.capacity]
            return solution in self.solutions

    def best(self) -> Solution | None:
        with self.lock:
            return self.solutions[0] if self.solutions else None

    def get_restart_point(self, rng: np.random.Generator) -> Solution | None:
        """Get a solution for restart (random from top half of pool)."""
        with self.lock:
            if not self.solutions:
                return None
            n = max(1, len(self.solutions) // 2)
            idx = rng.integers(0, n)
            return self.solutions[idx]


class ParallelSearch:
    """Run independent SA threads sharing a solution pool."""

    def __init__(self, n_threads: int = 4):
        self.n_threads = n_threads

    def solve(self, model_factory, time_limit: float = 10.0,
              seed: int = 42) -> SearchResult:
        """
        Run parallel search.

        model_factory: callable that returns a fresh (Model, close=True) instance.
            Each thread needs its own model to avoid data races.
        """
        pool = SolutionPool()
        threads = []
        results: list[SearchResult] = [None] * self.n_threads

        for i in range(self.n_threads):
            t = threading.Thread(
                target=self._worker,
                args=(model_factory, pool, seed + i, time_limit, results, i),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Return best from pool
        best = pool.best()
        if best is not None:
            return SearchResult(
                objective=best.objective,
                feasible=best.feasible,
                best_state=best.state,
            )
        # Fallback: return best from any thread
        valid = [r for r in results if r is not None]
        if valid:
            return min(valid, key=lambda r: (not r.feasible, r.objective))
        return SearchResult()

    def _worker(self, model_factory, pool: SolutionPool,
                seed: int, time_limit: float,
                results: list, thread_idx: int):
        """Worker thread: build model, solve, submit to pool."""
        try:
            model = model_factory()
            result = solve(model, time_limit=time_limit, seed=seed)
            results[thread_idx] = result

            sol = Solution(
                state=result.best_state,
                objective=result.objective,
                feasible=result.feasible,
            )
            pool.submit(sol)
        except Exception as e:
            # Don't crash the whole search
            print(f"Thread {thread_idx} failed: {e}")
