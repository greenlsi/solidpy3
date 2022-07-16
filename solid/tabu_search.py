from __future__ import annotations
import multiprocessing
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Callable, Deque, Generic
from .utils import S, F, parallel_score


class TabuSearch(ABC, Generic[S, F]):
    def __init__(self, initial_state: S, score_f: Callable[[S], F], tabu_size: int, n_neighbors: int):
        """
        Abstract Base Class to conduct tabu search.
        :param initial_state: initial state, must implement __eq__.
        :param score_f: score function to be used. The return value must implement __gt__.
        :param tabu_size:
        :param n_neighbors:
        """
        self.initial_state: S = initial_state
        self.score_f: Callable[[S], F] = score_f
        if tabu_size < 1:
            raise ValueError('tabu_size must be greater than 0')
        self.tabu_size: int = tabu_size
        if n_neighbors < 1:
            raise ValueError('n_neighbors must be greater than 0')
        self.n_neighbors: int = n_neighbors

        self.current_steps: int = 0
        self.current_state: S | None = None
        self.current_score: F | None = None
        self.best_state: S | None = None
        self.best_score: F | None = None

    def __str__(self):
        return 'TABU SEARCH:\n' + \
               f'CURRENT STEPS: {self.current_steps}\n' + \
               f'BEST MEMBER: {str(self.best_state)}\n' + \
               f'BEST SCORE: {self.best_score}\n\n'

    def __repr__(self):
        return self.__str__()

    def _clear(self):
        """ Resets the variables that are altered on a per-run basis of the algorithm """
        self.current_steps = 0
        self.current_state = None
        self.current_score = None
        self.best_state = None
        self.best_score = None

    @abstractmethod
    def _neighbor(self) -> S:
        """
        Returns list of all members of neighborhood of current state, given self.current_steps
        :return: list of members of neighborhood
        """
        pass

    def _neighborhood(self, parallel: bool) -> list[tuple[S, F]]:
        """

        :param parallel:
        :return:
        """
        if parallel:
            manager = multiprocessing.Manager()
            plist = manager.list()
            jobs = list()
            for _ in range(self.n_neighbors):
                neighbor = self._neighbor()
                p = multiprocessing.Process(target=parallel_score, args=(self.score_f, neighbor, plist))
                p.start()
                jobs.append(p)
            for t in jobs:
                t.join()
            return [(neighbor, score) for neighbor, score in plist]
        else:
            neighbors = [self._neighbor() for _ in range(self.n_neighbors)]
            return [(neighbor, self.score_f(neighbor)) for neighbor in neighbors]

    def run(self, max_steps: int, max_score: F = None, parallel: bool = False, verbose: bool = True):
        """
        Conducts tabu search.
        :param max_steps: maximum number of steps to explore.
        :param max_score: score to stop algorithm once reached.
        :param parallel:
        :param verbose: indicates whether to print progress regularly or not
        :return: best state and objective function value of best state
        """
        self._clear()
        tabu_list: Deque[S] = deque(maxlen=self.tabu_size)
        self.current_state = deepcopy(self.initial_state)
        self.current_score = self.score_f(self.current_state)
        self.best_state, self.best_score = deepcopy(self.initial_state), self.current_score

        for i in range(max_steps):
            self.current_steps += 1
            if verbose and ((i + 1) % 10 == 0):
                print(self)
            neighborhood: list[tuple[S, F]] = self._neighborhood(parallel)
            neighborhood.sort(key=lambda x: x[1])  # Now, the last element of the list contains the best neighborhood
            while neighborhood:
                neighbor_state, neighbor_score = neighborhood[-1]
                if neighbor_state in tabu_list:
                    if neighbor_score > self.best_score:
                        tabu_list.append(neighbor_state)
                        self.best_state, self.best_score = deepcopy(neighbor_state), neighbor_score
                        break
                    else:
                        neighborhood.pop()
                else:
                    tabu_list.append(neighbor_state)
                    self.current_state, self.current_score = neighbor_state, neighbor_score
                    if self.current_score > self.best_score:
                        self.best_state, self.best_score = deepcopy(self.current_state), self.current_score
                    break
            if not neighborhood:
                print('TERMINATING - NO SUITABLE NEIGHBORS')
                print(self)
                return self.best_state, self.best_score
            if max_score is not None and self.best_score >= max_score:
                print('TERMINATING - REACHED MAXIMUM SCORE')
                print(self)
                return self.best_state, self.best_score
        print('TERMINATING - REACHED MAXIMUM STEPS')
        print(self)
        return self.best_state, self.best_score
