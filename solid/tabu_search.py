from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import deque
from numpy import argmax
from typing import Deque, Generic, TypeVar


S = TypeVar('S')  # Generic type for the state


class TabuSearch(ABC, Generic[S]):
    def __init__(self, initial_state: S, tabu_size: int, max_steps: int, max_score: float = None):
        """
        Abstract Base Class to conduct tabu search.
        :param initial_state: initial state, must implement __eq__ or __cmp__.
        :param tabu_size: number of states to keep in tabu list.
        :param max_steps: maximum number of steps to run algorithm for.
        :param max_score: score to stop algorithm once reached.
        """
        self.initial_state: S = initial_state
        if not isinstance(tabu_size, int) or tabu_size <= 0:
            raise TypeError('Tabu size must be a positive integer')
        self.tabu_size: int = tabu_size
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise TypeError('Maximum steps must be a positive integer')
        self.max_steps: int = max_steps
        self.max_score: float | None = None
        if max_score is not None:
            if not isinstance(max_score, (int, float)):
                raise TypeError('Maximum score must be a numeric type')
            self.max_score = float(max_score)
        self.current_state: S | None = None
        self.best_state: S | None = None
        self.tabu_list: Deque[S] = deque(maxlen=self.tabu_size)
        self.cur_steps: int = 0

    def __str__(self):
        return 'TABU SEARCH: \n' + \
                f'CURRENT STEPS: {self.cur_steps} \n' + \
                f'BEST SCORE: {self._score(self.best_state)} \n' + \
                f'BEST MEMBER: {str(self.best_state)} \n\n'

    def __repr__(self):
        return self.__str__()

    def _clear(self):
        """ Resets the variables that are altered on a per-run basis of the algorithm """
        self.current_state = None
        self.best_state = None
        self.tabu_list.clear()
        self.cur_steps = 0

    @abstractmethod
    def _score(self, state: S) -> float:
        """
        Returns objective function value of a state

        :param state: a state
        :return: objective function value of state
        """
        pass

    @abstractmethod
    def _neighborhood(self) -> list[S]:
        """
        Returns list of all members of neighborhood of current state, given self.current

        :return: list of members of neighborhood
        """
        pass

    def _best(self, neighborhood: list[S]) -> S:
        """
        Finds the best member of a neighborhood

        :param neighborhood: a neighborhood
        :return: best member of neighborhood
        """
        return neighborhood[argmax(self._score(x) for x in neighborhood)]

    def run(self, verbose: bool = True):
        """
        Conducts tabu search

        :param verbose: indicates whether to print progress regularly or not
        :return: best state and objective function value of best state
        """
        self._clear()
        self.current_state = deepcopy(self.initial_state)
        self.best_state = deepcopy(self.initial_state)

        for i in range(self.max_steps):
            self.cur_steps += 1

            if ((i + 1) % 100 == 0) and verbose:
                print(self)

            neighborhood = self._neighborhood()
            neighborhood_best = self._best(neighborhood)

            while True:
                if all(x in self.tabu_list for x in neighborhood):
                    print('TERMINATING - NO SUITABLE NEIGHBORS')
                    return self.best_state, self._score(self.best_state)
                if neighborhood_best in self.tabu_list:
                    if self._score(neighborhood_best) > self._score(self.best_state):
                        self.tabu_list.append(neighborhood_best)
                        self.best_state = deepcopy(neighborhood_best)
                        break
                    else:
                        neighborhood.remove(neighborhood_best)
                        neighborhood_best = self._best(neighborhood)
                else:
                    self.tabu_list.append(neighborhood_best)
                    self.current_state = neighborhood_best
                    if self._score(self.current_state) > self._score(self.best_state):
                        self.best_state = deepcopy(self.current_state)
                    break

            if self.max_score is not None and self._score(self.best_state) > self.max_score:
                print('TERMINATING - REACHED MAXIMUM SCORE')
                return self.best_state, self._score(self.best_state)
        print('TERMINATING - REACHED MAXIMUM STEPS')
        return self.best_state, self._score(self.best_state)
