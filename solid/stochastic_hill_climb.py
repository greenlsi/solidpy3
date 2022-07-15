from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from math import exp
from random import random
from typing import TypeVar, Generic


S = TypeVar('S')  # Generic type for the state


class StochasticHillClimb(ABC, Generic[S]):
    def __init__(self, initial_state: S, temp: float, max_steps: int, max_objective: float = None):
        """
        Abstract Base Class to conduct stochastic hill climb.
        :param initial_state: initial state of hill climbing
        :param max_steps: maximum steps to run hill climbing for
        :param temp: temperature in probabilistic acceptance of transition
        :param max_objective: objective function to stop algorithm once reached
        """
        self.initial_state: S = initial_state
        if not isinstance(temp, (float, int)):
            raise ValueError('Temperature must be a numeric type')
        self.temp = float(temp)
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError('Max steps must be a positive integer')
        self.max_steps = max_steps
        self.max_objective: float | None = None
        if max_objective is not None:
            if not isinstance(max_objective, (float, int)):
                raise ValueError('Maximum objective must be a numeric type')
            self.max_objective = float(max_objective)
        self.current_state: S | None = None
        self.best_state: S | None = None
        self.best_objective: float | None = None
        self.cur_steps: int = 0

    def __str__(self):
        return 'STOCHASTIC HILL CLIMB: \n' + \
               f'CURRENT STEPS: {self.cur_steps} \n' + \
               f'BEST OBJECTIVE: {self.best_objective} \n' + \
               f'BEST STATE: {str(self.best_state)} \n\n'

    def __repr__(self):
        return self.__str__()

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm
        """
        self.current_state = None
        self.best_state = None
        self.best_objective = None
        self.cur_steps = 0

    @abstractmethod
    def _neighbor(self) -> S:
        """
        Returns a random member of the neighbor of the current state

        :return: a random neighbor, given access to self.current_state
        """
        pass

    @abstractmethod
    def _objective(self, state: S) -> float:
        """
        Evaluates a given state

        :param state: a state
        :return: objective function value of state
        """
        pass

    def _accept_neighbor(self, neighbor: S) -> bool:
        """
        Probabilistically determines whether to accept a transition to a neighbor or not

        :param neighbor: a state
        :return: boolean indicating whether transition was accepted or not
        """
        try:
            p = 1. / (1 + (exp((self._objective(self.current_state) - self._objective(neighbor)) / self.temp)))
        except OverflowError:
            return True
        return True if p >= 1 else p >= random()

    def run(self, verbose: bool = True):
        """
        Conducts hill climb

        :param verbose: indicates whether or not to print progress regularly
        :return: best state and best objective function value
        """
        self._clear()
        self.current_state = self.initial_state
        for i in range(self.max_steps):
            self.cur_steps += 1

            if ((i + 1) % 100 == 0) and verbose:
                print(self)

            neighbor = self._neighbor()

            if self._accept_neighbor(neighbor):
                self.current_state = neighbor

            if self._objective(self.current_state) > (self.best_objective or 0):
                self.best_objective = self._objective(self.current_state)
                self.best_state = deepcopy(self.current_state)

            if self.max_objective is not None and (self.best_objective or 0) > self.max_objective:
                print("TERMINATING - REACHED MAXIMUM OBJECTIVE")
                return self.best_state, self.best_objective
        print("TERMINATING - REACHED MAXIMUM STEPS")
        return self.best_state, self.best_objective
