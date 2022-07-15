from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from math import exp
from random import random
from typing import Generic, TypeVar


S = TypeVar('S')  # Generic type for the state


class SimulatedAnnealing(ABC, Generic[S]):
    def __init__(self, initial_state: S, start_temp, schedule_constant: float, max_steps: int,
                 min_energy: float = None, schedule: str = 'exponential'):
        """
        Abstract Base Class to conduct simulated annealing algorithms.
        :param initial_state: initial state of annealing algorithm
        :param max_steps: maximum number of iterations to conduct annealing for
        :param start_temp: beginning temperature
        :param schedule_constant: constant value in annealing schedule function
        :param min_energy: energy value to stop algorithm once reached
        :param schedule: 'exponential' or 'linear' annealing schedule
        """
        self.initial_state: S = initial_state
        if not isinstance(start_temp, (float, int)):
            raise ValueError('Starting temperature must be a numeric type')
        self.start_temp: float = float(start_temp)
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError('Max steps must be a positive integer')
        self.max_steps: int = max_steps
        self.min_energy: float | None = None
        if min_energy is not None:
            if not isinstance(min_energy, (float, int)):
                raise ValueError('Minimum energy must be a numeric type')
            self.min_energy = float(min_energy)

        self.adjust_temp = self._get_schedule(schedule, schedule_constant)

        self.current_state: S | None = None
        self.best_state: S | None = None
        self.cur_steps: int = 0
        self.current_energy: float | None = None
        self.best_energy: float | None = None
        self.current_temp: float | None = None

    def __str__(self):
        return 'SIMULATED ANNEALING: \n' + \
                f'CURRENT STEPS: {self.cur_steps} \n' + \
                f'CURRENT TEMPERATURE: {self.current_temp} \n' + \
                f'BEST ENERGY: {self.best_energy} \n' + \
                f'BEST STATE: {str(self.best_state)} \n\n'

    def __repr__(self):
        return self.__str__()

    def _get_schedule(self, schedule_str: str, schedule_constant: float):
        if schedule_str == 'exponential':
            return self._exponential(schedule_constant)
        elif schedule_str == 'linear':
            return self._linear(schedule_constant)
        else:
            raise ValueError('Annealing schedule must be either "exponential" or "linear"')

    def _exponential(self, schedule_constant: float):
        def f():
            self.current_temp *= schedule_constant
        return f

    def _linear(self, schedule_constant: float):
        def f():
            self.current_temp -= schedule_constant
        return f

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm

        :return: None
        """
        self.cur_steps = 0
        self.current_state = None
        self.best_state = None
        self.current_energy = None
        self.best_energy = None

    @abstractmethod
    def _neighbor(self) -> S:
        """
        Returns a random member of the neighbor of the current state

        :return: a random neighbor, given access to self.current_state
        """
        pass

    @abstractmethod
    def _energy(self, state: S) -> float:
        """
        Finds the energy of a given state

        :param state: a state
        :return: energy of state
        """
        pass

    def _accept_neighbor(self, neighbor: S):
        """
        Probabilistically determines whether or not to accept a transition to a neighbor

        :param neighbor: a state
        :return: boolean indicating whether or not transition is accepted
        """
        try:
            p = exp(-(self._energy(neighbor) - self._energy(self.current_state)) / self.current_temp)
        except OverflowError:
            return True
        return True if p >= 1 else p >= random()

    def run(self, verbose: bool = True):
        """
        Conducts simulated annealing

        :param verbose: indicates whether to print progress regularly or not
        :return: best state and best energy
        """
        self._clear()
        self.current_state = self.initial_state
        self.current_temp = self.start_temp
        self.best_energy = self._energy(self.current_state)
        for i in range(self.max_steps):
            self.cur_steps += 1

            if verbose and ((i + 1) % 100 == 0):
                print(self)

            neighbor = self._neighbor()

            if self._accept_neighbor(neighbor):
                self.current_state = neighbor
            self.current_energy = self._energy(self.current_state)

            if self.current_energy < self.best_energy:
                self.best_energy = self.current_energy
                self.best_state = deepcopy(self.current_state)

            if self.min_energy is not None and self.current_energy < self.min_energy:
                print("TERMINATING - REACHED MINIMUM ENERGY")
                return self.best_state, self.best_energy

            self.adjust_temp()
            if self.current_temp < 0.000001:
                print("TERMINATING - REACHED TEMPERATURE OF 0")
                return self.best_state, self.best_energy
        print("TERMINATING - REACHED MAXIMUM STEPS")
        return self.best_state, self.best_energy
