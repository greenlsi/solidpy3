from __future__ import annotations
import multiprocessing
from abc import ABC, abstractmethod
from collections import deque
from copy import deepcopy
from typing import Callable, Deque, Generic
from .utils import S, F, parallel_score
import time

def time_string(seconds):
    """Returns time in seconds as a string formatted HHHH:MM:SS."""
    s = int(round(seconds))  # round to nearest second
    h, s = divmod(s, 3600)   # get hours and remainder
    m, s = divmod(s, 60)     # split remainder into minutes and seconds
    return '%4i:%02i:%02i' % (h, m, s)

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
        self.tabu_list: Deque[S] = deque(maxlen=self.tabu_size)
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

    def file_update(self):
        elapsed = time.time() - self.start
        file = open("tabusearch_log.txt","a")
        file.write('\r%5.0f  %13.2f  %s  %16.0f\r' %
                   (self.current_steps, self.best_score, time_string(elapsed), self.best_iteration))

    def _clear(self):
        """ Resets the variables that are altered on a per-run basis of the algorithm """
        self.current_steps = 0
        self.start = time.time()
        self.tabu_list.clear()
        self.current_state = None
        self.current_score = None
        self.best_state = None
        self.best_score = None
        self.best_iteration = 0
        file = open("tabusearch_log.txt","w")
        file.write(' Step          Score     Elapsed    Best iteration')

    @abstractmethod
    def _neighbor(self) -> S:
        """
        Returns a new neighbor of current state.
        :return: new neighbor
        """
        pass

    def _neighborhood(self, parallel: bool) -> list[tuple[S, F]]:
        """
        Creates self.n_neighbors new neighbors. Those who are already in the tabu list are dropped.
        The remaining neighbors are evaluated according to the score function, self.f_score.
        :param parallel: if true, it evaluates all the valid neighbors in parallel using multiprocessing.
        :return: list of tuples with new valid neighbors and their score.
        """
        # First, we create the list of neighbors
        neighbors: list[S] = list()
        for _ in range(self.n_neighbors):  # As much as n_neighbors
            neighbor = self._neighbor()
            if neighbor not in self.tabu_list:  # If new neighbor is already in tabu list, we drop it
                neighbors.append(neighbor)
        # If parallel, we evaluate the new neighbors using multiprocessing
        if parallel:
            manager = multiprocessing.Manager()
            plist = manager.list()
            jobs = list()
            for neighbor in neighbors:
                p = multiprocessing.Process(target=parallel_score, args=(self.score_f, neighbor, plist))
                p.start()
                jobs.append(p)
            for t in jobs:
                t.join()
            return [(neighbor, score) for neighbor, score in plist]
        # Otherwise, we do it sequentially
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
        self.current_state = deepcopy(self.initial_state)
        self.current_score = self.score_f(self.current_state)
        self.best_state, self.best_score = deepcopy(self.initial_state), self.current_score

        for i in range(max_steps):            
            if verbose and ((i + 1) % 1 == 0):
                print(self)
                self.file_update()
            self.current_steps += 1
            # First, we create and sort a new neighborhood
            neighborhood: list[tuple[S, F]] = self._neighborhood(parallel)
            # If neighborhood is empty, then there are no suitable neighbors
            if not neighborhood:
                print('TERMINATING - NO SUITABLE NEIGHBORS')
                print(self)
                return self.best_state, self.best_score
            # Otherwise, we sort them according to the value of their score function
            neighborhood.sort(key=lambda x: x[1])
            # the best neighborhood is at the back of the list
            neighbor_state, neighbor_score = neighborhood[-1]
            # New best neighbor is added to the tabu list.
            self.tabu_list.append(neighbor_state)
            self.current_state, self.current_score = neighbor_state, neighbor_score
            # If new best neighbor is better than the currently best, we update it
            if self.current_score > self.best_score:
                self.best_state, self.best_score = deepcopy(self.current_state), self.current_score
                self.best_iteration = neighbor_state.n_index
            # If we reach the maximum score, we stop the optimization
            if max_score is not None and self.best_score >= max_score:
                print('TERMINATING - REACHED MAXIMUM SCORE')
                print(self)
                return self.best_state, self.best_score
        print('TERMINATING - REACHED MAXIMUM STEPS')
        print(self)
        return self.best_state, self.best_score
