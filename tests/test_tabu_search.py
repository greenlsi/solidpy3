from random import choice, randint
from string import ascii_lowercase
from solid.tabu_search import TabuSearch
from copy import deepcopy


def my_score(state: str) -> int:
    return sum(state[i] == "clout"[i] for i in range(5))


class Algorithm(TabuSearch[str, int]):
    def __init__(self, initial_state: str, tabu_size: int, n_neighbors: int):
        super().__init__(initial_state, my_score, tabu_size, n_neighbors)

    """ Tries to get a randomly-generated string to match string 'clout' """
    def _neighbor(self) -> str:
        neighbor: list[str] = list(deepcopy(self.current_state))
        neighbor[randint(0, 4)] = choice(ascii_lowercase)
        return ''.join(neighbor)


if __name__ == '__main__':
    algorithm = Algorithm('aaaaa', 50, 10)
    algorithm.run(max_steps=500, max_score=5, parallel=True)
