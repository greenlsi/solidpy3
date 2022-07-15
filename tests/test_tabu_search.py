from random import choice, randint
from string import ascii_lowercase
from solid.tabu_search import TabuSearch
from copy import deepcopy


class Algorithm(TabuSearch[str]):
    """ Tries to get a randomly-generated string to match string 'clout' """
    def _neighborhood(self) -> list[str]:
        member = list(self.current_state)
        neighborhood: list[str] = []
        for _ in range(10):
            neighbor = deepcopy(member)
            neighbor[randint(0, 4)] = choice(ascii_lowercase)
            neighbor = ''.join(neighbor)
            neighborhood.append(neighbor)
        return neighborhood

    def _score(self, state):
        return float(sum(state[i] == "clout"[i] for i in range(5)))


if __name__ == '__main__':
    algorithm = Algorithm('abcde', 50, 500, max_score=None)
    algorithm.run()
