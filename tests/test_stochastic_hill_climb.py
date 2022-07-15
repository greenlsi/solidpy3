from numpy import array
from random import uniform
from solid.stochastic_hill_climb import StochasticHillClimb


class Algorithm(StochasticHillClimb[list[float]]):
    """
    Tries to get a randomly-generated list to match [.1, .2, .3, .2, .1]
    """
    def _neighbor(self):
        return list(array(self.current_state) + array([uniform(-.02, .02) for _ in range(5)]))

    def _objective(self, state):
        return 1. / sum(abs(state[i] - [.1, .2, .3, .2, .1][i]) if (state[i] > 0 or state[i] < 0) else 1 for i in range(5))


if __name__ == '__main__':
    algorithm = Algorithm(list([uniform(0, 1) for _ in range(5)]), .01, 1000)
    algorithm.run()
