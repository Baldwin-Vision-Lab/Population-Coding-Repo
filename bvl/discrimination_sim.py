from numpy import random

class SimulatedObserver:

    def __init__(self, biasAlpha, slopeSigma, lapseLambda):
        self.biasAlpha   = biasAlpha
        self.slopeSigma  = slopeSigma
        self.lapseLambda = lapseLambda

    def __simulate_decision__(self, noisyResponse):
        # handle random behaviour during lapses
        if random.rand() < self.lapseLambda:
            if random.rand() < 0.5:
                return True
            else:
                return False
        # if not lapsing, make decision (with bias)
        if noisyResponse < (-1*self.biasAlpha):
            return True
        else:
            return False

    def get_response(self, expectedResponseValue):
        # response randomly sampled from normal distribution
        noisyResponse = random.randn() * self.slopeSigma - expectedResponseValue
        return self.__simulate_decision__(noisyResponse)