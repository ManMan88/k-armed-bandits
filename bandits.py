import numpy as np

class Bandit:
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
    
    def get_reward(self):
        return np.random.normal(self.mean,self.std)

class KArmedBandits:
    def __init__(self,k,stationary):
        self.k = k
        self.stationary = stationary
        self.bandits = []
        if stationary:
            self.initialize_bandits()

    def initialize_bandits(self):
        self.bandits = []
        self.optimal_bandit = -1
        max_mean = -1000
        for i in range(self.k):
            mean = np.random.normal(0,1)
            self.bandits.append(Bandit(mean,1))
            if mean > max_mean:
                self.optimal_bandit = i # assuming there's only one optimal bandit
                max_mean = mean
    
    def get_bandit_reward(self,bandit_num):
        if not self.stationary:
            self.initialize_bandits()
        return self.bandits[bandit_num].get_reward()
            