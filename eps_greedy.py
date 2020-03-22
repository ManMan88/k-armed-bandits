import numpy as np
import sys

class GeneralMethod:
    def __init__(self,k,initial_estimate,step_size):
        self.k = k
        self.estimates = [initial_estimate]*k
        self.num_of_steps = [0]*k
        self.step_size = step_size
    
    def argmax_a(self):
        return self.estimates.index(max(self.estimates))
    
    def update_estimate(self,ind,target):
        if self.step_size == "n":
            if self.num_of_steps[ind] != 0:
                self.estimates[ind] = self.estimates[ind] + (1/self.num_of_steps[ind])*(target-self.estimates[ind])
        else:
            self.estimates[ind] = self.estimates[ind] + float(self.step_size)*(target-self.estimates[ind])
        
        self.num_of_steps[ind] += 1

class Egreedy(GeneralMethod):
    def __init__(self,k,initial_estimate,step_size,eps):
        super().__init__(k,initial_estimate,step_size)
        self.eps = eps

    def select_action(self):
        max_a = self.argmax_a()
        action = max_a
        if np.random.uniform() < self.eps:
            while action==max_a:
                action = int(np.random.uniform()*self.k)
        return action