import numpy as np

class ActionSpace:
    def __init__(self, min_acceleration, max_acceleration, num_actions):
        self.min_acceleration = min_acceleration
        self.max_acceleration = max_acceleration
        self.num_actions = num_actions
        self.actions = np.linspace(min_acceleration, max_acceleration, num_actions)
        print(self.actions)
    
    def get_action(self, action_idx):
        return self.actions[action_idx]