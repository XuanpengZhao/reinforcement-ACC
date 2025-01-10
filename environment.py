import numpy as np

class ACCEnvironment:
    def __init__(self, leading_vehicle_trajectory, friction_coefficient=0.02, time_step=0.1):
        self.leading_vehicle_trajectory = leading_vehicle_trajectory
        self.time_step = time_step
        self.current_step = 0
        self.friction_coefficient = friction_coefficient
        self.max_steps = len(leading_vehicle_trajectory)
        self.ego_vehicle_state = {
            'position': 0,
            'speed': 0,
            'acceleration': 0
        }
        self.ego_trajectory = []
        self.leading_trajectory = []
    
    def reset(self):
        self.current_step = 0
        self.ego_vehicle_state = {
            'position': 0,
            'speed': 0,
            'acceleration': 0
        }
        self.ego_trajectory = [0]  # Start with initial position
        self.leading_trajectory = [self.leading_vehicle_trajectory[0]['position']]
        return self._get_state()
    
    def step(self, action):
        self.ego_vehicle_state['acceleration'] = action - self.friction_coefficient * self.ego_vehicle_state['speed']
        self.ego_vehicle_state['speed'] += self.ego_vehicle_state['acceleration'] * self.time_step
        self.ego_vehicle_state['position'] += self.ego_vehicle_state['speed'] * self.time_step + 0.5 * self.ego_vehicle_state['acceleration'] * self.time_step**2
         
        self.current_step += 1
        self.ego_trajectory.append(self.ego_vehicle_state['position'])
 
        self.leading_trajectory.append(self.leading_vehicle_trajectory[self.current_step]['position'])
        
        state = self._get_state()
        relative_distance = state[1]
        speed = state[2]
        acceleration = state[0]
        end_reason = False
        if relative_distance <= 0:
            end_reason = "collision"
        elif speed < 0 or (speed == 0 and acceleration < 0):
            end_reason = "reversing"
        elif self.current_step > self.max_steps - 2:
            end_reason = "episode_end"
        
        return state, end_reason
    
    def _get_state(self):
        leading_vehicle_position = self.leading_vehicle_trajectory[self.current_step]['position']
        leading_vehicle_speed = self.leading_vehicle_trajectory[self.current_step]['speed']
        
        relative_distance = leading_vehicle_position - self.ego_vehicle_state['position']
        relative_speed = leading_vehicle_speed - self.ego_vehicle_state['speed']
        
        return np.array([
            self.ego_vehicle_state['acceleration'],
            relative_distance,
            self.ego_vehicle_state['speed'],
            relative_speed,
            self.current_step
        ])
    
    def _is_collision(self, state):
        relative_distance = state[1]
        if relative_distance <= 0:
            return "collision"
        return False
    
    def _is_reversing(self, state):
        speed = state[2]
        acceleration = state[0]
        return speed < 0 or (speed == 0 and acceleration < 0)