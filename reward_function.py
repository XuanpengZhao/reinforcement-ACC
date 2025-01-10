import numpy as np

class RewardFunction:
    def __init__(self, target_distance, distance_weight, speed_variance_weight, relative_speed_weight, power_weight, time_weight, window_size=10):
        self.target_distance = target_distance
        self.distance_weight = distance_weight
        self.speed_variance_weight = speed_variance_weight
        self.relative_speed_weight = relative_speed_weight
        self.power_weight = power_weight
        self.time_weight = time_weight
        self.window_size = window_size
        self.speed_history = []

    def calculate_reward(self, state, end_reason):
        if end_reason == "collision":
            return -1000
        elif end_reason == "reversing":
            return -1000
         

        acceleration, relative_distance, speed, relative_speed, current_step = state

        # Update speed history
        self.speed_history.append(speed)
        if len(self.speed_history) > self.window_size:
            self.speed_history.pop(0)

        # Calculate speed variance
        speed_variance = np.var(self.speed_history) if len(self.speed_history) > 1 else 0

        # Calculate relative speed term
        relative_speed_term = abs(relative_speed)

        # Calculate power consumption (simplified model)
        power_consumption = abs(acceleration * speed)

        # Calculate reward components
        distance_reward = -self.distance_weight * abs(relative_distance - self.target_distance)
        speed_variance_reward = -self.speed_variance_weight * speed_variance
        relative_speed_reward = -self.relative_speed_weight * relative_speed_term
        power_reward = -self.power_weight * power_consumption
        time_reward = self.time_weight * current_step

        total_reward = distance_reward + speed_variance_reward + relative_speed_reward + power_reward + time_reward

        return total_reward