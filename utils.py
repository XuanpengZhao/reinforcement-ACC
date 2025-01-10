import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def generate_leading_vehicle_trajectory(num_steps, time_step):
    trajectory = []
    position = 30
    speed = 0  # Initial speed
    for _ in range(num_steps):
        acceleration = np.random.uniform(-2, 2)  # Random acceleration between -2 and 2 m/s^2
        speed += acceleration * time_step
        speed = max(0, min(speed, 65))  # Keep speed between 0 and 65 m/s
        if speed == 0:
            acceleration = abs(acceleration)
        position += speed * time_step + 0.5 * acceleration * time_step**2
        trajectory.append({'position': position, 'speed': speed})
    return trajectory

def plot_trajectory(leading_trajectory, ego_trajectory=None, episode=None):
    plt.figure(figsize=(10, 5))
    plt.plot(leading_trajectory, label='Leading Vehicle Position')
    if ego_trajectory:
        plt.plot(ego_trajectory, label='Ego Vehicle Position')
    if episode:
        plt.title(f'Episode {episode} Trajectories')
    plt.xlabel('Time Step')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_metrics(episodes, rewards, avg_rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards, label='Episode Reward')
    plt.plot(episodes, avg_rewards, label='Average Reward')
    plt.title('Training Metrics')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()