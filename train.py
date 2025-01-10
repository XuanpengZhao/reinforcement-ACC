from environment import ACCEnvironment
from action_space import ActionSpace
from reward_function import RewardFunction
from agent import DRLAgent
from utils import generate_leading_vehicle_trajectory, plot_trajectory, plot_metrics
import numpy as np
import tensorflow as tf

def train(num_episodes, steps_per_episode, learning_rate, epsilon, model_name, time_step=1):
    leading_vehicle_trajectory = generate_leading_vehicle_trajectory(steps_per_episode, time_step)
    positions = [step['position'] for step in leading_vehicle_trajectory]
    plot_trajectory(positions)
    env = ACCEnvironment(leading_vehicle_trajectory, time_step=time_step)
    action_space = ActionSpace(-3, 3, num_actions=60) # max 3m/s^2, min -3m/s^2
    reward_function = RewardFunction(
        target_distance=30,
        distance_weight=3,
        speed_variance_weight=1, 
        relative_speed_weight=2,
        power_weight=0.1,
        time_weight = 3
    )
    agent = DRLAgent(state_dim=5, action_dim=action_space.num_actions, learning_rate=learning_rate, epsilon=epsilon)
    
    all_rewards = []
    avg_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        for _ in range(steps_per_episode):
            step += 1
            action_idx = agent.get_action(state)
            action = action_space.get_action(action_idx)
            next_state, end_reason = env.step(action)
            done = 0
            if end_reason:
                done = 1
            reward = reward_function.calculate_reward(next_state, end_reason)
            
            agent.remember(state, action_idx, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
             
            
            if end_reason:
                break
        average_reward = total_reward/step
        if episode % 10 == 0:
            agent.update_target_model()
        
        all_rewards.append(average_reward)
        avg_rewards.append(np.mean(all_rewards[-100:]))  # Moving average of last 100 episodes
        print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {average_reward}, Steps: {step}, end_reason: {end_reason})")
      
        # Plot overall metrics every 10 episodes
        if ((episode + 1) in [40, 100] and step > 10)or (average_reward < -1e3 and end_reason != "reversing"):
            episodes = range(1, episode + 2)
            plot_metrics(episodes, all_rewards, avg_rewards)
            # Plot the trajectory at the end of each episode
            plot_trajectory(env.leading_trajectory, env.ego_trajectory, episode + 1)
    
    # Save the trained model
    agent.model.save(f'models/{model_name}.h5')
    
    # Save the leading vehicle trajectory for evaluation
    np.save(f'models/{model_name}_trajectory.npy', leading_vehicle_trajectory)

if __name__ == "__main__":
    # Train models with different parameters
    train(num_episodes=500, steps_per_episode=360, learning_rate=0.001, epsilon=1.0, model_name="model_2")
    # train(num_episodes=1000, steps_per_episode=200, learning_rate=0.0001, epsilon=0.5, model_name="model_2")