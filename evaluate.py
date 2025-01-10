from environment import ACCEnvironment
from action_space import ActionSpace
from reward_function import RewardFunction
from utils import generate_leading_vehicle_trajectory, plot_trajectory, plot_metrics
import numpy as np
import tensorflow as tf

def evaluate(model_name, num_episodes=10, steps_per_episode=50, time_step=1):
    # Load the saved model
    # model = tf.keras.models.load_model(f'models/{model_name}.h5', custom_objects={'mse': tf.keras.losses.mean_squared_error})
    model = tf.keras.models.load_model(f'models/{model_name}.h5', 
                                   custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    # Generate a new leading vehicle trajectory for evaluation
    leading_vehicle_trajectory = generate_leading_vehicle_trajectory(steps_per_episode, time_step)
    
    # Setup environment and action space
    env = ACCEnvironment(leading_vehicle_trajectory, time_step=time_step)
    action_space = ActionSpace(-3, 3, num_actions=10)
    
    # Setup reward function (same as in training)
    reward_function = RewardFunction(
        target_distance=30,
        distance_weight=3,
        speed_variance_weight=1, 
        relative_speed_weight=2,
        power_weight=0.1,
        time_weight=3
    )
    
    all_rewards = []
    all_steps = []
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        step = 0
        
        for _ in range(steps_per_episode):
            step += 1
            action_idx = np.argmax(model.predict(state.reshape(1, -1))[0])
            action = action_space.get_action(action_idx)
            next_state, end_reason = env.step(action)
            reward = reward_function.calculate_reward(next_state, end_reason)
            
            state = next_state
            total_reward += reward
            
            if end_reason:
                break
        
        average_reward = total_reward / step
        all_rewards.append(average_reward)
        all_steps.append(step)
        
        print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {average_reward}, Steps: {step}, end_reason: {end_reason}")
        
        # Plot the trajectory for each episode
        plot_trajectory(env.leading_trajectory, env.ego_trajectory, episode + 1)
    
    # Plot overall metrics
    episodes = range(1, num_episodes + 1)
    plot_metrics(episodes, all_rewards, np.cumsum(all_rewards) / np.arange(1, len(all_rewards) + 1))
    
    print(f"\nEvaluation Results for {model_name} on new trajectory:")
    print(f"Average Reward: {np.mean(all_rewards):.2f}")
    print(f"Average Steps: {np.mean(all_steps):.2f}")

if __name__ == "__main__":
    evaluate("model_1")