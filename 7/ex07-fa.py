import gym
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# State aggregation parameters
# -----------------------------
N_POS = 20
N_VEL = 20

POS_MIN = -1.2
POS_MAX = 0.6

VEL_MIN = -0.07
VEL_MAX = 0.07


def discretize_state(state):
    """
    Maps continuous state (position, velocity)
    to discrete grid indices.
    """
    pos, vel = state

    pos_bin = int(
        np.clip(
            (pos - POS_MIN) / (POS_MAX - POS_MIN) * N_POS,
            0,
            N_POS - 1
        )
    )

    vel_bin = int(
        np.clip(
            (vel - VEL_MIN) / (VEL_MAX - VEL_MIN) * N_VEL,
            0,
            N_VEL - 1
        )
    )

    return pos_bin, vel_bin


def get_value_function(Q):
    """
    Computes V(s) = max_a Q(s,a)
    """
    return np.max(Q, axis=2)


def plot_all_value_functions(value_snapshots, episodes):
    """
    Plot all value-function snapshots in a single figure.
    """
    n = len(value_snapshots)

    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(18, 15),
        constrained_layout=True
    )

    axes = np.array(axes).flatten()

    im = None

    for i, (V, ep) in enumerate(zip(value_snapshots, episodes)):
        im = axes[i].imshow(
            V.T,
            origin="lower",
            aspect="auto",
            extent=[POS_MIN, POS_MAX, VEL_MIN, VEL_MAX]
        )

        axes[i].set_title(f"Episode {ep}")
        axes[i].set_xlabel("Position")
        axes[i].set_ylabel("Velocity")

    # Remove unused axes
    for i in range(len(value_snapshots), len(axes)):
        fig.delaxes(axes[i])

    # Shared colorbar
    fig.colorbar(
        im,
        ax=axes[:len(value_snapshots)],
        shrink=0.85,
        label="V(s)"
    )

    fig.suptitle("Evolution of the Value Function", fontsize=16)

    plt.show()


def plot_learning_curve(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curve")
    plt.grid(True)
    plt.show()


def q_learning(
    env,
    episodes=500,
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
):
    """
    Q-learning with state aggregation.
    """

    n_actions = env.action_space.n

    # Q[position_bin, velocity_bin, action]
    Q = np.zeros((N_POS, N_VEL, n_actions))

    rewards = []

    value_snapshots = []
    snapshot_episodes = []

    for episode in range(episodes):

        state = env.reset()

        # Gym compatibility
        if isinstance(state, tuple):
            state = state[0]

        s = discretize_state(state)

        total_reward = 0
        done = False

        while not done:

            # epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[s])

            step_result = env.step(action)

            # Gym/Gymnasium compatibility
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result

            s_next = discretize_state(next_state)

            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + gamma * np.max(Q[s_next])

            Q[s][action] += alpha * (target - Q[s][action])

            s = s_next
            total_reward += reward

        rewards.append(total_reward)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Save value function every 20 episodes
        if (episode + 1) % 20 == 0:

            avg_reward = np.mean(rewards[-20:])

            print(
                f"Episode {episode + 1:4d} | "
                f"Avg Reward: {avg_reward:7.2f} | "
                f"Epsilon: {epsilon:.3f}"
            )

            value_snapshots.append(get_value_function(Q))
            snapshot_episodes.append(episode + 1)

    return Q, rewards, value_snapshots, snapshot_episodes


def main():
    env = gym.make("MountainCar-v0")

    Q, rewards, value_snapshots, snapshot_episodes = q_learning(
        env,
        episodes=400,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    env.close()

    # Plot all value-function snapshots in one window
    plot_all_value_functions(
        value_snapshots,
        snapshot_episodes
    )

    # Plot learning curve
    plot_learning_curve(rewards)


if __name__ == "__main__":
    main()