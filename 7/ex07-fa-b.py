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


def q_learning(
    env,
    episodes=5000,
    alpha=0.2,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.999,
    epsilon_min=0.05
):
    """
    Q-learning with state aggregation.
    Returns:
        successes: list of 0/1 values
        steps_per_episode: number of steps in each episode
    """

    n_actions = env.action_space.n

    # optimistic initialization
    Q = np.ones((N_POS, N_VEL, n_actions))

    successes = []
    steps_per_episode = []

    for episode in range(episodes):

        state = env.reset()

        # gym / gymnasium compatibility
        if isinstance(state, tuple):
            state = state[0]

        s = discretize_state(state)

        done = False
        steps = 0
        goal_reached = False

        while not done:

            # epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[s])

            result = env.step(action)

            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            s_next = discretize_state(next_state)

            goal_reached = next_state[0] >= 0.5

            # Q-learning update
            if goal_reached:
                target = reward
            else:
                target = reward + gamma * np.max(Q[s_next])

            Q[s][action] += alpha * (target - Q[s][action])

            s = s_next
            steps += 1

        successes.append(1 if goal_reached else 0)
        steps_per_episode.append(steps)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % 500 == 0:
            success_rate = np.mean(successes[-500:])

            print(
                f"Episode {episode + 1:5d} | "
                f"Success rate (last 500): {success_rate:.3f} | "
                f"Epsilon: {epsilon:.3f}"
            )

    return successes, steps_per_episode


def run_experiments(num_runs=10, episodes=5000):

    all_successes = []
    all_steps = []

    for run in range(num_runs):

        print("\n" + "=" * 50)
        print(f"Run {run + 1}/{num_runs}")
        print("=" * 50)

        env = gym.make("MountainCar-v0")

        successes, steps = q_learning(
            env,
            episodes=episodes,
            alpha=0.2,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.999,
            epsilon_min=0.05
        )

        env.close()

        all_successes.append(successes)
        all_steps.append(steps)

    all_successes = np.array(all_successes)
    all_steps = np.array(all_steps)

    # cumulative successes for each run
    cumulative_successes = np.cumsum(all_successes, axis=1)

    # average over runs
    avg_cumulative_successes = np.mean(
        cumulative_successes,
        axis=0
    )

    avg_steps = np.mean(
        all_steps,
        axis=0
    )

    return avg_cumulative_successes, avg_steps


def plot_results(avg_cumulative_successes, avg_steps):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---------------------------------
    # Averaged cumulative successes
    # ---------------------------------
    axes[0].plot(avg_cumulative_successes)

    axes[0].set_title(
        "Average Cumulative Number of Successes"
    )

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Successes")
    axes[0].grid(True)

    # ---------------------------------
    # Averaged steps per episode
    # ---------------------------------
    axes[1].plot(avg_steps)

    axes[1].set_title(
        "Average Steps per Episode"
    )

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def main():

    avg_cumulative_successes, avg_steps = run_experiments(
        num_runs=10,
        episodes=5000
    )

    plot_results(
        avg_cumulative_successes,
        avg_steps
    )


if __name__ == "__main__":
    main()