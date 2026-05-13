import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')


def plot_blackjack(V):
    dealer = np.arange(1, 11)
    player = np.arange(12, 22)

    X, Y = np.meshgrid(dealer, player)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    Z1 = V[:, :, 0]

    ax1.plot_surface(X, Y, Z1, cmap='viridis')
    ax1.set_title("Usable Ace")
    ax1.set_xlabel("Dealer showing")
    ax1.set_ylabel("Player sum")
    ax1.set_zlabel("Value")

    ax2 = fig.add_subplot(122, projection='3d')
    Z2 = V[:, :, 1]

    ax2.plot_surface(X, Y, Z2, cmap='viridis')
    ax2.set_title("No Usable Ace")
    ax2.set_xlabel("Dealer showing")
    ax2.set_ylabel("Player sum")
    ax2.set_zlabel("Value")

    plt.tight_layout()
    plt.show()


def single_run_20():
    """ run the policy that sticks for >= 20 """
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    # It can be used for the subtasks
    # Use a comment for the print outputs to increase performance (only there as example)
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    ret = 0.
    while not done:
        #print("observation:", obs)
        states.append(obs)
        if obs[0] >= 20:
            #print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            #print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        #print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    #print("final observation:", obs)
    return states, ret


def policy_evaluation(maxiter=500000):
    """ Implementation of first-visit Monte Carlo prediction """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))
    for i in range(maxiter):
        print(f"{int((i + 1) / maxiter * 100)}%", end="\r", flush=True)
        episode = single_run_20()
        (states, final_reward) = episode

        # only the final reward counts, the rewards between the time steps is 0
        rewards = [0] * (len(states) - 1) + [final_reward]
        assert len(states) == len(rewards)

        G = 0
        for t in reversed(range(len(states))):
            G += rewards[t]
            state = states[t]

            if state in states[:t]:
                continue

            player_sum, dealer_card, usable_ace = state
            if player_sum < 12 or player_sum > 21:
                continue

            p = player_sum - 12
            d = dealer_card - 1
            u = 0 if usable_ace else 1

            returns[p][d][u] += G
            visits[p][d][u] += 1
            V[p][d][u] = returns[p][d][u] / visits[p][d][u]
    
    return V


def monte_carlo_es():
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    pi = np.zeros((10, 10, 2))
    # Q = np.zeros((10, 10, 2, 2))
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    maxiter = 100000000  # use whatever number of iterations you want
    for i in range(maxiter):
        if i % 100000 == 0:
            print("Iteration: " + str(i))
            print(pi[:, :, 0])
            print(pi[:, :, 1])


def main():
    V = policy_evaluation()
    plot_blackjack(V)
    # monte_carlo_es()


if __name__ == "__main__":
    main()
