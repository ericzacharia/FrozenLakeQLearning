import numpy as np
import gym
import time
from lake_envs import *

""" 
For the algorithms the common parameters P, nS, nA, gamma are
defined below. Additional parameters are defined along with each
algorithm.

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [0, nS-1] and actions in [0, nA-1],
        P[state][action] is a list of tuples of the form
        (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """
    value_function = np.zeros(nS)
    while True:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            a = policy[s]
            value = 0
            for outcome in P[s][a]:
                probability = outcome[0]
                reward = outcome[2]
                next_state_value = value_function[outcome[1]]
                value += probability * (reward + gamma * next_state_value)
            value_function[s] = value
            delta = max(delta, abs(v-value_function[s]))
        if delta < tol:
            break
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """
    old_policy = policy.copy()
    for s in range(nS):
        max_value = 0 
        for a in range(nA):
            value = 0
            for outcome in P[s][a]:
                probability = outcome[0]
                reward = outcome[2]
                next_state_value = value_from_policy[outcome[1]]
                value += probability * (reward + gamma * next_state_value)
            if value > max_value:
                max_value = value
                policy[s] = a
    policy_stable = all(old_policy == policy)

    return policy, policy_stable


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    # Step 1: Initialization
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    while True:
        # Step 2: Policy Evaluation
        value_function = policy_evaluation(
            P, nS, nA, policy, gamma, tol)
        # Step 3: Policy Improvement
        policy, policy_stable = policy_improvement(
            P, nS, nA, value_function, policy, gamma)
        if policy_stable:
            break
    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    while True:
        delta = 0
        for s in range(nS):
            v = value_function[s]
            optimal_policy_value = 0
            for a in range(nA):
                value = 0
                for outcome in P[s][a]:
                    probability = outcome[0]
                    reward = outcome[2]
                    next_state_value = value_function[outcome[1]]
                    value += probability * (reward + gamma * next_state_value)
                if value > optimal_policy_value:
                    optimal_policy_value = value
                    policy[s] = a
            value_function[s] = optimal_policy_value
            delta = max(delta, abs(v-value_function[s]))
        if delta < tol:
            break
    return value_function, policy


def sarsa(env, nS, nA, gamma=0.9, epsilon=0.8, alpha=0.1, max_steps=100,
          max_episodes=10000):
    """
    Learn the action value function and policy by the sarsa
    method for a given environment, gamma, epsilon, alpha, and a given
    maximum number of episodes to train for, and a given maximum number
    of steps to train for in an episode.

    Parameters:
    ----------
    env: environment object (which is not the same as P) 
    nS, nA, gamma: defined at beginning of file
    episilon: for the epsilon greedy policy used
    alpha: step update parameter
    max_steps: maximum number of steps to train for in an episode
    max_episodees: maximum number of episodes to train for

    Returns:
    ----------
    q_function: np.ndarray[nS, nA]
    policy: np.ndarray[nS]
    """
    q_function = np.zeros((nS, nA))
    policy = np.zeros(nS, dtype=int)

    for _ in range(max_episodes):
        state = env.reset()

        is_terminal = False
        for _ in range(max_steps):
            if np.random.uniform(0, 1) > epsilon:
                action = np.argmax(q_function[state, :])
                policy[state] = action
            else:
                action = np.random.randint(0, nA)

            next_state, reward, is_terminal, info = env.step(action)

            if np.random.uniform(0, 1) > epsilon:
                next_action = np.argmax(q_function[next_state, :])
            else:
                next_action = np.random.randint(0, nA)

            q_function[state, action] += alpha * (reward + gamma *
               q_function[next_state, next_action] - q_function[state, action])

            state = next_state

            if is_terminal:
                break

    return q_function, policy


def q_learning(env, nS, nA, gamma=0.9, epsilon=0.8, alpha=0.1, max_steps=100,
               max_episodes=10000):
    """
    Learn the action value function and policy by the q-learning
    method for a given environment, gamma, epsilon, alpha, and a given
    maximum number of episodes to train for, and a given maximum number
    of steps to train for in an episode.

    Parameters:
    ----------
    env: environment object (which is not the same as P) 
    nS, nA, gamma: defined at beginning of file
    episilon: for the epsilon greedy policy used
    alpha: step update parameter
    max_steps: maximum number of steps to train for in an episode
    max_episodees: maximum number of episodes to train for

    Returns:
    ----------
    q_function: np.ndarray[nS, nA]
    policy: np.ndarray[nS]
    """
    q_function = np.zeros((nS, nA))
    policy = np.zeros(nS, dtype=int)

    for _ in range(max_episodes):
        state = env.reset()

        is_terminal = False
        for _ in range(max_steps):
            if np.random.uniform(0, 1) > epsilon:
                action = np.argmax(q_function[state,:])
                policy[state] = action
            else:
                action = np.random.randint(0, nA)

            next_state, reward, is_terminal, info = env.step(action)

            q_function[state, action] += alpha * (reward + gamma * 
               np.max(q_function[next_state, :]) - q_function[state, action])

            state = next_state

            if is_terminal:
                break

    return q_function, policy


def double_q_learning(env, nS, nA, gamma=0.9, epsilon=0.8, alpha=0.1, max_steps=100,
               max_episodes=10000):
    """
    Learn the action value function and policy by the q-learning
    method for a given environment, gamma, epsilon, alpha, and a given
    maximum number of episodes to train for, and a given maximum number
    of steps to train for in an episode.

    Parameters:
    ----------
    env: environment object (which is not the same as P) 
    nS, nA, gamma: defined at beginning of file
    episilon: for the epsilon greedy policy used
    alpha: step update parameter
    max_steps: maximum number of steps to train for in an episode
    max_episodees: maximum number of episodes to train for

    Returns:
    ----------
    q_function: np.ndarray[nS, nA]
    policy: np.ndarray[nS]
    """
    q_function1 = np.zeros((nS, nA))
    q_function2 = np.zeros((nS, nA))
    policy = np.zeros(nS, dtype=int)

    for _ in range(max_episodes):
        state = env.reset()            
        is_terminal = False
        for _ in range(max_steps):

            def step(qA, qB):
                nonlocal state, is_terminal
                if np.random.uniform(0, 1) > epsilon:
                    action = np.argmax(qA[state, :])
                    policy[state] = action
                else:
                    action = np.random.randint(0, nA)
                next_state, reward, is_terminal, info = env.step(action)
                qA[state, action] += alpha * (reward + gamma *
                    np.max(qB[next_state, :]) - qA[state, action])
                state = next_state

            if np.random.uniform(0, 1) > 0.5:
                step(q_function1, q_function2)
            else:
                step(q_function2, q_function1)

            if is_terminal:
                break

    return q_function1, q_function2, policy

def render_single(env, policy, max_steps=100, show_rendering=True):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        if show_rendering:
            env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    if show_rendering:
        env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)


def evaluate(env, policy, max_steps=100, max_episodes=32):
    """
    This function does not need to be modified,
    evaluates your policy over multiple episodes.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_rewards = []
    dones = []
    for _ in range(max_episodes):
        episode_reward = 0
        ob = env.reset()
        for t in range(max_steps):
            a = policy[ob]
            ob, rew, done, _ = env.step(a)
            episode_reward += rew
            if done:
                break

        episode_rewards.append(episode_reward)
        dones.append(done)

    episode_rewards = np.array(episode_rewards).mean()
    terminates = np.array(dones).mean()

    print(
        f"> Average reward over {max_episodes} episodes:\t\t\t {episode_rewards}")
    print(
        f"> Percentage of episodes a terminal state reached:\t\t\t {terminates * 100:.0f}%")
    return episode_rewards
