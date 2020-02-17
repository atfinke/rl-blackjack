import gym
import sys
import numpy as np
import random

from timeit import default_timer as timer
from source import BlackjackEnv

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


sys.path.insert(1, './')
sys.path.append('/Users/andrewfinke/opt/miniconda3/envs/rl-blackjack/lib/python3.8/site-packages')
env = gym.make('BlackjackEnv-v0')
state = env.reset()


def player_hand_values_from_state(state):
    return sorted(list(state[0]))


def dealer_hand_values_from_state(state):
    return sorted(list(state[1]))


def str_for_action(action):
    if action == 0:
        return "hit"
    elif action == 1:
        return "stay"
    else:
        raise ValueError()


def fit(env, steps=1_000):
    print('fitting with steps: ' + str(steps))
    num_of_available_states = env.observation_space.n
    num_of_available_actions = env.action_space.n

    q = np.zeros((num_of_available_states, num_of_available_states, num_of_available_actions))
    n = np.zeros((num_of_available_states, num_of_available_states, num_of_available_actions))

    state = env.reset()
    rewards = np.zeros(steps)

    epsilon = 0.2
    discount = 0.95

    last_timer = None
    update_interval = 20_000

    diffs = np.zeros(5)
    diff_index = -1

    for i in range(steps):
        if i % update_interval == 0:
            diff_index += 1
            if diff_index >= diffs.shape[0]:
                diff_index = 0

            new = timer()
            if last_timer:
                diffs[diff_index] = new - last_timer
                if i == update_interval:
                    diffs.fill(diffs[diff_index])

            diff = np.mean(diffs)
            remaining_time = ((steps - i) / update_interval) * diff
            min_str = str(int(remaining_time / 60))
            sec_str = str(int(remaining_time % 60))
            per_str = str(round(i / steps * 100, 1))
            last_timer = new

            print('ETA: {}:{} ({}%)'.format(min_str.rjust(2, '0'), sec_str.rjust(2, '0'), per_str.ljust(4, '0')))

        if env.is_inital_deal_blackjack():
            state = env.reset()
            continue

        random = np.random.random(1)[0]

        action = None
        if random > epsilon:
            first_player_hand_value = random.choice(player_hand_values_from_state(state))
            first_dealer_hand_value = dealer_hand_values_from_state(state)[0]

            # Hmm, need to figure out how to account for the other value (with ace)
            hmmm = q[first_player_hand_value][first_dealer_hand_value]
            action = np.argmax(hmmm)
            if action == 0 and np.unique(hmmm).size == 1:
                action = env.action_space.sample()
        else:
            action = env.action_space.sample()

        # print("action: " + str_for_action(action))
        new_state, reward, done, _ = env.step(action)
        rewards[i] = reward
        # env.render()

        player_hand_values = player_hand_values_from_state(state)
        first_dealer_hand_value = dealer_hand_values_from_state(state)[0]

        for hand_state in player_hand_values:
            if hand_state > 21:
                continue

            new_n = np.add(n[hand_state][first_dealer_hand_value][action], 1)
            n[hand_state][first_dealer_hand_value][action] = new_n
            alpha = np.divide(1, new_n)

            new_player_hand_values = player_hand_values_from_state(new_state)
            new_first_dealer_hand_values = dealer_hand_values_from_state(new_state)
            for new_player_hand_state in new_player_hand_values:
                for new_dealer_hand_state in new_first_dealer_hand_values:
                    if new_player_hand_state > 21 or new_dealer_hand_state > 21:
                        # Not sure if this is correct for q learning to just dump the result...
                        continue
                    else:
                        new_state_max_index = np.argmax(q[new_player_hand_state][new_dealer_hand_state])
                        first_term = q[new_player_hand_state][new_dealer_hand_state][new_state_max_index]
                        first_term = np.multiply(discount, first_term)

                    second_term = q[hand_state][first_dealer_hand_value][action]
                    combined_terms = np.subtract(first_term, second_term)
                    combined_terms_with_reward = np.add(reward, combined_terms)

                    alpha_and_terms = np.multiply(alpha, combined_terms_with_reward)

                    q[hand_state][first_dealer_hand_value][action] += alpha_and_terms

        state = new_state
        if done:
            state = env.reset()

    return q


def predict(env, state_action_values):
    state = env.reset()

    first_player_hand_value = player_hand_values_from_state(state)[0]
    first_dealer_hand_value = dealer_hand_values_from_state(state)[0]

    actions_for_state = state_action_values[first_player_hand_value][first_dealer_hand_value]
    done = False

    number_of_steps = 0

    states = list()
    actions = list()
    rewards = list()

    while not done:
        if env.is_inital_deal_blackjack():
            state = env.reset()
            continue

        action = np.argmax(actions_for_state)

        env.render()
        # print('performing action: ' + str_for_action(action))

        state, reward, done, _ = env.step(action)

        first_player_hand_value = player_hand_values_from_state(state)[0]
        first_dealer_hand_value = dealer_hand_values_from_state(state)[0]

        if not done:
            actions_for_state = state_action_values[first_player_hand_value][first_dealer_hand_value]

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        number_of_steps += 1

    env.render()
    return np.array(states), np.array(actions), np.array(rewards)


fit_steps = 500_000
q = fit(env, fit_steps)
print('done fitting')

for player_start in range(1, 22):
    for dealer_start in range(1, 12):
        values = q[player_start][dealer_start]
        hit = str(round(values[0], 3)).ljust(6, '0')
        stay = str(round(values[1], 3)).ljust(6, '0')
        print('{} / {}: [{}, {}]'.format(str(player_start).rjust(2, ' '), str(dealer_start).rjust(2, ' '), hit, stay))

predict_steps = 100_000
wins = 0
losses = 0
for _ in range(predict_steps):
    result = predict(env, q)
    wins += np.count_nonzero(result[2] >= 1)
    losses += np.count_nonzero(result[2] == -1)


print('predict {} results'.format(predict_steps))
print('wins:   {} ({}%)'.format(str(wins).rjust(len(str(predict_steps))), str(round(wins / predict_steps * 100, 2)).ljust(5, '0')))
print('losses: {} ({}%)'.format(str(losses).rjust(len(str(predict_steps))), str(round(losses / predict_steps * 100, 2)).ljust(5, '0')))

env.close()
