import gym
import sys
import numpy as np
import random
import itertools
import pickle
from source.deck import Deck
from source.card import Card

from timeit import default_timer as timer
from source import BlackjackEnv
from results import plot_no_player_aces, print_no_player_aces


sys.path.insert(1, './')
sys.path.append('/Users/andrewfinke/opt/miniconda3/envs/rl-blackjack/lib/python3.8/site-packages')
env = gym.make('BlackjackEnv-v0')
state = env.reset()


def player_hand_values_from_state(state):
    return frozenset(state[0])


def dealer_hand_values_from_state(state):
    return frozenset(state[1])


# def all_reasonable_possible_frozensets():
#     deck = Deck()
#     deck._remove_aces()
#     cards = deck.cards

#     sets = []
#     # 1 + 1 + 1 + 1 + 2 + 2 + 2 + 2 + 3 + 3 + 3
#     for combination in itertools.combinations(cards, 11):
#         value = 4 + sum(list(map(lambda x: list(x.ranks)[0], combination)))
#         if value > 21:
#             continue

#         for heart_ace in range(1):
#             for diamond_ace in range(1):
#                 for spade_ace in range(1):
#                     for clubs_ace in range(1):
#                         hand = []
#                         if heart_ace:
#                             hand.append(Card(suit="hearts", rank="A"))
#                         if diamond_ace:
#                             hand.append(Card(suit="diamonds", rank="A"))
#                         if spade_ace:
#                             hand.append(Card(suit="spades", rank="A"))
#                         if clubs_ace:
#                             hand.append(Card(suit="clubs", rank="A"))
#                         for card in combination:
#                             hand.append(card)
#                         sets.append(frozenset(hand))
#     print(1)


# all_possible_frozensets()


def str_for_action(action):
    if action == 0:
        return "hit"
    elif action == 1:
        return "stay"
    else:
        raise ValueError()


def _show_progress(progress, remaining_time):
    min_str = str(int(remaining_time / 60))
    sec_str = str(int(remaining_time % 60))
    per_str = str(round(progress, 1))
    print('ETA: {}:{} ({}%)'.format(min_str.rjust(2, '0'), sec_str.rjust(2, '0'), per_str.ljust(4, '0')))


def fit(env, steps=1_000, update_steps=100, save=True):
    print('fitting with steps: ' + str(steps))
    num_of_available_states = env.observation_space.n
    num_of_available_actions = env.action_space.n

    q = {}  # np.zeros((num_of_available_states, num_of_available_states, num_of_available_actions))
    n = {}  # np.zeros((num_of_available_states, num_of_available_states, num_of_available_actions))

    state = env.reset()
    rewards = np.zeros(steps)

    epsilon = 0.2
    discount = 0.9

    last_timer = None
    update_interval = update_steps
    diffs = np.zeros(5)
    diff_index = -1

    for i in range(steps):

        # Progress Updates
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
            last_timer = new

            _show_progress(progress=i / steps * 100, remaining_time=((steps - i) / update_interval) * diff)

        # if env.is_inital_deal_blackjack():
        #     state = env.reset()
        #     continue

        random = np.random.random(1)[0]

        action = None
        if random > epsilon:
            inital_player_hand_values = player_hand_values_from_state(state)
            inital_dealer_hand_values = dealer_hand_values_from_state(state)
            q_for_player = q.get(inital_player_hand_values, {})
            q_for_player_and_dealer = q_for_player.get(inital_dealer_hand_values, np.zeros(num_of_available_actions))
            action = np.argmax(q_for_player_and_dealer)
            if action == 0 and np.unique(q_for_player_and_dealer).size == 1:
                action = env.action_space.sample()
        else:
            inital_player_hand_values = player_hand_values_from_state(state)
            inital_dealer_hand_values = dealer_hand_values_from_state(state)
            action = env.action_space.sample()

        # print("action: " + str_for_action(action))
        new_state, reward, done, _ = env.step(action)
        rewards[i] = reward
        # env.render()

        existing_n_arr = n.get(inital_player_hand_values, {})
        existing_n_arr_and_dealer = existing_n_arr.get(inital_dealer_hand_values, np.zeros(num_of_available_actions))
        new_n = np.add(existing_n_arr_and_dealer[action], 1)
        existing_n_arr_and_dealer[action] = new_n
        existing_n_arr[inital_dealer_hand_values] = existing_n_arr_and_dealer
        n[inital_player_hand_values] = existing_n_arr

        alpha = np.divide(1, new_n)

        new_player_hand_values = player_hand_values_from_state(new_state)
        # Does this matter? Same as inital, should it be final dealer hand?
        new_dealer_hand_values = dealer_hand_values_from_state(new_state)

        # Discount * max(Q(S', a))
        q_for_new_player = q.get(new_player_hand_values, {})
        q_for_new_player_and_dealer = q_for_new_player.get(new_dealer_hand_values, np.zeros(num_of_available_actions))
        new_state_max_index = np.argmax(q_for_new_player_and_dealer)

        first_term = q_for_new_player_and_dealer[new_state_max_index]
        first_term = np.multiply(discount, first_term)

        # Q(S, A)
        q_for_player = q.get(inital_player_hand_values, {})
        q_for_player_and_dealer = q_for_player.get(inital_dealer_hand_values, np.zeros(num_of_available_actions))
        second_term = q_for_player_and_dealer[action]

        # R + Discount * max(Q(S', a))] - Q(S, A)
        combined_terms = np.subtract(first_term, second_term)
        combined_terms_with_reward = np.add(reward, combined_terms)

        alpha_and_terms = np.multiply(alpha, combined_terms_with_reward)

        q_for_player_and_dealer[action] += alpha_and_terms
        q_for_player[inital_dealer_hand_values] = q_for_player_and_dealer

        q[inital_player_hand_values] = q_for_player

        state = new_state
        if done:
            state = env.reset()

    if save:
        pickle.dump(q, open('q-{}.pkl'.format(steps), 'wb'))
    print('done fitting')
    return q


def predict(env, state_action_values):

    player_hand_values = None
    dealer_hand_value = None

    while player_hand_values not in state_action_values or dealer_hand_value not in state_action_values[player_hand_values]:
        state = env.reset()
        player_hand_values = player_hand_values_from_state(state)
        dealer_hand_value = dealer_hand_values_from_state(state)

    actions_for_state = state_action_values[player_hand_values][dealer_hand_value]
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
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        inital_player_hand_values = player_hand_values_from_state(state)
        inital_dealer_hand_values = dealer_hand_values_from_state(state)

        if not done:
            if inital_player_hand_values not in state_action_values or inital_dealer_hand_values not in state_action_values[inital_player_hand_values]:
                while inital_player_hand_values not in state_action_values or inital_dealer_hand_values not in state_action_values[inital_player_hand_values]:
                    state = env.reset()
                    inital_player_hand_values = player_hand_values_from_state(state)
                    inital_dealer_hand_values = dealer_hand_values_from_state(state)
                actions_for_state = state_action_values[inital_player_hand_values][inital_dealer_hand_values]

            actions_for_state = state_action_values[inital_player_hand_values][inital_dealer_hand_values]

        number_of_steps += 1

    env.render()
    return np.array(states), np.array(actions), np.array(rewards)


def test_prediction(q, steps=1_000):
    wins = 0
    losses = 0
    for _ in range(steps):
        result = predict(env, q)
        wins += np.count_nonzero(result[2] >= 1)
        losses += np.count_nonzero(result[2] == -1)

    print('predict {} results'.format(steps))
    print('wins:   {} ({}%)'.format(str(wins).rjust(len(str(steps))), str(round(wins / steps * 100, 2)).ljust(5, '0')))
    print('losses: {} ({}%)'.format(str(losses).rjust(len(str(steps))), str(round(losses / steps * 100, 2)).ljust(5, '0')))


def print_and_plot_existing_q_pickle(file_name):
    existing_q = pickle.load(open(file_name, "rb"))
    print_no_player_aces(existing_q)
    plot_no_player_aces(existing_q)


fit_steps = 100_000_000
update_steps = 50_000
predict_steps = 100_000

q = fit(env=env, steps=fit_steps, update_steps=update_steps)
test_prediction(q=q, steps=predict_steps)

env.close()

# print_and_plot_existing_q_pickle(file_name='q-100000000.pkl')
