import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

from .utils import Action
from .blackjack import Blackjack


class BlackjackEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.game = Blackjack(printing=False)
        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Discrete(100)

    def step(self, action):
        if action == 0:
            reward = self.game.player_hit()
        elif action == 1:
            reward = self.game.player_done()
        else:
            raise ValueError()

        return [self.game.player.hand_values(), self.game.dealer.cards[0].ranks], reward, self.game.is_player_turn_over, {}

    def reset(self):
        self.game.reset()
        self.game.deal()
        return [self.game.player.hand_values(), self.game.dealer.cards[0].ranks]

    def render(self, mode='human', close=False):
        # print(self.game.player)
        # print(self.game.dealer)
        # self.game.printing = True
        # self.game.calculate_reward()
        # self.game.printing = False
        pass

    def is_inital_deal_blackjack(self):
        return 21 in self.game.player.hand_values()
