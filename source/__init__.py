from gym.envs.registration import register

from .blackjack_env import BlackjackEnv

register(
    id='{}-{}'.format('BlackjackEnv', 'v0'),
    entry_point='source:{}'.format('BlackjackEnv'),
    max_episode_steps=100)
