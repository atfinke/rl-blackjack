from .card import Card
from .utils import hand_values


class Player:

    def __init__(self, name):
        self.name = name
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)

    def hand_values(self):
        return hand_values(self.cards)

    def reset(self):
        self.cards = []

    def is_busted(self):
        return min(self.hand_values()) > 21

    def has_blackjack(self):
        return 21 in self.hand_values()

    def rep_hand(self):
        return '{}: {}'.format(self.hand_values(), self.cards)

    def rep_only_first_card(self):
        return '{} {}: {{{}, ??}}'.format(self.name, set(self.cards[0].ranks), self.cards[0])

    def __repr__(self):
        return '{} {}'.format(self.name, self.rep_hand())
