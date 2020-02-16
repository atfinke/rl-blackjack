import itertools
from enum import Enum
from .card import Card


class Action(Enum):
    STAY = 1
    HIT = 1


def hand_values(cards):
    ranks = map(lambda x: x.ranks, cards)
    values = set()
    for ranks_combo in itertools.product(*ranks):
        values.add(sum(ranks_combo))

    return values


if __name__ == '__main__':
    from deck import Deck
    deck = Deck()
    cards = deck.cards[0:5]
    print(cards)
    print(hand_values(cards))
