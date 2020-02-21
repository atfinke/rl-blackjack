from .card import Card
from random import shuffle


class Deck:

    def __init__(self):
        self._cards = []
        for suit in ["clubs", "diamonds", "hearts", "spades"]:
            for rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]:
                card = Card(suit, rank)
                self._cards.append(card)
        self.reset()

    def reset(self):
        self.cards = self._cards.copy()
        shuffle(self.cards)

    def draw(self):
        card = self.cards[0]
        self.cards = self.cards[1:]
        return card

    def _remove_aces(self):
        self.cards = filter(lambda card: card.str_rank != "A", self.cards)


if __name__ == '__main__':
    deck = Deck()
    print(deck.cards)
    print(deck.draw())
    print(deck.draw())
    deck.reset()
    print(deck.cards)
