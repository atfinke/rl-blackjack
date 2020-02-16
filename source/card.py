class Card:

    def __init__(self, suit, rank):
        self.suit = suit
        self.str_rank = rank
        try:
            self.ranks = set([int(rank)])
        except:
            if rank == "J" or rank == "Q" or rank == "K":
                self.ranks = set([10])
            elif rank == "A":
                self.ranks = set([1, 11])
            else:
                raise ValueError(rank)

    def __repr__(self):
        return self.str_rank + self._pretty_suit()

    def _pretty_suit(self):
        if self.suit == "clubs":
            return "♣"
        elif self.suit == "diamonds":
            return "♦"
        elif self.suit == "hearts":
            return "♥"
        elif self.suit == "spades":
            return "♠"
        else:
            raise ValueError(self.suit)


if __name__ == '__main__':
    card = Card("diamonds", "A")
    print(card)
    card = Card("spades", "5")
    print(card)
