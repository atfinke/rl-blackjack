from .player import Player
from .deck import Deck


class Blackjack:

    def __init__(self, printing=True):
        self.dealer = Player('dealer')
        self.player = Player('human')
        self.deck = Deck()
        self.printing = printing
        self.is_round_over = False

    def reset(self):
        self.dealer.reset()
        self.player.reset()
        self.deck.reset()
        self.is_round_over = False

    def deal(self):
        card = self.deck.draw()
        self.player.add_card(card)
        card = self.deck.draw()
        self.dealer.add_card(card)

        card = self.deck.draw()
        self.player.add_card(card)
        card = self.deck.draw()
        self.dealer.add_card(card)

        self.print_if_enabled(self.player)
        self.print_if_enabled(self.dealer.rep_only_first_card())

    def can_player_hit(self):
        return self._can_hit(self.player)

    def player_hit(self):
        assert self._can_hit(self.player)

        card = self.deck.draw()
        self.player.add_card(card)

        self.print_if_enabled(self.player.rep_hand())
        self.is_player_turn_over = not self.can_player_hit()
        if self.is_player_turn_over:
            return self.calculate_reward()
        else:
            return 0

    def player_done(self):
        self.is_player_turn_over = True
        return self.calculate_reward()

    def calculate_reward(self):
        if self.player.is_busted():
            self.print_if_enabled('player busted')
            return -1

        self.dealer_finish()
        if self.player.has_blackjack() and self.dealer.has_blackjack():
            return 0
        elif self.player.has_blackjack():
            self.print_if_enabled('player blackjack')
            return 1.5
        elif self.dealer.is_busted():
            self.print_if_enabled('dealer busted')
            return 1
        elif self.dealer.has_blackjack():
            self.print_if_enabled('dealer blackjack')
            return -1

        player_value = max(filter(lambda x: x <= 21, self.player.hand_values()))
        dealer_value = max(filter(lambda x: x <= 21, self.dealer.hand_values()))
        if player_value == dealer_value:
            self.print_if_enabled("tie!")
            return 0

        player_wins = player_value > dealer_value
        if player_wins:
            self.print_if_enabled('{} wins!'.format(self.player.name))
            return 1
        else:
            self.print_if_enabled('{} wins!'.format(self.dealer.name))
            return - 1

    def dealer_finish(self):
        # self.print_if_enabled("\ndealer's turn...")
        # self.print_if_enabled(self.dealer.rep_hand())
        while self._can_hit(self.dealer) and (max(self.dealer.hand_values()) < 17 or (max(self.dealer.hand_values()) > 21 and min(self.dealer.hand_values()) < 17)):
            player_value = max(filter(lambda x: x <= 21, self.player.hand_values()))
            dealer_value = max(filter(lambda x: x <= 21, self.dealer.hand_values()))
            if dealer_value > player_value:
                break
            card = self.deck.draw()
            self.dealer.add_card(card)
            # self.print_if_enabled(self.dealer.rep_hand())

    def _can_hit(self, player):
        if min(player.hand_values()) > 21:
            return False
        else:
            return True

    def print_if_enabled(self, string):
        if not self.printing:
            return
        print(string)


if __name__ == '__main__':
    game = Blackjack()
    while True:
        print("\n====================")
        game.reset()
        game.deal()

        while game.can_player_hit():
            i = input("hit? ")
            hit = i == "" or i == "y"
            if hit:
                game.player_hit()
            else:
                break

        reward = game.player_done()
        print(reward)

        i = input("again? ")
        again = i == "" or i == "y"
        if not again:
            break
