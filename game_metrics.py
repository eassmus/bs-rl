cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

class RoundPlayed:
    def __init__(self, player_index, total_turns, card_played, card_amt, bs_calls, was_bs, starting_hands, ending_hands, starting_pile, ending_pile):
        self.player_index = player_index
        self.total_turns = total_turns
        self.card_played = card_played
        self.card_amt = card_amt
        self.bs_calls = bs_calls
        self.was_bs = was_bs
        self.starting_hands = starting_hands
        self.ending_hands = ending_hands
        self.starting_pile = starting_pile
        self.ending_pile = ending_pile
    
    def __str__(self):
        out = ""
        out += ("-----------") + "\n"
        out += (f"Turn: {self.total_turns} Card to Play: {cards[self.total_turns % 13]}") + "\n"
        out += (f"Pile {self.starting_pile}") + "\n"
        out += (f"Player {self.player_index} current hand {self.starting_hands[self.player_index]}") + "\n"
        out += (f"Player {self.player_index} plays {self.card_played} {self.card_amt} time(s).") + "\n"
        out += (f"Player {self.player_index} new hand {self.ending_hands[self.player_index]}") + "\n"
        for player in self.bs_calls:
            out += (f"Player {player} bids BS") + "\n"
        if self.was_bs and len(self.bs_calls) > 0:
            out += (f"Player {self.player_index} takes the pile. Their new hand is {self.ending_hands[self.player_index]}") + "\n"
        elif not self.was_bs and len(self.bs_calls) > 0:
            out += (f"It was not BS. Splitting the pile {self.starting_pile + [self.card_played] * self.card_amt}") + "\n"
            for loser_index in self.bs_calls:
                out += (f"Player {loser_index} old hand: {self.starting_hands[loser_index]}") + "\n"
                out += (f"Player {loser_index} new hand: {self.ending_hands[loser_index]}") + "\n"
        return out
    
class GameMetrics:
    def __init__(self, rounds, num_players, decks, winner):
        self.num_players = num_players
        self.rounds = rounds
        self.decks = decks
        self.winner = winner

    def get_bs_rate(self, player_index): # returns correct, incorrect calls
        correct = 0
        incorrect = 0
        for round in self.rounds:
            if player_index in round.bs_calls == round.was_bs:
                correct += 1
            else:
                incorrect += 1
        return correct, incorrect

    def get_text(self):
        out = ""
        for round in self.rounds:
            out += round.__str__()
            out += "\n\n"
        return out
