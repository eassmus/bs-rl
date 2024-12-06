from agents.agent import Agent

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

class HumanAgent(Agent):
    def __init__(self, my_index, num_players, agent_args = []):
        self.my_index = my_index

    def print_hand(self, hand): 
        out = []
        for card in cards:
            out = out + [card] * hand[card]
        print("Your hand:", out)

    def get_card(self, intended_card, hand) -> tuple[str, int]:
        print("\nYour turn!")
        self.print_hand(hand) 
        print("Intended card:", intended_card)
        while True:
            card = input("Choose card: ").strip().upper()
            if card in cards:
                while True:
                    num = int(input("Choose amount: ").strip())
                    if hand[card] >= num and num >= 0:
                        return card, num
                    else:
                        print("Invalid amount")
            else:
                print("Invalid card")
        

    def get_call_bs(self, player_index, card, card_amt, hand) -> bool:
        print()
        print("Player:", str(player_index) + "'s", "turn")
        print(player_index, "played", card_amt, card + "s")
        self.print_hand(hand)
        while True:
            response = input("Call BS (Y|N): ")
            if response.strip().upper() == "Y":
                return True
            elif response.strip().upper()== "N":
                return False
            else:
                print("Invalid input")

    def give_info(self, player_indexes_picked_up):
        print("Player", player_indexes_picked_up, "picked up the pile")
    
    def give_full_info(self, was_bs):
        pass

    def reset(self):
        pass

    def give_winner(self, winner):
        print("Player", winner, "is the winner")