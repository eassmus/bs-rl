from agents.agent import Agent

cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

class HumanAgent(Agent):
    def __init__(self, my_index, num_players, agent_args = []):
        pass

    def print_hand(self, hand): 
        sorted_hand = dict(sorted(hand.items(), key=lambda x: cards.index(x[0])))
        print(sorted_hand)

    def get_card(self, intended_card, hand) -> tuple[str, int]:
        
        self.print_hand(hand) 
        print("Intended card:", intended_card)
        while True:
            card = input("Choose card:").strip()
            if card in cards:
                while True:
                    num = int(input("Choose amount:").strip())
                    if hand[card] >= num and num >= 0:
                        return card, num
                    else:
                        print("Invalid amount")
            else:
                print("Invalid card")
        

    def get_call_bs(self, player_index, card, card_amt, hand) -> bool:
        print(player_index, "played", card_amt, card)
        self.print(hand)
        while True:
            response = input("Call BS (Y|N):")
            if response.strip() == "Y":
                return True
            elif response .strip()== "N":
                return False
            else:
                print("Invalid input")

    def give_info(self, player_indexes_picked_up):
        print("Player", player_indexes_picked_up, "picked up the pile")
    
    def reset(self):
        pass