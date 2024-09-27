import env
import random

class RandomAgent(env.Agent):
    def __init__(self, my_index, num_players):
        pass 

    def get_card(self, intended_card, hand):
        return random.choice(hand), 1
    
    def get_call_bs(self, player_index, card, card_amt, hand):
        return random.choice([True, False])

    def give_info(self, player_indexes_picked_up):
        pass


env = env.BSEnv(agent_types = [RandomAgent, RandomAgent, RandomAgent, RandomAgent])
game_results = env.run_game()
[print(a) for a in game_results.rounds]
print(len(game_results.rounds))