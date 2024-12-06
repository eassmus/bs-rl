from agents.agent import Agent
from agents.smarter_simple_agent import SmartSimpleAgent
from agents.bs_call_learning_agent import BSCallLearningAgent

class SmartSimpleAgentBSLearner(Agent):
    def __init__(self, my_index, num_players, agent_args = []):
        self.smart_agent = SmartSimpleAgent(my_index, num_players, agent_args)
        self.bs_agent = BSCallLearningAgent(my_index, num_players, agent_args)

    def get_card(self, intended_card, hand):
        self.bs_agent.get_card(intended_card, hand)
        return self.smart_agent.get_card(intended_card, hand)
    
    def get_call_bs(self, player_index, card, card_amt, hand):
        self.smart_agent.get_call_bs(player_index, card, card_amt, hand)
        return self.bs_agent.get_call_bs(player_index, card, card_amt, hand)

    def give_info(self, player_indexes_picked_up):
        self.smart_agent.give_info(player_indexes_picked_up)
        self.bs_agent.give_info(player_indexes_picked_up)

    def give_full_info(self, was_bs):
        self.smart_agent.give_full_info(was_bs)
        self.bs_agent.give_full_info(was_bs)

    def reset(self):
        self.smart_agent.reset()
        self.bs_agent.reset()

    def give_winner(self, winner):
        self.smart_agent.give_winner(winner)
        self.bs_agent.give_winner(winner)
    