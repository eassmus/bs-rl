from agents.agent import Agent

from agents.ppo_agent import PPOAgent
from agents.bs_call_learning_agent import BSCallLearningAgent

class PPOAgentBSLearner(Agent):
    def __init__(self, my_index, num_players, agent_args = []):
        self.ppo_agent = PPOAgent(my_index, num_players, agent_args)
        self.bs_agent = BSCallLearningAgent(my_index, num_players, agent_args)

    def get_card(self, intended_card, hand) -> tuple[str, int]:
        self.bs_agent.get_card(intended_card, hand)
        return self.ppo_agent.get_card(intended_card, hand)

    def get_call_bs(self, player_index, card, card_amt, hand) -> bool:
        self.ppo_agent.get_call_bs(player_index, card, card_amt, hand)
        return self.bs_agent.get_call_bs(player_index, card, card_amt, hand)

    def give_info(self, player_indexes_picked_up):
        self.ppo_agent.give_info(player_indexes_picked_up)
        self.bs_agent.give_info(player_indexes_picked_up)
    
    def reset(self):
        self.ppo_agent.reset()
        self.bs_agent.reset()
    
    def give_full_info(self, was_bs):
        self.ppo_agent.give_full_info(was_bs)
        self.bs_agent.give_full_info(was_bs)

    def give_winner(self, winner):
        self.ppo_agent.give_winner(winner)
        self.bs_agent.give_winner(winner)
    