class Agent:
    def __init__(self, my_index, num_players, agent_args = []):
        raise NotImplementedError

    def get_card(self, intended_card, hand) -> tuple[str, int]:
        raise NotImplementedError

    def get_call_bs(self, player_index, card, card_amt, hand) -> bool:
        raise NotImplementedError

    def give_info(self, player_indexes_picked_up):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def give_full_info(self, was_bs):
        raise NotImplementedError
