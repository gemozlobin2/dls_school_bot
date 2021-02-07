from enum import Enum


class StateEnum(Enum):
    BASE_STATE = 1

    STYLE_TRANSFER_STAGE_0 = 2
    STYLE_TRANSFER_STAGE_1 = 3

    CYCLE_GAN = 4


class UserState:
    def __init__(self, state):
        self.state = state
        self.img = None


class StateManager:
    def __init__(self):
        self.store = {}

    def set_state(self, user_id, state):
        if state == StateEnum.BASE_STATE:
            self.store.pop(user_id, None)
        elif user_id in self.store:
            self.store[user_id].state = state
        else:
            self.store[user_id] = UserState(state)

    def set_attr(self, user_id, name, value):
        if user_id not in self.store or not hasattr(self.store[user_id], name):
            return False
        setattr(self.store[user_id], name, value)
        return True

    def get_attr(self, user_id, name):
        if user_id not in self.store or not hasattr(self.store[user_id], name):
            return None
        return getattr(self.store[user_id], name)

    def get_attrs(self, user_id):
        if user_id not in self.store:
            return None
        return self.store[user_id].__dict__

    def get_state(self, user_id):
        if user_id in self.store:
            return self.store[user_id].state
        return StateEnum.BASE_STATE
