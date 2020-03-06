from agents.Q import QAgent


class QRMAgent(QAgent):
    def __init__(self, env, alpha, alpha_min, epsilon, epsilon_min, discount, rm_states, rm_logicals):
        super().__init__(env, alpha, alpha_min, epsilon, epsilon_min, discount)
        self.rm_states = rm_states
        self.rm_logicals = rm_logicals
        self.current_rm_state = self.get_rm_state(self.current_state)

    def get_rm_state(self, state):
        rm_state = -1
        for il, state_logical in enumerate(self.rm_logicals):
            logic = True
            for expressions in state_logical:
                if state[expressions[0]] != expressions[1]:
                    logic = False
                    break
            if logic:
                rm_state = self.rm_states[il]
        return rm_state

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        self.sa_visits[self.current_state][action] += 1
        self.current_state = next_state
        self.current_rm_state = self.get_rm_state(self.current_state)
        return next_state, reward, done


class RewardMachine():
    def __init__(self, state_space, state_transitions, initial_state):
        self.state_space = state_space
        self.state_transitions = state_transitions
        self.current_state = initial_state

    def step(self, state_vars):
        transitions = self.state_transitions[self.current_state]


