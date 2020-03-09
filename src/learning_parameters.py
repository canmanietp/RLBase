class Parameters():
    def __init__(self, alpha, alpha_min, epsilon, epsilon_min, discount, decay, num_episodes, phi=None, phi_min=None, subspaces=None, size_state_vars=None):
        self.ALPHA = alpha
        self.ALPHA_MIN = alpha_min
        self.EPSILON = epsilon
        self.EPSILON_MIN = epsilon_min
        self.DISCOUNT = discount
        self.DECAY_RATE = decay
        self.PHI = phi
        self.PHI_MIN = phi_min
        self.num_episodes = num_episodes
        self.subspaces = subspaces
        self.size_state_vars = size_state_vars
