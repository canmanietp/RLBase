class Parameters:
    def __init__(self, num_episodes, discount, epsilon, epsilon_min, decay):
        self.num_episodes = num_episodes
        self.DISCOUNT = discount
        self.EPSILON = epsilon
        self.EPSILON_MIN = epsilon_min
        self.DECAY_RATE = decay


class DiscreteParameters(Parameters):
    def __init__(self, alpha, alpha_min, epsilon, epsilon_min, discount, decay, num_episodes, phi=None, phi_min=None, size_state_vars=None, sub_spaces=None):
        super().__init__(num_episodes, discount, epsilon, epsilon_min, decay)
        self.ALPHA = alpha
        self.ALPHA_MIN = alpha_min
        self.PHI = phi
        self.PHI_MIN = phi_min
        self.size_state_vars = size_state_vars
        self.sub_spaces = sub_spaces


class ContinuousParameters(Parameters):
    def __init__(self, init_model, memory_size, batch_size, learning_rate, discount, epsilon, epsilon_min, decay, observation_space, retrain_steps, num_episodes, meta_model=None, sub_models=None, phi=None, phi_min=None, sub_spaces=None):
        super().__init__(num_episodes, discount, epsilon, epsilon_min, decay)
        self.INIT_MODEL = init_model
        self.MEMORY_SIZE = memory_size
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.META_MODEL = meta_model
        self.sub_models = sub_models
        self.PHI = phi
        self.PHI_MIN = phi_min
        self.observation_space = observation_space
        self.retrain_steps = retrain_steps
        self.sub_spaces = sub_spaces


