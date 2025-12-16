import numpy as np
class ReplayBuffer:
    def __init__(self, state_dim, capacity=100_000):
        self.state_dim = state_dim
        self.capacity = capacity

        # Pré-allocation des tableaux
        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity,), dtype=np.int32)
        self.rewards     = np.zeros((capacity,), dtype=np.float32)
        self.dones       = np.zeros((capacity,), dtype=np.float32)

        self.size = 0       # nombre d’éléments réellement remplis
        self.ptr  = 0       # index où on va écrire la prochaine transition

    def store(self, state, action, reward, next_state, done):
        """
        state / next_state : shape (state_dim,)
        action : int
        reward : float
        done   : bool
        """
        self.states[self.ptr]      = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.dones[self.ptr]       = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Renvoie un batch aléatoire:
        (states, actions, rewards, next_states, dones)
        """
        assert self.size >= batch_size, "Pas assez d'échantillons dans le buffer"
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch_states      = self.states[idxs]
        batch_actions     = self.actions[idxs]
        batch_rewards     = self.rewards[idxs]
        batch_next_states = self.next_states[idxs]
        batch_dones       = self.dones[idxs]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones


def select_action(state, q_network, n_actions, epsilon):
    """
    state : np.array de shape (state_dim,)
    q_network : ton modèle Keras
    epsilon : float dans [0,1]
    """
    # Tirage exploration / exploitation
    if np.random.rand() < epsilon:
        # EXPLORATION : action aléatoire
        return np.random.randint(0, n_actions)
    else:
        # EXPLOITATION : on choisit l'action avec Q max
        # Ajouter l'axe batch: (state_dim,) -> (1, state_dim)
        state_input = state[None, :]  # shape (1, state_dim)
        q_values = q_network(state_input).numpy()[0]  # shape (n_actions,)
        action = int(np.argmax(q_values))
        return action

def select_action_qvalues(state, q_model, n_actions=3, epsilon=0.0):
    """
    Pour un modèle qui sort Q(s,a) (Dense(..., activation=None)).
    - epsilon=0.0 => greedy pur
    - epsilon>0   => epsilon-greedy (utile si tu veux tester un peu d’explo)
    """
    if np.random.rand() < epsilon:
        return np.random.randint(0, n_actions)

    state_input = state[None, :].astype(np.float32)  # (1, state_dim)
    q_vals = q_model(state_input).numpy()[0]         # (n_actions,)
    return int(np.argmax(q_vals))
