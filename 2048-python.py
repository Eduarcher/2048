import random as rd
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv
from keras.optimizers import Adam


## Game Logic
# Initializes a new game with size n
def new_game(n):
    return np.matrix([[0] * n] * n)


# Adds a new tile to the matrix
def add_tile(matrix):
    n, m = matrix.shape
    i = rd.randint(0, n - 1)
    j = rd.randint(0, m - 1)

    while matrix[i, j] != 0:
        i = rd.randint(0, n - 1)
        j = rd.randint(0, m - 1)

    matrix[i, j] = 4 if (rd.random() < 0.1) else 2

    return matrix


# Returns whether a game given by a matrix is over or not
def is_over(matrix):
    m, n = matrix.shape

    # Check for empty entries
    for i in range(n):
        for j in range(m):
            if matrix[i, j] == 0:
                return False

    # Check for left/right entries
    for i in range(n):
        for j in range(m - 1):
            if matrix[i, j] == matrix[i, j + 1]:
                return False

    # Check for up/down entries
    for i in range(n - 1):
        for j in range(m):
            if matrix[i, j] == matrix[i + 1, j]:
                return False

    return True


# Move tiles to the left
def cover_up(matrix):
    n, m = matrix.shape
    new = new_game(n)
    updated = False
    for i in range(n):
        count = 0
        for j in range(m):
            if matrix[i, j] != 0:
                new[i, count] = matrix[i, j]
                updated = j != count
                count += 1
    return new, updated


# Merge tiles to the left
def merge(matrix):
    n, m = matrix.shape
    updated = False
    score = 0
    for i in range(n):
        for j in range(m - 1):
            if matrix[i, j] == matrix[i, j + 1] != 0:
                matrix[i, j] *= 2
                matrix[i, j + 1] = 0
                updated = True
                score += matrix[i, j]
    return matrix, updated, score


# Simulates an up movement
def up(matrix):
    matrix = np.rot90(matrix)
    matrix, updated = cover_up(matrix)
    temp = merge(matrix)
    matrix = cover_up(temp[0])[0]
    matrix = np.rot90(matrix, 3)
    updated = updated or temp[1]
    score = temp[2]

    return matrix, updated, score


# Simulates a down movement
def down(matrix):
    matrix = np.rot90(matrix, 3)
    matrix, updated = cover_up(matrix)
    temp = merge(matrix)
    matrix = cover_up(temp[0])[0]
    matrix = np.rot90(matrix)
    updated = updated or temp[1]
    score = temp[2]

    return matrix, updated, score


# Simulates a left movement
def left(matrix):
    matrix, updated = cover_up(matrix)
    temp = merge(matrix)
    matrix = cover_up(temp[0])[0]
    updated = updated or temp[1]
    score = temp[2]

    return matrix, updated, score


# Simulates a right movement
def right(matrix):
    matrix = np.rot90(matrix, 2)
    matrix, updated = cover_up(matrix)
    temp = merge(matrix)
    matrix = cover_up(temp[0])[0]
    matrix = np.rot90(matrix, 2)
    updated = updated or temp[1]
    score = temp[2]

    return matrix, updated, score



## Environment
# Constants
ACTIONS = [up, right, down, left]


class Environment:
    def __init__(self, state, reward, done, score):
        self.state = state
        self.reward = reward
        self.done = done
        self.score = score


# Returns the initial state
def initial_state():
    state = new_game(4)
    state = add_tile(state)
    state = add_tile(state)
    return Environment(state=state, reward=0, done=False, score=0)


# Simulates a action given a environment object
def step(action, env):
    if not env.done:
        state, updated, reward = ACTIONS[action](env.state)
        if updated:
            state = add_tile(state)
            done = is_over(state)
            score = env.score + reward
            return Environment(state, reward, done, score)
        else:
            state = env.state
            reward = 0
            done = True
            score = env.score
            return Environment(state, reward, done, score)
    else:
        state = env.state
        reward = 0
        done = True
        score = env.score
        return Environment(state, reward, done, score)


## Deep Q-Value Network
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = None
        self.batch_size = 64

    # Builds the DQN Model
    def build_model(self):
        # Network Layers
        inputs = Input(shape=(self.state_size,), name='inputs')
        hidden_1 = Dense(64, activation='relu', name='hidden_1')(inputs)
        hidden_2 = Dense(32, activation='relu', name='hidden_2')(hidden_1)
        outputs = Dense(self.action_size, activation='linear', name='outputs')(hidden_2)

        # Model
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['acc'])
        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        mem = (state, action, reward, next_state, done)
        if mem not in self.memory:
            self.memory.append(mem)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return rd.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def replay(self):
        batch = rd.sample(self.memory, min(len(self.memory), self.batch_size))
        for state, action, reward, next_state, done in batch:
            target = reward if (done) else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            q_values = np.reshape(self.model.predict(state)[0], (1, 4))
            q_values[0, action] = target
            self.model.fit(state, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Train the model of a given agent
def train(n_episodes, n_steps, agent):
    print(f"Start training for {n_episodes} episodes with {n_steps} steps...")
    for episode in range(n_episodes):
        env = initial_state()
        for _ in range(n_steps):
            action = agent.act(env.state.flatten())
            next_env = step(action, env)

            agent.remember(env.state.flatten(), action, next_env.reward, next_env.state.flatten(), next_env.done)
            agent.replay()

            env = next_env
            if env.done:
                break

        print(f"episode: {episode + 1}/{n_episodes}, score: {env.score}")

    print("Finished training")
    agent.model.save("DQN_Model.h5")


# Test an agent
def test(n_episodes=5000, n_steps=500, model_path=None):
    agent = DQNAgent(state_size=16, action_size=4)
    if model_path:
        print(f"Load model from: {model_path}")
        agent.model = load_model(model_path)
    else:
        agent.build_model()
        train(n_episodes, n_steps, agent)
    env = initial_state()
    print("Start playing...")
    while not env.done:
        action = agent.act(env.state)
        env = step(action, env)
    print(f"Finished playing with a score of {env.score}")


test(2000, 1000)



