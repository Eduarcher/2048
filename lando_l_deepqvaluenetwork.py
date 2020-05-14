import random as rd
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from game import Game

game = Game()


class Environment:
    def __init__(self, full_play=False):
        self.game = Game()
        self.state = np.matrix(self.game.board.tiles)
        self.reward = 0
        self.done = False
        self.score = 0
        self.full_play = full_play

    # Returns the initial state
    def initial_state(self):
        self.game = Game()
        self.state = np.matrix(self.game.board.tiles)
        self.reward = 0
        self.done = False
        self.score = 0

    # Return the flatten state
    def get_state(self):
        return self.state.flatten()

    # Simulates a action given a environment object
    def step(self, action):
        if not self.done:
            new_state, updated, self.reward, over = self.game.move(action)
            if updated:
                self.done = over
                self.score += self.reward
                self.state = new_state
            elif self.full_play is False or over is True:
                self.reward = 0
                self.done = True
        else:
            self.reward = 0
            self.done = True


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = None

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
        mem
        if not mem:
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return rd.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values)

    def replay(self, batch_size):
        batch = rd.sample(self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in batch:
            target = reward if (done) else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            q_values = np.reshape(self.model.predict(state)[0], (1, 4))
            q_values[0, action] = target
            self.model.fit(state, q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



# Train the model of a given agent
def train(n_episodes, n_steps, n_replay, agent, late_game_full_play = False):
    print(f"Start training for {n_episodes} episodes with {n_steps} steps...")
    env = Environment()
    for episode in range(n_episodes):
        env.initial_state()
        for _ in range(n_steps):
            prev_state = env.get_state()
            action = agent.act(prev_state)
            env.step(action)
            agent.remember(prev_state, action, env.reward, env.get_state(), env.done)
            agent.replay(n_replay)
            if env.done:
                break
        print(f"episode: {episode + 1}/{n_episodes}, score: {env.score}")
        if episode > (n_episodes * 0.8) and late_game_full_play:
            env.full_play = True
        if episode % 30 == 0 and episode > 0:
            print("Agent Saved")
            agent.model.save("DQN_Model.h5")
    print("Finished training")
    agent.model.save("DQN_Model.h5")


# Test an agent
def test(n_episodes=5000, n_steps=500, n_replay=32, model_path=None, state_size=16, action_size=4, late_game_full_play=False):
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    if model_path:
        print(f"Load model from: {model_path}")
        agent.model = load_model(model_path)
    else:
        agent.build_model()
        train(n_episodes, n_steps, n_replay, agent, late_game_full_play=late_game_full_play)

    print("Start playing...")
    env = Environment(full_play=True)
    while not env.done:
        action = agent.act(env.get_state())
        env.step(action)
    print(f"Finished playing with a score of {env.score}")
    env.game.print_board()


test(5000, 500, late_game_full_play=True)
test(model_path="DQN_Model.h5")
