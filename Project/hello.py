# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0 *0.01  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        
        #model.add(MaxPooling2D(pool_size=(2, 2), input_shape=state_shape))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        
        #model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        
        #model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        
        #model.add(MaxPooling2D(pool_size=(2, 2)))

        #model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))

        #model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))

        #model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))

        #model.add(Flatten())
        model.add(Dense(24, activation='relu', input_shape=state_shape))
        #model.add(Dense(256, activation='relu'))
        model.add(Dense(24, activation='relu'))

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            #print target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    		

if __name__ == "__main__":
    env = gym.make('SpaceInvaders-ram-v0')
    state_shape = (env.observation_space.shape)
    action_size = env.action_space.n
    print state_shape
    agent = DQNAgent(state_shape, action_size)
    agent.load("./save/hello-dqn.h5")
    done = False
    batch_size = 32
    
    for e in range(EPISODES):
        state = env.reset()
        #state = np.reshape(state, [1, state_size])
        state = np.reshape(state, (1,) + state.shape)
        score = 0
        time = 0
        lives = env.unwrapped.ale.lives()
        
        while (True):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            reward = reward + 0.1 if not done else -1000
            newlives = env.unwrapped.ale.lives()
            if (newlives < lives):
                reward = - 1000
                lives = newlives
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
            #print reward
            
            #next_state = np.reshape(next_state, [1, state_size])
            next_state = np.reshape(next_state, (1,) + next_state.shape)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            score = score + reward
            time += 1
            #print time
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, score, agent.epsilon))
                break
        '''
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        if e % 10 == 0:
           agent.save("./save/hello-dqn.h5")


'''




