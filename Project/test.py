import gym
import numpy as np

env = gym.make('FlappyBird-v0')

print (env.observation_space.shape)
print env.action_space.n

for i_episode in range(10):
	observation = env.reset()
	for t in range(10000):
		env.render()
		#print(observation)
		action = env.action_space.sample()
		#action = 2
		
		observation, reward, done, info = env.step(action)
		print reward, done, t, action
		
		if done:
		    print("Episode finished after {} timesteps".format(t+1))
		    break
		
