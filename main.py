from __future__ import division
import gym
import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc
from ReactorModel import Reactor
import matplotlib.pylab as plt

import train
import buffer

# creating environment
env = Reactor()

MAX_EPISODES = 5000				# max amount of times the reactor is being looped over
MAX_STEPS = 200					# max amount of timesteps (dt) in a single run of the reactor
MAX_BUFFER = 1000000			# max amount of (state, action reward, new_state) in the buffer

S_DIM = 3						# state space
A_DIM = 1						# action space

dt = 0.25						# timestep for reactor simulation
PLOT_CLOSE = False				# True if plot is open

ram = buffer.MemoryBuffer(MAX_BUFFER)				# initializing buffer
trainer = train.Trainer(S_DIM, A_DIM, ram)			# initializing neural nets
# trainer.load_models(300)							# used to load past model

for _ep in range(MAX_EPISODES):
	# resetting reactor to initial conditions
	observation = env.reset()

	# used for plotting later on
	Aintrac, Bintrac, Atrac, Btrac, Ctrac = [], [] , [], [], []

	for r in range(MAX_STEPS):

		state = np.float32(observation)[:3]

		if _ep%20 == 0:
			# validate every 5th episode
			actionnc = trainer.get_exploitation_action(state)
		else:
			# get action based on observation, use exploration policy here ( add OU noise )
			actionnc = trainer.get_exploration_action(state)

		# actionnc = trainer.get_exploitation_action(state) 	# uncomment to stop exploration

		# clipping action between 0 and 1
		action = actionnc.clip(0, 1)

		# setting time step of length dt in the reactor
		A, B, C, reward = env.ReactorStep(action, dt)

		# logging data for later plotting
		Aintrac.append(env.Ain)
		Bintrac.append(env.Bin)
		Atrac.append(float(A))
		Btrac.append(float(B))
		Ctrac.append(float(C))

		# assemble new state from env rollouts
		new_state = np.float32([float(A), float(B), float(C)])
		# push this exp in ram
		ram.add(state, action, reward, new_state)

		observation = new_state #[float(A), float(B), float(C)] #np.float32([float(A), float(B), float(C)])

		# perform optimization
		trainer.optimize()

	# plot every x episodes
	if _ep % 20 == 0:

		if PLOT_CLOSE:
			plt.close(fig)
		fig = plt.figure()
		x = np.arange(0, MAX_STEPS * dt, dt)
		ax = fig.add_subplot(1, 1, 1)
		ax.plot(x, Aintrac, x, Bintrac, x, Btrac, x, Atrac, x, Ctrac)
		ax.legend(['Ain', 'Bin', 'Btrac', 'Atrac', 'Ctrac'])
		ax.set_title("reward " + str(round(reward, 4)) + "last C " + str(round(Ctrac[-1],4)))
		fig.show()
		fig.savefig('plots/printindex_' + str(_ep) + '_reward_' + str(round(reward, 4)) + '.png')
		PLOT_CLOSE = True

	print('EPISODE :- ', _ep,'REWARD :- ', reward, 'LAST_C  :- ', Ctrac[-1])

	# check memory consumption and clear memory
	gc.collect()

	# save neural networks every x episodes
	if _ep%100 == 0:
		trainer.save_models(_ep)


print('Completed episodes')
