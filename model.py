import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class Critic(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		Critic network: State and action parameters are fed seperately into different layers of the network
		"""
		super(Critic, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim,256)
		self.fcs2 = nn.Linear(256,128)

		self.fca1 = nn.Linear(action_dim,128)

		self.fc2 = nn.Linear(256,128)
		self.fc3 = nn.Linear(128,1)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		"""
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x = torch.cat((s2,a1),dim=1)

		x = F.relu(self.fc2(x))
		x = self.fc3(x)

		return x


class Actor(nn.Module):

	def __init__(self, state_dim, action_dim):
		"""
		Actor network: Simple multilayer linear neural network
		"""
		super(Actor, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fc1 = nn.Linear(state_dim,256)
		self.fc2 = nn.Linear(256,128)
		self.fc3 = nn.Linear(128,64)
		self.fc4 = nn.Linear(64,action_dim)

	def forward(self, state):
		"""
		Returns a single value representing the action, based on the current state as the input
		"""
		x = torch.sigmoid(self.fc1(state))
		x = torch.sigmoid(self.fc2(x))
		x = torch.sigmoid(self.fc3(x))
		action = torch.sigmoid(self.fc4(x))

		return action



