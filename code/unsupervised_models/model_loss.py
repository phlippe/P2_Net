import torch
import torch.nn as nn
import numpy as np 
import sys
sys.path.append("../")
import os
import math
import time
from vocab import get_SOS_index, get_UNK_index
from model_utils import *

class LossStyleModule(nn.Module):

	def __init__(self, style_size, response_style_size=-1, hidden_dims=None):
		super(LossStyleModule, self).__init__()

		if hidden_dims is None:
			hidden_dims = int(style_size)
		if response_style_size <= 0:
			response_style_size = style_size

		self.style_classifier = nn.Sequential(
				nn.Linear(style_size + response_style_size, hidden_dims),
				nn.ReLU6(),
				nn.BatchNorm1d(hidden_dims),
				nn.Linear(hidden_dims, 1),
				nn.Sigmoid()
			)
		self.loss_module = nn.BCELoss()

		self.to(get_device())

	def forward(self, context_1_style, context_2_style, par_1_style, par_2_style):
		# Taking the sampled styles. Remove tuple if necessary
		context_1_style = context_1_style[0] if isinstance(context_1_style, tuple) else context_1_style
		context_2_style = context_2_style[0] if isinstance(context_2_style, tuple) else context_2_style
		par_1_style = par_1_style[0] if isinstance(par_1_style, tuple) else par_1_style
		par_2_style = par_2_style[0] if isinstance(par_2_style, tuple) else par_2_style
		# Determine batch size for later
		batch_size = context_1_style.size(0)
		# Stack input tensors over features
		input_con_1_par_1 = torch.cat([context_1_style, par_1_style], dim=-1)
		input_con_1_par_2 = torch.cat([context_1_style, par_2_style], dim=-1)
		input_con_2_par_1 = torch.cat([context_2_style, par_1_style], dim=-1)
		input_con_2_par_2 = torch.cat([context_2_style, par_2_style], dim=-1)
		# Stack all possible input combinations over batch size
		overall_input = torch.cat([input_con_1_par_1, input_con_2_par_2, input_con_1_par_2, input_con_2_par_1], dim=0)
		overall_targets = torch.cat([context_1_style.new_ones(size=(batch_size*2,), dtype=torch.float), 
									 context_2_style.new_zeros(size=(batch_size*2,), dtype=torch.float)], dim=0)
		# Run the model
		overall_output = self.style_classifier(overall_input).squeeze(dim=-1)
		# Calculate loss and accuracy
		loss = self.loss_module(overall_output, overall_targets)
		acc = ((overall_output > 0.5).float() == overall_targets).sum().float() / overall_targets.size(0)

		return loss, acc

class LossStyleSimilarityModule(nn.Module):

	def __init__(self, style_size, counter_loss=False):
		super(LossStyleSimilarityModule, self).__init__()
		self.counter_loss = counter_loss
		self.to(get_device())

	def forward(self, context_1_style, context_2_style, par_1_style, par_2_style):
		# Taking the mean (mu) of styles. Remove tuple if necessary
		# We do not take the sampled ones because they would push sigmas towards 0.
		context_1_style = context_1_style[1] if isinstance(context_1_style, tuple) else context_1_style
		context_2_style = context_2_style[1] if isinstance(context_2_style, tuple) else context_2_style
		par_1_style = par_1_style[1] if isinstance(par_1_style, tuple) else par_1_style
		par_2_style = par_2_style[1] if isinstance(par_2_style, tuple) else par_2_style
		
		loss = (1 - F.cosine_similarity(context_1_style, par_1_style, dim=-1)).mean()
		loss += (1 - F.cosine_similarity(context_2_style, par_2_style, dim=-1)).mean()
		if self.counter_loss:
			loss += (F.cosine_similarity(context_1_style, par_2_style, dim=-1)).mean() / 2.0
			loss += (F.cosine_similarity(context_2_style, par_1_style, dim=-1)).mean() / 2.0
		loss = loss / 2
		acc = loss * 0.0
		return loss, acc

class LossStylePrototypeSimilarityModule(nn.Module):

	def __init__(self, style_size, stop_grads=True):
		super(LossStylePrototypeSimilarityModule, self).__init__()
		self.stop_grads = stop_grads
		self.to(get_device())

	def forward(self, context_1_style, context_2_style, par_1_style, par_2_style, proto_dists):
		context_1_style = context_1_style[0] if isinstance(context_1_style, tuple) else context_1_style
		context_2_style = context_2_style[0] if isinstance(context_2_style, tuple) else context_2_style
		par_1_style = par_1_style[0] if isinstance(par_1_style, tuple) else par_1_style
		par_2_style = par_2_style[0] if isinstance(par_2_style, tuple) else par_2_style

		if par_1_style.size(-1) != context_1_style.size(-1):
			return torch.tensor([0.0]).to(get_device()), torch.tensor([0.0]).to(get_device())
		
		if self.stop_grads:
			# Stop gradients for paraphrase styles to prevent finding loopholes that are suboptimal for diversity
			par_1_style = par_1_style.detach()
			par_2_style = par_2_style.detach()

		loss = (1 - F.cosine_similarity(context_1_style, par_1_style, dim=-1)).mean()
		loss += (1 - F.cosine_similarity(context_2_style, par_2_style, dim=-1)).mean()
		loss = loss / 2
		acc = 1 - loss 
		return loss, acc

class LossStylePrototypeDistModule(nn.Module):

	def __init__(self, style_size, stop_grads=True):
		super(LossStylePrototypeDistModule, self).__init__()
		self.stop_grads = stop_grads
		self.to(get_device())

	def forward(self, context_1_style, context_2_style, par_1_style, par_2_style, proto_dists):
		par_1_proto_dist, par_2_proto_dist, context_1_proto_dist, context_2_proto_dist = proto_dists

		if self.stop_grads:
			par_1_proto_dist = par_1_proto_dist.detach()
			par_2_proto_dist = par_2_proto_dist.detach()

		loss = (par_1_proto_dist - context_1_proto_dist).abs().sum(dim=-1).mean()
		loss += (par_2_proto_dist - context_2_proto_dist).abs().sum(dim=-1).mean()
		loss = loss / 2
		acc = 1 - loss 
		return loss, acc



