import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import sys
sys.path.append("../")
import os
from random import random
from model_utils import *
from vocab import get_EOS_index, get_SOS_index, get_UNK_index
from data import DATA_GLOVE, DATA_BERT


def createDecoder(model_params, embedding_module):
	if model_params["type"] == ModelTypes.DEC_LSTM:
		return DecoderLSTM(model_params, embedding_module)
	else:
		print("[!] ERROR: Unknown decoder model type %s." % (str(model_params["type"])))
		sys.exit(1)


#######################
## PARAPHRASE MODELS ##
#######################

class DecoderModule(nn.Module):

	def __init__(self, embedding_layer):
		super(DecoderModule, self).__init__()
		self.embedding_layer = embedding_layer

	def forward(self, semantics, styles, UNK_embeds, UNK_lengths, UNK_word_ids=None, labels=None, min_generation_steps=0, max_generation_steps=30, teacher_forcing_ratio=0.0, beams=-1, style_dropout_mask=None, beam_search_method="diverse"):

		batch_size = semantics.size(0)
		# The input for the very first time step is the Start of Sentence token 
		first_token_input = semantics.new_zeros(size=(batch_size,), dtype=torch.long) + get_SOS_index() 
		last_UNK_embeds = semantics.new_zeros(size=(batch_size, self.embedding_layer.embedding_size))
		
		if labels is None and beams == -1: ## Decoding without Teacher Forcing ##
			hidden = None # Hidden has the dimensionality [batch_size, hidden_size]. Can contain more variables (e.g. can be a list or tuple)
			last_output = first_token_input # Output has the dimensionality [batch_size, num_classes]
			all_outputs = list()
			all_preds = list()
			all_UNK_weights = list()
			time_step = 0
			reached_EOS = semantics.new_zeros(size=(batch_size,), dtype=torch.long) # Record if sentence reached end token yet or not
			lengths = semantics.new_zeros(size=(batch_size,), dtype=torch.long) # Lengths of the generated sentences. Make it easier for postprocessing
			UNK_mask = semantics.new_ones(size=(batch_size, UNK_embeds.size(1)), dtype=torch.float32) # Record whether a slot already has been used or not
			UNK_index_range = torch.arange(start=0, end=UNK_mask.size(1), dtype=torch.long, device=UNK_mask.device)
			# Apply style dropout mask directly, but only first step
			styles = styles if style_dropout_mask is None else styles * style_dropout_mask[:,0,:]
			# Iterate over sequence length. We wait until we reach the end of sentence symbol, or the sentence reached a length larger than a certain threshold
			while time_step < min_generation_steps or (time_step < max_generation_steps and reached_EOS.sum().item() < reached_EOS.shape[0]): 
				lengths += (1 - reached_EOS).long() # Add 1 for each batch element which has not reached the end of the sentence yet
				# Perform single decoding step 
				output, last_UNK_embeds, UNK_weights, hidden = self._next_step(semantic_embeds = semantics,
																			   style_embeds = styles,
																			   hidden = hidden, 
																			   last_output = last_output, 
																			   last_UNK_embeds = last_UNK_embeds,
																			   UNK_embeds = UNK_embeds,
																			   UNK_lengths = UNK_lengths,
																			   UNK_start_ids = None,
																			   UNK_mask = (UNK_mask > 0).float(),
																			   time_step = time_step)
				all_outputs.append(output)
				all_UNK_weights.append(UNK_weights)
				# Check if new token is EOS
				_, word_preds = output.max(dim=-1)
				if (UNK_word_ids is None) or (UNK_embeds.size(1) == 0):
					_, UNK_argmax = UNK_weights.max(dim=-1)
					is_UNK = (UNK_argmax != 0).long()
					UNK_argmax = UNK_argmax - 1
					UNK_step_indices = get_UNK_index()
				else:
					_, UNK_argmax = UNK_weights[:,1:].max(dim=-1)
					is_UNK = (UNK_weights[:,0] < 0.5).long()
					UNK_step_indices = torch.gather(UNK_word_ids, dim=1, index=UNK_argmax.unsqueeze(-1)).squeeze(-1)

				word_preds = word_preds * (1 - is_UNK) + UNK_step_indices * is_UNK
				reached_EOS = reached_EOS + (word_preds == get_EOS_index()).long() * (1 - reached_EOS)
				UNK_mask = UNK_mask - (is_UNK.unsqueeze(dim=-1) * (UNK_argmax.unsqueeze(dim=-1) == UNK_index_range).long()).float()
				# As next input is the last output, we update this variable here.
				last_output = word_preds.detach() # Detach as history is not important anymore
				all_preds.append(last_output)
				time_step += 1
			all_outputs = torch.stack(all_outputs, dim=1) # B x seq_len x num_classes
			all_UNK_weights = torch.stack(all_UNK_weights, dim=1) # B x num_UNK+1
			# _, all_preds = torch.max(all_outputs, dim=-1)
			all_preds = torch.stack(all_preds, dim=1) # B x seq_len

		elif labels is None and beams != -1:
			# Results to store after beam search
			all_outputs = list()
			all_lengths = list()
			all_preds = list()
			all_UNK_weights = list()
			# Beam search is executed sentence by sentence
			for batch_index in range(batch_size):
				
				def prep_tensor(t):
					return t[batch_index:batch_index+1].expand(*([beams]+[-1]*(len(t.shape)-1)))

				# Select embeddings for batch, but use batch dimension now for the different beams
				batch_semantics = prep_tensor(semantics)
				batch_styles = prep_tensor(styles)
				batch_UNK_embeds = prep_tensor(UNK_embeds)
				batch_UNK_lengths = prep_tensor(UNK_lengths)
				batch_UNK_word_ids = prep_tensor(UNK_word_ids)
				batch_last_UNK_embeds = prep_tensor(last_UNK_embeds)
				# Variables to store over time steps
				beam_log_probs = [0 for _ in range(beams)] # Log probability for each beam: log p(w_1) + log p(w_2|w_1) + log p(w_3|w_2,w_1) * ...
				hidden = None # The last hidden states for each beam
				output = [semantics.new_zeros(size=(1,), dtype=torch.long) + get_SOS_index() for _ in range(beams)] # The last output words selected for each beam
				all_batch_outputs = [list() for _ in range(beams)] # The overall list of outputs for every beam (full probability distribution for each position)
				all_batch_preds = [list() for _ in range(beams)] # The overall list of word predictions for every beam
				MIN_PROB = torch.FloatTensor([np.finfo(np.float32).min]).expand(beams).to(get_device()) # Minimum log probability a beam can have. Used if beam reached EOS, probability of continuing with any word is minimal
				time_step = 0 # Word position over time
				reached_EOS = np.zeros(shape=(beams,), dtype=np.long) # Record if sentence/beam reached end token yet or not
				lengths = np.zeros(shape=(beams,), dtype=np.long) # Lengths of the generated sentences. Makes it easier for postprocessing
				UNK_mask = [semantics.new_ones(size=(UNK_embeds.size(1),), dtype=torch.float32) for _ in range(beams)] # Record whether a slot already has been used or not
				UNK_index_range = torch.arange(start=0, end=UNK_embeds.size(1), dtype=torch.long, device=UNK_embeds.device)
			
				def sample_from_beam(logits, k=beams):
					if beam_search_method == "standard":
						top_vals, top_ind = logits.topk(k=k, dim=-1) # Determine num_beams beams with highest probability to continue
					elif beam_search_method == "stochastic":
						probs = (logits - logits.max()).exp()
						probs = logits.exp()
						probs = probs / (probs.sum() + 1e-10)
						if probs.sum() < (1-1e-5):
							probs += 1e-5 
							probs = probs / probs.sum()
						# print("-")
						# print("Shape probs: ", probs.shape, "Shape logits: ", logits.shape)
						# print("Probs sum: ", probs.sum(), "k: ", k)
						# print("Logits: ", logits[:10], "Probs: ", probs[:10])
						top_ind = torch.multinomial(probs, num_samples=k, replacement=False)
						# print("Top ind max: ", top_ind.max(), "min: ", top_ind.min())
						# print("-")
						top_vals = logits[top_ind] # logits.gather(top_ind, dim=0)
					elif beam_search_method == "diverse":
						if logits.size(-1) == beams:
							top_vals, top_ind = logits.topk(k=k, dim=-1)
						else:
							gamma = 1
							sort_indices = torch.argsort(logits, dim=-1, descending=True).float()
							div_logits = (logits - sort_indices * gamma).view(logits.shape)
							top_vals, top_ind = div_logits.topk(k=k, dim=-1)
					return top_vals, top_ind

				# Iterate over sequence length. We wait until we reach the end of sentence symbol, or the sentence reached a length larger than a certain threshold
				# But at least, we continue for min_generation_steps (necessary if we train on labels with certain length)
				while time_step < min_generation_steps or (time_step < max_generation_steps and reached_EOS.sum().item() < reached_EOS.shape[0]): 
					lengths += 1 - reached_EOS # Add 1 for each beam which has not reached the end of the sentence yet
					step_beams = [None for _ in range(beams)] # The probability of each sub-beam
					# Generate outputs for every beam
					b_output, batch_last_UNK_embeds, UNK_weights, b_hidden = self._next_step(semantic_embeds = batch_semantics,
																							 style_embeds = batch_styles,
																							 hidden = tuple([torch.stack(h, dim=1) for h in hidden]) if time_step > 0 else None, 
																							 last_output = torch.stack(output, dim=0), 
																							 UNK_embeds = batch_UNK_embeds,
																							 UNK_lengths = batch_UNK_lengths,
																							 last_UNK_embeds = batch_last_UNK_embeds,
																							 UNK_start_ids = None,
																							 UNK_mask = (torch.stack(UNK_mask, dim=0) > 0).float(),
																							 time_step = time_step)
					
					b_prob_sum = b_output.exp().sum(dim=-1).log()[:,None] # Softmax nominator
					b_output = b_output + UNK_weights[:,0].log().unsqueeze(dim=1) - b_prob_sum
					UNK_weights = ((UNK_weights[:,1:] + 1e-10) * UNK_weights[:,0:1]).log()
					b_output = torch.cat([UNK_weights, b_output], dim=-1)

					hidden = [None for _ in range(beams)] # Placeholder for storing new hidden states
					is_UNK = [None for _ in range(beams)] # Placeholder for storing whether the next word is a UNK token or not
					UNK_indices = [None for _ in range(beams)] # Placeholder for storing the possible indices of the UNK tokens
					
					# For each beam, we create num_beams new beams and determine their probability
					# Hence, we have num_beams^2 options to continue. We take those num_beams options
					# with the highest probability
					top_ind_list =[]
					for b in range(beams):
						# Determine the indices and probabilities of the num_beams most likely words to continue the sentence of the beam
						top_vals, top_ind = sample_from_beam(b_output[b], k=beams)
						top_ind = top_ind - UNK_weights.shape[1] # Negative indices for UNKs
						if reached_EOS[b] == 1: # If already stopped, we do not want to take it into account for extending the sentence to new ones
							step_beams[b] = MIN_PROB
						else:
							step_beams[b] = beam_log_probs[b] + top_vals # Shape of (num_beams,)
						# Store those for later
						hidden[b] = tuple([h[:,b] for h in b_hidden])
						is_UNK[b] = (top_ind < 0).float()
						UNK_indices[b] = top_ind + (is_UNK[b] * UNK_weights.shape[1]).long()
						# print("Is UNK: " + str(is_UNK[b]))
						# print("UNK indices: " + str(UNK_indices[b]))
						# print("Batch UNK word ids: " + str(batch_UNK_word_ids))
						# print("Number of UNK words: " + str(UNK_weights.shape[1]))
						# print("Top ind: " + str(top_ind))
						output[b] = torch.where(is_UNK[b] == 0, top_ind, torch.gather(batch_UNK_word_ids, dim=1, index=torch.min(UNK_indices[b], UNK_indices[b]*0.0+UNK_weights.shape[1]-1).unsqueeze(-1)).squeeze(-1)) # Top num_beams words for this beam
						top_ind_list.append(top_ind)
						# print("Output: " + str(output[b]))

					if time_step == 0: # In the first time step, all beams started from the same position/word. Hence, we choose the first beam only and take its num_beams highest probs
						step_beams = step_beams[0]

						top_vals, _ = step_beams.topk(k=beams, dim=-1) # sample_from_beam(step_beams, k=beams)

						top_ind_beams = np.zeros(shape=(beams,), dtype=np.int32)
						top_ind_words = np.arange(beams, dtype=np.int32)
					else:
						step_beams = torch.cat(step_beams, dim=-1) # Shape of (num_beams^2,)
						
						top_vals, top_ind = step_beams.topk(k=beams, dim=-1) # sample_from_beam(step_beams, k=beams)

						top_ind_beams = (top_ind / beams).cpu().numpy() # Which previous beams to continue
						top_ind_words = torch.fmod(top_ind, beams).cpu().numpy() # Which word to choose

					# print("Top ind beams: " + str(top_ind_beams))
					# print("Top ind words: " + str(top_ind_words))

					# Update all variables to new beams
					new_outputs = list()
					new_hidden_h = list()
					new_hidden_c = list()
					new_probs = list()
					new_batch_outputs = list()
					new_batch_preds = list()
					new_UNK_mask = list()

					fb = 0
					for b in range(beams):
						if reached_EOS[b] == 1: # If a beam has already ended, we do not change it
							fb += 1
							new_outputs.append(output[b][0].detach())
							new_hidden_h.append(hidden[b][0].detach())
							new_hidden_c.append(hidden[b][1].detach())
							new_probs.append(beam_log_probs[b].detach())
							new_batch_outputs.append(all_batch_outputs[b] + [b_output[b].detach()])
							new_batch_preds.append(all_batch_preds[b] + [output[b][0].detach()])
							new_UNK_mask.append(UNK_mask[b].detach())
						else:
							beam_index = b - fb # top n beam. If a beam before it already stopped, we need to take that into account here
							new_outputs.append(output[top_ind_beams[beam_index]][top_ind_words[beam_index]].detach())
							new_hidden_h.append(hidden[top_ind_beams[beam_index]][0].detach())
							new_hidden_c.append(hidden[top_ind_beams[beam_index]][1].detach())
							new_probs.append(top_vals[beam_index].detach())
							new_batch_outputs.append(all_batch_outputs[top_ind_beams[beam_index]] + [b_output[top_ind_beams[beam_index]].detach()])
							new_batch_preds.append(all_batch_preds[top_ind_beams[beam_index]] + [output[top_ind_beams[beam_index]][top_ind_words[beam_index]].detach()])
							new_UNK_mask.append((UNK_mask[top_ind_beams[beam_index]] - (is_UNK[top_ind_beams[beam_index]][top_ind_words[beam_index]].unsqueeze(dim=-1) * ((UNK_weights.shape[1] + top_ind_list[top_ind_beams[beam_index]][top_ind_words[beam_index]].unsqueeze(dim=-1)) == UNK_index_range).float())).detach())
							reached_EOS[b] = (output[top_ind_beams[beam_index]][top_ind_words[beam_index]] == get_EOS_index()).detach()
							# if new_outputs[-1] < 0:
							# 	print("New outputs", new_outputs[-1])
							# 	print("Is UNK", is_UNK[top_ind_beams[beam_index]][top_ind_words[beam_index]])
							# 	print("UNK mask", UNK_mask[top_ind_beams[beam_index]])
							# 	print("Top ind list", top_ind_list[top_ind_beams[beam_index]][top_ind_words[beam_index]])
							# 	print("UNK index range", UNK_index_range)

					output = new_outputs
					hidden = (new_hidden_h, new_hidden_c)
					beam_log_probs = new_probs
					all_batch_outputs = new_batch_outputs
					all_batch_preds = new_batch_preds
					UNK_mask = new_UNK_mask

					time_step += 1

				# Stack/concatenate lists to convert list to tensor
				# all_outputs.append(torch.stack([torch.stack(b) for b in all_batch_outputs], dim=0).detach()) # Commented due to memory 
				all_preds.append(torch.stack([torch.stack(b) for b in all_batch_preds], dim=0).detach())
				all_lengths.append(torch.LongTensor(lengths).to(get_device()).detach())
				all_UNK_weights.append(torch.stack([torch.stack([a[:batch_UNK_lengths[0].item()].detach() for a in b], dim=0) for b in all_batch_outputs], dim=0))
			
			max_output_len = max([o_tensor.shape[1] for o_tensor in all_preds])
			for i in range(len(all_preds)):
				o_shape = all_preds[i].shape
				if o_shape[1] < max_output_len:
					# all_outputs[i] = torch.cat([all_outputs[i], all_outputs[i].new_zeros(size=(o_shape[0], max_output_len - o_shape[1], o_shape[2]))], dim=1)
					all_preds[i] = torch.cat([all_preds[i], all_preds[i].new_zeros(size=(o_shape[0], max_output_len - o_shape[1]))], dim=1)

			# all_outputs = torch.stack(all_outputs, dim=0).detach()
			all_outputs = None
			all_preds = torch.stack(all_preds, dim=0).detach()
			lengths = torch.stack(all_lengths, dim=0).detach()

		elif teacher_forcing_ratio == 1.0: ## Teacher Forcing ##
			UNK_start_ids = UNK_lengths.new_zeros(size=UNK_lengths.shape)
			hidden = None # Hidden has the dimensionality [batch_size, hidden_size]. Can contain more variables (e.g. can be a list or tuple)
			all_outputs = list()
			all_UNK_weights = list()
			lengths = semantics.new_zeros(size=(batch_size,), dtype=torch.long) # Lengths of the generated sentences. Make it easier for postprocessing
			for i in range(labels.size(1)): # Iterate over sequence length
				# Perform single decoding step 
				output, last_UNK_embeds, UNK_weights, hidden = self._next_step(semantic_embeds = semantics,
																			   style_embeds = styles if style_dropout_mask is None else styles * style_dropout_mask[:,i,:],
																			   hidden = hidden, 
																			   last_output = labels[:,i-1] if i > 0 else first_token_input, 
																			   last_UNK_embeds = last_UNK_embeds,
																			   UNK_embeds = UNK_embeds,
																			   UNK_lengths = UNK_lengths,
																			   UNK_start_ids = UNK_start_ids,
																			   time_step = i)
				all_outputs.append(output)
				all_UNK_weights.append(UNK_weights)
				lengths += (labels[:,i] != -1).long()
				is_UNK = (labels[:,i] < -1).long()
				UNK_start_ids += is_UNK
			all_outputs = torch.stack(all_outputs, dim=1) # B x seq_len x num_classes
			all_UNK_weights = torch.stack(all_UNK_weights, dim=1) # B x seq_len x num_UNK+1
			_, all_preds = torch.max(all_outputs, dim=-1)
			

			if (UNK_word_ids is None) or (UNK_embeds.size(1) == 0):
				_, UNK_argmax = all_UNK_weights.max(dim=-1)
				is_UNK = (UNK_argmax != 0).long()
				UNK_step_indices = get_UNK_index()
			else:
				_, UNK_argmax = all_UNK_weights[:,:,1:].max(dim=-1)
				is_UNK = (all_UNK_weights[:,:,0] < 0.5).long()
				UNK_step_indices = torch.gather(UNK_word_ids, dim=1, index=UNK_argmax)
			all_preds = all_preds * (1 - is_UNK) + UNK_step_indices * is_UNK
			all_preds = all_preds.detach()

		else: ## Decoding with partial Teacher Forcing ## (deciding at every time step whether labels should be used as input or not)
			UNK_start_ids = UNK_lengths.new_zeros(size=UNK_lengths.shape)
			hidden = None # Hidden has the dimensionality [batch_size, hidden_size]. Can contain more variables (e.g. can be a list or tuple)
			last_output = first_token_input # Output has the dimensionality [batch_size, num_classes]
			all_outputs = list()
			all_preds = list()
			all_UNK_weights = list()
			time_step = 0
			lengths = semantics.new_zeros(size=(batch_size,), dtype=torch.long) # Lengths of the generated sentences. Make it easier for postprocessing
			# Iterate over sequence length. We wait until we reach the end of sentence symbol, or the sentence reached a length larger than a certain threshold
			for time_step in range(labels.size(1)): # Iterate over sequence length
				# Decide whether to take next action on 
				if time_step > 0:
					use_tf = (last_output.new_zeros(size=last_output.shape, dtype=torch.float32).uniform_() < teacher_forcing_ratio).long()
					last_output = last_output * (1 - use_tf) + labels[:,time_step-1] * use_tf
				# Perform single decoding step 
				output, last_UNK_embeds, UNK_weights, hidden = self._next_step(semantic_embeds = semantics,
																			   style_embeds = styles if style_dropout_mask is None else styles * style_dropout_mask[:,time_step,:],
																			   hidden = hidden, 
																			   last_output = last_output, 
																			   last_UNK_embeds = last_UNK_embeds,
																			   UNK_embeds = UNK_embeds,
																			   UNK_lengths = UNK_lengths,
																			   UNK_start_ids = UNK_start_ids,
																			   time_step = time_step)
				all_outputs.append(output)
				all_UNK_weights.append(UNK_weights)
				# Check if new token is EOS
				_, word_preds = output.max(dim=-1)
				if (UNK_word_ids is None) or (UNK_embeds.size(1) == 0):
					_, UNK_argmax = UNK_weights.max(dim=-1)
					is_UNK = (UNK_argmax != 0).long()
					UNK_step_indices = get_UNK_index()
				else:
					_, UNK_argmax = UNK_weights[:,1:].max(dim=-1)
					is_UNK = (UNK_weights[:,0] < 0.5).long()
					UNK_step_indices = torch.gather(UNK_word_ids, dim=1, index=UNK_argmax.unsqueeze(-1)).squeeze(-1)

				word_preds = word_preds * (1 - is_UNK) + UNK_step_indices * is_UNK
				# As next input might be the last output, we update this variable here.
				last_output = word_preds.detach() # Detach as history is not important anymore
				all_preds.append(last_output)
				lengths += (labels[:,time_step] != -1).long() # Add 1 for each batch element which has not reached the end of the sentence yet
				is_UNK = (labels[:,time_step] < -1).long()
				UNK_start_ids += is_UNK

			all_outputs = torch.stack(all_outputs, dim=1) # B x seq_len x num_classes
			all_UNK_weights = torch.stack(all_UNK_weights, dim=1) # B x num_UNK+1
			# _, all_preds = torch.max(all_outputs, dim=-1)
			all_preds = torch.stack(all_preds, dim=1) # B x seq_len

		return all_outputs, all_UNK_weights, all_preds, lengths

	def _next_step(self, semantics, styles, hidden, last_output, time_step):
		raise NotImplementedError

	def get_loss_module(self, weight=None, ignore_index=-1, reduction='mean'):
		loss_module = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index).to(get_device())
		return loss_module



class DecoderLSTMModule(nn.Module):

	def __init__(self, model_params, embedding_layer):
		super(DecoderLSTMModule, self).__init__()
		self.embedding_layer = embedding_layer
		semantics_size = get_param_val(model_params, "semantic_size", allow_default=False, error_location="DecoderLSTMModule - model_params")
		style_size = get_param_val(model_params, "style_size", allow_default=False, error_location="DecoderLSTMModule - model_params")
		hidden_size = get_param_val(model_params, "hidden_size", allow_default=False, error_location="DecoderLSTMModule - model_params")
		num_layers = get_param_val(model_params, "num_layers", 1)
		input_dropout = get_param_val(model_params, "input_dropout", 0.0)
		lstm_dropout = get_param_val(model_params, "lstm_dropout", 0.0)
		output_dropout = get_param_val(model_params, "output_dropout", 0.0)
		input_size = self.embedding_layer.embedding_size
		self.lstm_additional_input = get_param_val(model_params, "lstm_additional_input", False)
		if self.lstm_additional_input:
			self.lstm_additional_input_layer = nn.Sequential(
					nn.Dropout(input_dropout),
					nn.Linear(semantics_size + style_size + self.embedding_layer.embedding_size, input_size),
					nn.Dropout(input_dropout),
					nn.Tanh()
				)
			input_size = input_size * 2
		## Setting up all layers ##
		# Layer for creating initial states based on dialog and template
		self.embed_to_state_layer = nn.Sequential(
				nn.Linear(semantics_size + style_size, hidden_size * num_layers * 2),
				nn.Dropout(input_dropout),
				nn.Tanh()
			)
		# LSTM stack
		self.lstm_stack = nn.LSTM(input_size = input_size,
								  hidden_size = hidden_size,
								  num_layers = num_layers,
								  dropout = lstm_dropout,
								  bidirectional = False,
								  batch_first = True)
		## Save certain parameters ##
		self.state_size = [-1, num_layers, 2, hidden_size]

	def forward(self, semantic_embeds, style_embeds, hidden, last_output, time_step, last_UNK_embeds=None):
		if time_step == 0:
			if hidden is not None:
				print("[!] WARNING: At time step 0, the given hidden state was not empty. It will be overwritten by embedding layer...")
			batch_size = semantic_embeds.shape[0]
			state_embeds = self.embed_to_state_layer(torch.cat([semantic_embeds, style_embeds], dim=-1))
			state_embeds = state_embeds.view(*self.state_size)
			state_embeds = state_embeds.transpose(1,0)
			hidden = (state_embeds[:,:,0], state_embeds[:,:,1])
		if len(last_output.shape) == 1:
			last_output = last_output[:,None] # Adding extra dimensionality for sequence length being 1
		if len(last_output.shape) == 2:
			last_output = self.embedding_layer(last_output, use_pos_encods=False)

		hidden = (hidden[0].contiguous(), hidden[1].contiguous())
		# Run single LSTM step
		if self.lstm_additional_input:
			if last_UNK_embeds is None:
				print("[!] ERROR - DecoderLSTMModule: Last unknown embeds are None although additional inputs are selected")
				sys.exit(1)
			additional_input = self.lstm_additional_input_layer(torch.cat([last_UNK_embeds, semantic_embeds, style_embeds], dim=-1))
			last_output = torch.cat([last_output, additional_input.unsqueeze(dim=1)], dim=-1)
		output, hidden = self.lstm_stack(last_output, hidden)
		output = output.squeeze(dim=1) # Remove single time step
		return output, hidden


class DecoderOutputModule(nn.Module):

	def __init__(self, model_params):
		super(DecoderOutputModule, self).__init__()
		semantics_size = get_param_val(model_params, "semantic_size", allow_default=False, error_location="DecoderOutputModule - model_params")
		style_size = get_param_val(model_params, "style_size", allow_default=False, error_location="DecoderOutputModule - model_params")
		num_classes = get_param_val(model_params, "num_classes", allow_default=False, error_location="DecoderOutputModule - model_params")
		hidden_size = get_param_val(model_params, "hidden_size", allow_default=False, error_location="DecoderOutputModule - model_params")
		output_dropout = get_param_val(model_params, "output_dropout", 0.0)
		## Setting up all layers ##
		# Compressing semantics, style and hidden encoding of LSTM into final embedding
		self.output_feature_layer = nn.Sequential(
				nn.Linear(hidden_size + semantics_size + style_size, hidden_size)
			)
		# Attention layer over masked unknown words and prediction distribution
		self.UNK_attention_layer = WordLevelAttention(input_size_words = hidden_size,
													  input_size_context = hidden_size,
													  hidden_size = hidden_size)
		# Layer for generating final output
		self.prediction_layer = nn.Sequential(
				nn.Dropout(output_dropout),
				nn.Linear(hidden_size, num_classes)
			)

	def forward(self, lstm_output, semantic_embeds, style_embeds, UNK_embeds, UNK_lengths):
		
		output_features = torch.cat([lstm_output, semantic_embeds, style_embeds], dim=-1) # B x feature_dim
		output_features = self.output_feature_layer(output_features) # B x hidden
		output_gen = self.prediction_layer(output_features) # B x vocab

		if UNK_embeds.size(-1) != output_features.size(-1):
			print("[!] ERROR: embeddings of the unknown tokens are unequal to the output feature size")
			sys.exit(1)

		attn_words = torch.cat([output_features.unsqueeze(1), UNK_embeds], dim=1) # B x UNK+1 x hidden
		comb_UNKs, attn_weights = self.UNK_attention_layer(attn_words, UNK_lengths+1, lstm_output) # B x UNK+1

		return output_gen, comb_UNKs, attn_weights


class DecoderPointerOutputModule(nn.Module):

	def __init__(self, model_params, UNK_dims):
		super(DecoderPointerOutputModule, self).__init__()
		semantics_size = get_param_val(model_params, "semantic_size", allow_default=False, error_location="DecoderPointerOutputModule - model_params")
		style_size = get_param_val(model_params, "style_size", allow_default=False, error_location="DecoderPointerOutputModule - model_params")
		num_classes = get_param_val(model_params, "num_classes", allow_default=False, error_location="DecoderPointerOutputModule - model_params")
		hidden_size = get_param_val(model_params, "hidden_size", allow_default=False, error_location="DecoderPointerOutputModule - model_params")
		output_dropout = get_param_val(model_params, "output_dropout", 0.0)
		self.concat_features = get_param_val(model_params, "concat_features", False)
		## Setting up all layers ##
		# Compressing semantics, style and hidden encoding of LSTM into final embedding
		if not self.concat_features:
			self.output_feature_layer = nn.Sequential(
					nn.Linear(hidden_size + semantics_size + style_size, hidden_size),
					nn.ReLU()
				)
			hidden_intermediate_size = hidden_size
		else:
			hidden_intermediate_size = hidden_size + semantics_size + style_size
		# Attention layer over masked unknown words and prediction distribution
		self.UNK_attention_layer = WordLevelAttention(input_size_words = UNK_dims,
													  input_size_context = hidden_intermediate_size,
													  hidden_size = UNK_dims)
		# Layer for generating final output
		self.prediction_layer = nn.Sequential(
				nn.Dropout(output_dropout),
				nn.Linear(hidden_intermediate_size + UNK_dims, num_classes)
			)
		# Layer for generating final output
		self.p_gen_classifier = nn.Sequential(
				nn.Dropout(output_dropout),
				nn.Linear(hidden_intermediate_size + UNK_dims, 1),
				nn.Sigmoid()
			)

	def forward(self, lstm_output, semantic_embeds, style_embeds, UNK_embeds, UNK_lengths, UNK_start_ids=None, UNK_mask=None):
		
		output_features = torch.cat([lstm_output, semantic_embeds, style_embeds], dim=-1) # B x feature_dim
		if not self.concat_features:
			output_features = self.output_feature_layer(output_features) # B x hidden

		comb_UNKs, attn_weights = self.UNK_attention_layer(UNK_embeds, UNK_lengths, output_features, encoder_start_ids=UNK_start_ids, encoder_mask=UNK_mask) # B x UNK
		comb_features = torch.cat([comb_UNKs, output_features], dim=-1)

		output_gen = self.prediction_layer(comb_features) # B x vocab
		p_gen = self.p_gen_classifier(comb_features)
		p_gen = torch.max(((UNK_lengths.unsqueeze(-1) == 0) | (attn_weights.sum(-1).unsqueeze(-1) == 0)).float(), p_gen) # If no slot is in a sentence, we don't want to train the p_gen classifier
		attn_weights = torch.cat([p_gen, attn_weights * (1 - p_gen)], dim=-1)

		return output_gen, comb_UNKs, attn_weights


class DecoderLSTM(DecoderModule):

	def __init__(self, model_params, embedding_layer):
		super(DecoderLSTM, self).__init__(embedding_layer)
		## Reading all parameters ##
		semantics_size = get_param_val(model_params, "semantic_size", allow_default=False, error_location="DecoderLSTMModule - model_params")
		self.style_size = get_param_val(model_params, "style_size", allow_default=False, error_location="DecoderLSTMModule - model_params")
		num_classes = get_param_val(model_params, "num_classes", allow_default=False, error_location="ParaphraseLSTM - model_params")
		hidden_size = get_param_val(model_params, "hidden_size", allow_default=False, error_location="ParaphraseLSTM - model_params")
		output_dropout = get_param_val(model_params, "output_dropout", 0.0)
		## Setting up all layers ##
		# LSTM decoder for time information
		self.lstm_decoder = DecoderLSTMModule(model_params, self.embedding_layer)
		# Layer for generating final output
		# self.output_module = DecoderOutputModule(model_params) # FIXME
		self.output_module = DecoderPointerOutputModule(model_params, UNK_dims=300)

	def _next_step(self, semantic_embeds, style_embeds, hidden, last_output, UNK_embeds, UNK_lengths, time_step, UNK_start_ids=None, UNK_mask=None, last_UNK_embeds=None):
		# Run single LSTM step
		lstm_output, hidden = self.lstm_decoder(semantic_embeds, style_embeds, hidden, last_output, time_step, last_UNK_embeds=last_UNK_embeds)
		# Generate output prediction over words
		output_gen, comb_UNKs, attn_weights = self.output_module(lstm_output, semantic_embeds, style_embeds, UNK_embeds, UNK_lengths, UNK_start_ids=UNK_start_ids, UNK_mask=UNK_mask)
		return output_gen, comb_UNKs, attn_weights, hidden



