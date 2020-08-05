import torch
import torch.nn as nn
import numpy as np 
import sys
sys.path.append("../")
import os
from model_utils import *
from data import DatasetHandler, DATA_GLOVE, DATA_BERT


def createEncoder(model_params, hierarchical_model=False):
	if model_params["type"] == ModelTypes.LSTM:
		return EncoderLSTM(model_params) if not hierarchical_model else EncoderHierarchicalLSTM(model_params)
	elif model_params["type"] == ModelTypes.BILSTM:
		return EncoderBiLSTM(model_params) if not hierarchical_model else EncoderHierarchicalBiLSTM(model_params)
	else:
		print("[!] ERROR: Unknown encoder type %s." % (str(model_params["type"])))
		sys.exit(1)


def sample_VAE_Gaussian(vals_mu, vals_sigma):
	samples = torch.randn_like(vals_mu) * vals_sigma + vals_mu
	return samples

####################
## ENCODER MODELS ##
####################

class EncoderModule(nn.Module):

	def __init__(self, model_params, data_embed=DATA_GLOVE):
		super(EncoderModule, self).__init__()
		self.semantic_size = get_param_val(model_params, "semantic_size", allow_default=False, error_location="EncoderBiLSTM - model_params")
		self.style_size = get_param_val(model_params, "style_size", allow_default=False, error_location="EncoderBiLSTM - model_params")
		self.data_embed_type = data_embed

	def prepare(self, embed_word_ids, lengths, embedding_module, slot_mask=None, slot_vals=None, slot_lengths=None):
		if isinstance(embed_word_ids, dict):
			embed_word_ids = embed_word_ids[self.data_embed_type]
		if isinstance(lengths, dict):
			lengths = lengths[self.data_embed_type]
		if self.data_embed_type == DATA_GLOVE:
			embed_words = embedding_module(embed_word_ids, use_pos_encods=True, slot_mask=slot_mask, slot_vals=slot_vals, slot_lengths=slot_lengths)
		else:
			embed_words = None
		return embed_word_ids, embed_words, lengths

	def forward(self, embed_word_ids, embed_words, lengths, masks, attn_context_vector=None, additional_input=None, debug=False, debug_proto=False):
		zero_length = (lengths == 0).long()
		encoding = self.encode(embed_word_ids, embed_words, lengths + zero_length, masks, attn_context_vector=attn_context_vector, additional_input=additional_input, debug=debug, debug_proto=debug_proto)
		return encoding

	def encode(self, embed_word_ids, embed_words, lengths, masks):
		raise NotImplementedError


class EncoderRNN(EncoderModule):

	def __init__(self, model_params):
		super(EncoderRNN, self).__init__(model_params, data_embed=DATA_GLOVE)
		hidden_size = get_param_val(model_params, "hidden_size", allow_default=False, error_location="EncoderBiLSTM - model_params")
		dropout = get_param_val(model_params, "dropout", 0.0)
		self.num_layers = get_param_val(model_params, "num_layers", 1)
		self.share_attention_layer = get_param_val(model_params, "share_attention", True)

		self.rnn = self._create_RNN(model_params)

		if self.share_attention_layer:
			self.attention_layer = WordLevelAttention(input_size_words = hidden_size,
													  input_size_context = hidden_size,
													  hidden_size = hidden_size)
		else:
			self.attention_layer_semantics = WordLevelAttention(input_size_words = hidden_size,
																input_size_context = hidden_size,
																hidden_size = hidden_size)
			self.attention_layer_style = WordLevelAttention(input_size_words = hidden_size,
															 input_size_context = hidden_size,
															 hidden_size = hidden_size)
		self.semantic_layer = nn.Sequential(
				nn.Dropout(dropout),
				nn.Linear(hidden_size, self.semantic_size)
			)
		self.style_layer = nn.Sequential(
				nn.Dropout(dropout),
				nn.Linear(hidden_size, self.style_size * 2)
			)

	def encode(self, embed_word_ids, embed_words, lengths, masks, attn_context_vector=None, additional_input=None, debug=False, debug_proto=False):
		final_states, word_outputs = self.rnn(embed_words, lengths)
		if self.num_layers > 1:
			final_states = final_states.view(final_states.shape[0], self.num_layers, -1)[:,-1,:]
		if self.share_attention_layer:
			final_embed, _ = self.attention_layer(encoder_word_embeds = word_outputs,
												  encoder_lengths = lengths,
												  context_state = final_states)
			final_embed_semantics, final_embed_style = final_embed, final_embed
		else:
			final_embed_semantics, _ = self.attention_layer_semantics(encoder_word_embeds = word_outputs,
																	  encoder_lengths = lengths,
																	  context_state = final_states)
			final_embed_style, _ = self.attention_layer_style(encoder_word_embeds = word_outputs,
															  encoder_lengths = lengths,
															  context_state = final_states)
		semantic_embeds = self.semantic_layer(final_embed_semantics)
		style_raw_vals = self.style_layer(final_embed_style)
		style_mu, style_sigma = style_raw_vals[:,:self.style_size], style_raw_vals[:,self.style_size:]
		style_sigma = torch.exp(style_sigma)
		if self.training:
			style_embeds = sample_VAE_Gaussian(style_mu, style_sigma)
		else:
			style_embeds = style_mu

		UNK_tokens, UNK_lengths = EncoderRNN._gather_UNK_token_embeddings(word_outputs, masks)

		return semantic_embeds, (style_embeds, style_mu, style_sigma), UNK_tokens, UNK_lengths

	def _create_RNN(self, model_params):
		raise NotImplementedError

	@staticmethod
	def _gather_UNK_token_embeddings(word_embeds, mask, word_ids=None):
		if len(mask.shape) == 2:
			mask = mask[:,:,None]
		max_num_tokens = torch.max(mask)
		UNK_tokens = word_embeds.new_zeros(size=(word_embeds.shape[0], max(1, max_num_tokens), word_embeds.shape[2]))
		UNK_word_ids = word_embeds.new_zeros(size=(word_embeds.shape[0], max(1, max_num_tokens)), dtype=torch.long)
		UNK_lengths = word_embeds.new_zeros(size=(word_embeds.shape[0],), dtype=torch.long)
		for i in range(max_num_tokens):
			mask_i = (mask == i+1).float()
			UNK_tokens[:,i,:] = (word_embeds * mask_i).sum(dim=1)
			squeezed_mask = mask_i.squeeze(dim=-1).long()
			UNK_lengths += squeezed_mask.sum(dim=-1)
			if word_ids is not None:
				UNK_word_ids[:,i] = (word_ids * squeezed_mask).sum(dim=1)
		if word_ids is None:
			return UNK_tokens, UNK_lengths
		else:
			return UNK_tokens, UNK_lengths, UNK_word_ids


class EncoderLSTM(EncoderRNN):

	def __init__(self, model_params):
		super(EncoderLSTM, self).__init__(model_params)

	def _create_RNN(self, model_params):
		return PyTorchLSTMChain(input_size = get_param_val(model_params, "input_size", allow_default=False, error_location="EncoderLSTM - model_params"),
								hidden_size = get_param_val(model_params, "hidden_size", allow_default=False, error_location="EncoderLSTM - model_params"),
								num_layers = get_param_val(model_params, "num_layers", 1),
								dropout = get_param_val(model_params, "dropout", 0.0),
								bidirectional = False)


class EncoderBiLSTM(EncoderRNN):

	def __init__(self, model_params):
		super(EncoderBiLSTM, self).__init__(model_params)

	def _create_RNN(self, model_params):
		return PyTorchLSTMChain(input_size = get_param_val(model_params, "input_size", allow_default=False, error_location="EncoderLSTM - model_params"),
								hidden_size = get_param_val(model_params, "hidden_size", allow_default=False, error_location="EncoderLSTM - model_params"),
								num_layers = get_param_val(model_params, "num_layers", 1),
								dropout = get_param_val(model_params, "dropout", 0.0),
								bidirectional = True)


class EncoderHierarchicalRNN(EncoderModule):

	def __init__(self, model_params):
		super(EncoderHierarchicalRNN, self).__init__(model_params, data_embed=DATA_GLOVE)
		# TODO: Option for making semantics in VAE style (mu, sigma prediction)
		hidden_size = get_param_val(model_params, "hidden_size", allow_default=False, error_location="EncoderBiLSTM - model_params")
		dropout = get_param_val(model_params, "dropout", 0.0)
		self.num_layers = get_param_val(model_params, "num_layers", 1)
		self.share_attention_layer = get_param_val(model_params, "share_attention", False)
		self.use_prototype_styles = get_param_val(model_params, "use_prototype_styles", False)
		self.use_semantic_for_context_proto = get_param_val(model_params, "use_semantic_for_context_proto", False)
		self.no_prototypes_for_context = get_param_val(model_params, "no_prototypes_for_context", False)
		self.num_prototypes = get_param_val(model_params, "num_prototypes", 5)
		self.response_style_size = get_param_val(model_params, "response_style_size", -1)
		if self.response_style_size <= 0:
			self.response_style_size = self.style_size
		
		self.rnn_word_level = self._create_RNN(model_params)
		self.rnn_dropout_layer = RNNDropout(dropout)
		self.rnn_sent_level = PyTorchLSTMChain(input_size = hidden_size,
											   hidden_size = hidden_size,
											   num_layers = 1,
											   dropout = dropout,
											   bidirectional = False)

		self.attention_layer_semantics = WordLevelAttention(input_size_words = hidden_size,
															input_size_context = hidden_size,
															hidden_size = hidden_size)
		if self.share_attention_layer:
			self.attention_layer_style = self.attention_layer_semantics
		else:
			self.attention_layer_style = WordLevelAttention(input_size_words = hidden_size,
															input_size_context = hidden_size,
															hidden_size = hidden_size)
		self.semantic_layer = nn.Sequential(
				nn.BatchNorm1d(hidden_size),
				nn.Dropout(dropout),
				nn.Linear(hidden_size, self.semantic_size),
				nn.Tanh()
			)
		context_hidden_size = hidden_size + (self.semantic_size if self.use_semantic_for_context_proto else 0)
		if not self.use_prototype_styles:
			self.style_layer = nn.Sequential(
					nn.BatchNorm1d(hidden_size),
					nn.Dropout(dropout),
					nn.Linear(hidden_size, self.response_style_size * 2)
				)
		else:
			self.prototype_style_layer = PrototypeAttention(input_size_prototypes = self.response_style_size,
															input_size_context = hidden_size,
															hidden_size = hidden_size,
															num_prototypes = self.num_prototypes)
		if not self.use_prototype_styles or self.no_prototypes_for_context:
			self.context_style_layer = nn.Sequential(
				nn.BatchNorm1d(context_hidden_size),
				nn.Dropout(dropout),
				nn.Linear(context_hidden_size, self.style_size * 2)
			)
		else:
			self.prototype_context_style_layer = PrototypeAttention(input_size_prototypes = self.style_size,
																	input_size_context = context_hidden_size,
																	hidden_size = hidden_size,
																	num_prototypes = self.num_prototypes)

	def encode(self, embed_word_ids, embed_words, lengths, masks=None, attn_context_vector=None, additional_input=None, debug=False, debug_proto=False):
		if len(lengths.shape) == 2: # Hierarchical!
			batch_size = lengths.size(0)
			sent_size = lengths.size(1)
			# Stack extra length over batch
			flat_lengths = lengths.view(-1)
			flat_embed_word_ids = embed_word_ids.view(embed_word_ids.size(0) * embed_word_ids.size(1), embed_word_ids.size(2))
			flat_embed_words = embed_words.view(embed_words.size(0) * embed_words.size(1), embed_words.size(2), embed_words.size(3))

			word_level_final_states, word_level_outputs = self.rnn_word_level(flat_embed_words, flat_lengths)
			if self.num_layers > 1:
				word_level_final_states = word_level_final_states.view(word_level_final_states.shape[0], self.num_layers, -1)[:,-1,:]
			if attn_context_vector is None:
				context_state = word_level_final_states
			else:
				attn_context_vector = attn_context_vector.unsqueeze(dim=1).repeat(1,sent_size,1).view(-1,attn_context_vector.size(-1))
				assert all([attn_context_vector.size(i) == word_level_final_states.size(i) for i in range(len(attn_context_vector.shape))]), \
					   "Dimensions of given attention context vector must be %s, but got %s" % (str(word_level_final_states.shape), str(attn_context_vector.shape))
				context_state = attn_context_vector
			sent_embeds, style_attn_weights = self.attention_layer_style(encoder_word_embeds = word_level_outputs,
																		 encoder_lengths = flat_lengths,
																		 context_state = context_state)

			sent_embeds = sent_embeds.view(batch_size, sent_size, -1) # Split up again to batch and sequence length
			style_attn_weights = style_attn_weights.view(batch_size, sent_size, -1)
			style_attn_embed = sent_embeds
			if sent_size > 1:
				sent_embeds = self.rnn_dropout_layer(sent_embeds) # Dropout on the sentence embeddings
				sent_lengths = lengths.new_zeros(size=(batch_size,)) + sent_size # Assume same number of context sentences for all sentences 
				sent_level_final_states, sent_level_outputs = self.rnn_sent_level(sent_embeds, sent_lengths)
			else:
				sent_level_final_states = sent_embeds.squeeze(dim=1)

			if self.use_semantic_for_context_proto:
				sent_level_final_states = torch.cat([sent_level_final_states, additional_input], dim=-1)
			if not self.use_prototype_styles or self.no_prototypes_for_context:
				raw_context_style = self.context_style_layer(sent_level_final_states)
				style_mu, style_sigma = raw_context_style[:,:self.style_size], raw_context_style[:,self.style_size:]
				style_sigma = torch.exp(style_sigma)
				if False and self.training:
					style_embeds = sample_VAE_Gaussian(style_mu, style_sigma)
				else:
					style_embeds = style_mu
				proto_dist = None
			else:
				style_embeds, proto_dist = self.prototype_context_style_layer(sent_level_final_states)
				style_mu, style_sigma = None, None
			
			context_style = (style_embeds, style_mu, style_sigma)
			semantic_embeds, semantic_attn_embed, semantic_attn_weights = None, None, None

		elif len(lengths.shape) == 1:
			final_states, word_outputs = self.rnn_word_level(embed_words, lengths)
			if self.num_layers > 1:
				final_states = final_states.view(final_states.shape[0], self.num_layers, -1)[:,-1,:]
			if attn_context_vector is None:
				context_state = final_states
			else:
				assert all([attn_context_vector.size(i) == final_states.size(i) for i in range(len(attn_context_vector.shape))]), \
					   "Dimensions of given attention context vector must be %s, but got %s" % (str(final_states.shape), str(attn_context_vector.shape))
				context_state = attn_context_vector
			semantic_attn_embed, semantic_attn_weights = self.attention_layer_semantics(encoder_word_embeds = word_outputs,
																						encoder_lengths = lengths,
																						context_state = context_state)
			semantic_embeds = self.semantic_layer(semantic_attn_embed)
			style_attn_embed, style_attn_weights = self.attention_layer_style(encoder_word_embeds = word_outputs,
																			  encoder_lengths = lengths,
																			  context_state = context_state)
			if not self.use_prototype_styles:
				raw_response_style = self.style_layer(style_attn_embed)
				style_mu, style_sigma = raw_response_style[:,:self.response_style_size], raw_response_style[:,self.response_style_size:]
				style_sigma = torch.exp(style_sigma)
				if False and self.training:
					style_embeds = sample_VAE_Gaussian(style_mu, style_sigma)
				else:
					style_embeds = style_mu

				context_style = (style_embeds, style_mu, style_sigma)
				proto_dist = None
			else:
				combined_prototype, proto_dist = self.prototype_style_layer(style_attn_embed)
				context_style = (combined_prototype, None, None)

		else:
			print("[!] ERROR: Unknown shape size of lengths. Please provide either 1 or 2 dimensions. Given: %i dimensions" % (len(lengths.shape)))
			sys.exit(1)

		to_return = [semantic_embeds, context_style]
		if debug:
			to_return += [(semantic_attn_embed, semantic_attn_weights), (style_attn_embed, style_attn_weights)]
		if debug_proto:
			to_return += [proto_dist]
		return to_return

	def _create_RNN(self, model_params):
		raise NotImplementedError


class EncoderHierarchicalLSTM(EncoderHierarchicalRNN):

	def __init__(self, model_params):
		super(EncoderHierarchicalLSTM, self).__init__(model_params)

	def _create_RNN(self, model_params):
		return PyTorchLSTMChain(input_size = get_param_val(model_params, "input_size", allow_default=False, error_location="EncoderLSTM - model_params"),
								hidden_size = get_param_val(model_params, "hidden_size", allow_default=False, error_location="EncoderLSTM - model_params"),
								num_layers = get_param_val(model_params, "num_layers", 1),
								dropout = get_param_val(model_params, "dropout", 0.0),
								bidirectional = False)
		

class EncoderHierarchicalBiLSTM(EncoderHierarchicalRNN):

	def __init__(self, model_params):
		super(EncoderHierarchicalBiLSTM, self).__init__(model_params)

	def _create_RNN(self, model_params):
		return PyTorchLSTMChain(input_size = get_param_val(model_params, "input_size", allow_default=False, error_location="EncoderLSTM - model_params"),
								hidden_size = get_param_val(model_params, "hidden_size", allow_default=False, error_location="EncoderLSTM - model_params"),
								num_layers = get_param_val(model_params, "num_layers", 1),
								dropout = get_param_val(model_params, "dropout", 0.0),
								bidirectional = True)
		
