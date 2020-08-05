import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import sys
import os
import math
import time

from vocab import load_word2vec_from_file, get_num_slot_tokens, get_slot_token_start_index


# Class for summarizing all model types that can be used
class ModelTypes:
	BOW = 0
	LSTM = 1
	BILSTM = 2
	BERT = 3

	PAR_LSTM = 0
	PAR_LSTM_ATTN = 1
	PAR_POINTER = 2

	DEC_LSTM = 0

	DISC_LINEAR = 0

	def encoder_string():
		return "BOW: %d, LSTM: %d, BiLSTM: %d, BERT: %d" % (ModelTypes.BOW, ModelTypes.LSTM, ModelTypes.BILSTM, ModelTypes.BERT)

	def paraphrase_string():
		return "LSTM: %d, LSTM with Attention: %d, Pointer Network: %d" % (ModelTypes.PAR_LSTM, ModelTypes.PAR_LSTM_ATTN, ModelTypes.PAR_POINTER)

	def decoder_string():
		return "LSTM: %d" % (ModelTypes.DEC_LSTM)

	def discriminator_string():
		return "Linear: %d" % (ModelTypes.DISC_LINEAR)


######################
## HELPER FUNCTIONS ##
######################

def get_device():
	return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_param_val(param_dict, key, default_val=None, allow_default=True, error_location=""):
	if key in param_dict:
		return param_dict[key]
	elif allow_default:
		return default_val
	else:
		print("[!] ERROR (%s): could not find key \"%s\" in the dictionary although it is required." % (error_location, str(key)))
		sys.exit(1)

def get_word_mask(lengths, seq_length=None, start_ids=None):
	if seq_length is None:
		seq_length = max(1, lengths.max())
	word_positions = torch.arange(start=0, end=seq_length, dtype=lengths.dtype, device=lengths.device)
	word_positions = word_positions.reshape(shape=[1] * (len(lengths.shape)) + [-1, 1])
	lengths = lengths.unsqueeze(dim=-1).unsqueeze(dim=-1)
	mask = (word_positions < lengths).float()
	if start_ids is not None:
		mask = mask * (word_positions >= start_ids.unsqueeze(dim=-1).unsqueeze(dim=-1)).float()
	return mask	

def create_style_dropout_mask(style, length, training, p=0.4, p_complete=0.0, size_to_augment=-1):
	if p_complete == 1.0 or p == 0.0:
		if size_to_augment <= 0:
			style_dropout_mask = style.new_zeros(style.size(0), length, style.size(1))
		else:
			non_dropout_channels = style.new_ones(style.size(0), length, style.size(1) - size_to_augment)
			non_dropout_channels += (style.size(1) * 1.0 / (style.size(1) - size_to_augment) - 1)
			style_dropout_mask = style.new_zeros(style.size(0), length, size_to_augment)
			style_dropout_mask = torch.cat([style_dropout_mask, non_dropout_channels], dim=-1)
	else:
		if training:
			style_dropout_lengths = style.new_zeros(style.size(0)).uniform_()
			style_complete_dropout = style.new_zeros(style.size(0)).uniform_()
			style_complete_dropout_mask = (style_complete_dropout < p_complete).float()
			style_dropout_lengths = style_dropout_lengths + style_complete_dropout_mask
			acc_prob = p
			for i in range(length):
				style_dropout_lengths += (style_dropout_lengths <= (acc_prob)).float() * (i + 2 - style_dropout_lengths)
				acc_prob += p * ((1 - p) ** (i+1))
			style_dropout_lengths += (style_dropout_lengths < 2).float() * (length + 2 - style_dropout_lengths) - 2
			word_positions = torch.arange(start=0, end=length, dtype=style_dropout_lengths.dtype, device=style_dropout_lengths.device).unsqueeze(dim=0)
			style_dropout_lengths = style_dropout_lengths.unsqueeze(dim=1)
			style_dropout_mask = (style_dropout_lengths <= word_positions).float().unsqueeze(dim=-1)
			if size_to_augment > 0:
				non_dropout_channels = style_dropout_mask.new_ones(style.size(0), length, style.size(1) - size_to_augment)
				non_dropout_channels += (1 - style_dropout_mask) * (style.size(1) * 1.0 / (style.size(1) - size_to_augment) - 1)
				style_dropout_mask = style_dropout_mask.expand(-1, -1, size_to_augment)
				style_dropout_mask = torch.cat([style_dropout_mask, non_dropout_channels], dim=-1)
		else:
			style_dropout_mask = style.new_ones(style.size(0), length, style.size(1))
	return style_dropout_mask.detach()

###############################
## HIGH LEVEL MODEL TEMPLATE ##
###############################

class ModelTemplate(nn.Module):

	def __init__(self):
		super(ModelTemplate, self).__init__()

	def forward(self, _inputs, labels=None, beams=5, min_generation_steps=0, max_generation_steps=30):
		raise NotImplementedError		

	def get_loss_module(self, weight=None, ignore_index=-1, reduction='mean'):
		raise NotImplementedError

	def requires_BERT_input(self):
		raise NotImplementedError

###################################
## LOW LEVEL LSTM IMPLEMENTATION ##
###################################

class PyTorchLSTMChain(nn.Module):

	def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False):
		super(PyTorchLSTMChain, self).__init__()
		if bidirectional:
			assert hidden_size % 2 == 0, "[!] ERROR: Hidden size must be even for a Bidirectional LSTM!"
			hidden_size = int(hidden_size/2)
		self.lstm_cell = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
		self.hidden_size = hidden_size

	def forward(self, word_embeds, lengths):
		batch_size = word_embeds.shape[0]
		time_dim = word_embeds.shape[1]
		embed_dim = word_embeds.shape[2]

		sorted_lengths, perm_index = lengths.sort(0, descending=True)
		word_embeds = word_embeds[perm_index]

		packed_word_embeds = torch.nn.utils.rnn.pack_padded_sequence(word_embeds, sorted_lengths, batch_first=True)
		packed_outputs, (final_hidden_states, _) = self.lstm_cell(packed_word_embeds)
		outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

		# Redo sort
		_, unsort_indices = perm_index.sort(0, descending=False)
		outputs = outputs[unsort_indices]
		final_hidden_states = final_hidden_states[:,unsort_indices]

		final_states = final_hidden_states.transpose(0, 1).transpose(1, 2)
		final_states = final_states.reshape([batch_size, self.hidden_size * final_states.shape[-1]])

		return final_states, outputs


class WordLevelAttention(nn.Module):

	def __init__(self, input_size_words, input_size_context, hidden_size):
		super(WordLevelAttention, self).__init__()
		self.input_word_weights = nn.Linear(input_size_words, hidden_size)
		self.input_context_weights = nn.Linear(input_size_context, hidden_size)
		self.hidden_layer = nn.Linear(hidden_size, 1, bias=False)
		self.hidden_size = hidden_size

	def forward(self, encoder_word_embeds, encoder_lengths, context_state, encoder_start_ids=None, encoder_mask=None):
		"""
		Attention is calculated by a small MLP on word embeddings and a context state.
		a'_t = tanh(embeds * w_embed + context * w_context + b_inner) * w_final + b_final
		a_t = softmax(a'_t)

		Inputs:
			`encoder_word_embeds`: Word-level embeddings/features on which attention should be calculated. Shape: [B x seq_len x features_e]
			`encoder_lengths`: Lengths of the word sequence/sentence for every batch element. Shape: [B]
			`context_state`: Feature vector for the context for which we want to calculate the attention. Shape: [B x features_c]
			`encoder_start_ids`: Optional argument specifying the start position of valid word sequences. Can be used to mask out words from the first position on. Shape: [B]

		Outputs:
			`attention_features`: The summarized features based on the attention distribution. Shape: [B x features_e]
			`attention_weights`: Attention vector a_t. Shape: [B x seq_len]
		"""
		batch_size = encoder_word_embeds.size(0)
		seq_length = encoder_word_embeds.size(1)
		word_mask = get_word_mask(encoder_lengths, seq_length=encoder_word_embeds.shape[1], start_ids=encoder_start_ids)
		if encoder_mask is not None:
			word_mask = word_mask * encoder_mask.unsqueeze(dim=-1)

		if seq_length == 1:
			return encoder_word_embeds.squeeze(1), torch.ones_like(encoder_lengths, dtype=torch.float).unsqueeze(-1)

		word_features = self.input_word_weights(encoder_word_embeds) # B x seq x h
		context_features = self.input_context_weights(context_state) # B x h
		context_features = context_features.unsqueeze(1).expand(batch_size, seq_length, self.hidden_size) # B x seq x h
		word_context_features = torch.tanh(word_features + context_features) # B x seq x h

		attention_weights = self.hidden_layer(word_context_features) # B x seq x 1
		attention_weights = F.softmax(attention_weights, dim=1) * word_mask # B x seq x 1
		attention_weights = attention_weights / (1e-10 + attention_weights.sum(dim=1, keepdim=True)) # B x seq x 1

		attention_features = (encoder_word_embeds * attention_weights).sum(dim=1) # B x enc_features

		return attention_features, attention_weights.squeeze(-1)


class PrototypeAttention(WordLevelAttention):

	def __init__(self, input_size_prototypes, input_size_context, hidden_size, num_prototypes):
		super(PrototypeAttention, self).__init__(input_size_prototypes, input_size_context, hidden_size)
		self.prototypes = nn.Parameter(data=torch.randn(num_prototypes, input_size_prototypes), requires_grad=True)
		self.prototype_dropout = nn.Dropout(p=0.1) # For regularization purposes
		self.num_prototypes = num_prototypes
		self.sample_distribution = torch.distributions.dirichlet.Dirichlet(concentration=torch.tensor([0.25] * num_prototypes))
		self.spreadout_attn_dists = PrototypeAttention._create_prototype_distributions(self.num_prototypes)

	def forward(self, context_state):
		prototype_features = self.input_word_weights(self.prototype_dropout(self.prototypes)) # proto x h
		prototype_features = prototype_features.unsqueeze(dim=0) # 1 x proto x h
		context_features = self.input_context_weights(context_state) # B x h
		context_features = context_features.unsqueeze(dim=1) # B x 1 x h
		prototype_context_features = torch.tanh(prototype_features + context_features) # B x proto x h

		attention_weights = self.hidden_layer(self.prototype_dropout(prototype_context_features)) # B x proto x 1
		attention_weights = F.softmax(attention_weights, dim=1)

		attention_features = (self.prototypes.unsqueeze(dim=0) * attention_weights).sum(dim=1) # B x enc_features
		attention_features = self.prototype_dropout(attention_features)

		return attention_features, attention_weights.squeeze(-1)

	def sample_prototypes(self, batch_size):
		sampled_attn = self.sample_distribution.sample((batch_size,)).unsqueeze(dim=-1).to(get_device())
		attention_features = (self.prototypes.unsqueeze(dim=0) * sampled_attn).sum(dim=1)
		return attention_features

	def avg_prototypes(self, batch_size):
		attention_features = self.prototypes.mean(dim=0, keepdim=True)
		attention_features = attention_features.expand(batch_size, -1)
		return attention_features

	def generate_prototype_dist(self):
		attn_dists = self.spreadout_attn_dists.to(get_device())
		attention_features = (self.prototypes.unsqueeze(dim=0) * attn_dists.unsqueeze(dim=-1)).sum(dim=1)
		return attention_features, attn_dists

	@staticmethod
	def _create_prototype_distributions(num_prototypes):
		one_hot_vectors = np.eye(num_prototypes, dtype=np.float32)
		equal_mean = np.ones((1, num_prototypes), dtype=np.float32) / num_prototypes
		
		num_pairs = int(num_prototypes * (num_prototypes - 1) / 2)
		num_pairs = min(num_pairs, 100) # Do not allow more than 100 pairs to prevent explosion of this vector (scales quadratic with num_prototypes)
		pair_dists = np.zeros((num_pairs, num_prototypes), dtype=np.float32)
		pair_index = 0
		for proto_1 in range(num_prototypes):
			for proto_2 in range(num_prototypes):
				if proto_1 >= proto_2:
					continue
				pair_dists[pair_index, proto_1] = 0.5
				pair_dists[pair_index, proto_2] = 0.5
				pair_index += 1
				if pair_index >= num_pairs:
					break
			if pair_index >= num_pairs:
				break

		prototype_dists = np.concatenate([one_hot_vectors, equal_mean, pair_dists], axis=0)
		return torch.Tensor(prototype_dists)



#######################
## ADDITIONAL LAYERS ##
#######################

# Class widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/modules/input_variational_dropout.py
# Here copied from https://github.com/coetaur0/ESIM/blob/master/esim/layers.py
class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.

    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.

        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).

        Returns:
            A new tensor on which dropout has been applied.
        """
        ones = sequences_batch.data.new_ones(*(sequences_batch.shape[0:-2]+sequences_batch.shape[-1:]))
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(dim=-2) * sequences_batch


class EmbeddingModule(nn.Module):

	def __init__(self, model_params, wordvec_tensor, max_pos_embeds=10):
		super(EmbeddingModule, self).__init__()

		self.embeddings = nn.Embedding(wordvec_tensor.shape[0], wordvec_tensor.shape[1])
		with torch.no_grad():
			self.embeddings.weight.data.copy_(torch.from_numpy(wordvec_tensor))
			self.embeddings.weight.requires_grad = get_param_val(model_params, "finetune_embeds", False)
		self.embedding_dropout = RNNDropout(get_param_val(model_params, "embed_dropout", 0.0))
		self.use_slot_embeddings = get_param_val(model_params, "slot_embeddings", True)
		self.use_slot_value_embeddings = get_param_val(model_params, "slot_value_embeddings", True)
		self.positional_embedding_factor = get_param_val(model_params, "positional_embedding_factor", 0.25)
		self.num_slots = get_num_slot_tokens()
		self.slot_start_index = get_slot_token_start_index()
		self.embedding_size = wordvec_tensor.shape[1]
		self.num_words = wordvec_tensor.shape[0]

		if self.use_slot_embeddings:
			self.slot_embeddings = nn.Embedding(self.num_slots, wordvec_tensor.shape[1])
			self.slot_embeddings.weight.requires_grad = True

			self.positional_embeddings = nn.Embedding(max_pos_embeds, wordvec_tensor.shape[1])
			self.positional_embeddings.weight.data.copy_(self.generate_positional_encoding(max_pos_embeds))
			self.positional_embeddings.weight.requires_grad = False

			if self.use_slot_value_embeddings:
				self.slot_value_encoder = SlotEncoderBOW(model_params["slot_encoder_module"])
				self.gated_slot_embedding_layer = nn.Sequential(
						nn.Linear(wordvec_tensor.shape[1], wordvec_tensor.shape[1]),
						nn.Sigmoid(),
						nn.Dropout(get_param_val(model_params, "embed_dropout", 0.0))
					)


	def forward(self, _input, use_pos_encods=False, skip_slots=False, slot_mask=None, slot_vals=None, slot_lengths=None):
		if isinstance(_input, dict):
			_input = _input[DATA_GLOVE]
		embed_input = _input.clamp(min=0, max=self.num_words-1)
		embeds = self.embeddings(embed_input)

		if not skip_slots and self.use_slot_embeddings:
			slot_input = (self.slot_start_index - _input).clamp(min=0, max=self.num_slots-1)
			slot_embeds = self.slot_embeddings(slot_input)

			is_slot_embed = (_input <= self.slot_start_index).float()
			
			if self.use_slot_value_embeddings:
				if slot_mask is not None and slot_vals is not None and slot_lengths is not None:
					slot_value_embeddings = self.slot_value_encoder(slot_vals=slot_vals, slot_lengths=slot_lengths, embedding_module=self)
					
					step_size = torch.arange(start=0, end=slot_mask.size(0), dtype=slot_mask.dtype, device=slot_mask.device) * slot_value_embeddings.shape[1]
					extended_slot_mask = (slot_mask - 1) * (slot_mask < 0).long()
					extended_slot_mask = (extended_slot_mask + step_size.unsqueeze(dim=-1)).view(-1)

					slot_value_embeddings = torch.embedding(slot_value_embeddings.view(-1, slot_value_embeddings.size(-1)), extended_slot_mask).view(slot_mask.size(0), slot_mask.size(1), slot_value_embeddings.size(-1))
					slot_value_gate = self.gated_slot_embedding_layer(slot_embeds)
					
					slot_embeds = slot_embeds + slot_value_gate * slot_value_embeddings
				elif slot_vals is not None or slot_lengths is not None:
					print("[#] WARNING: At least one of the slot embedding values were None, although another was given.")
					print("\tSlot mask: %s, slot vals: %s, slot lengths: %s" % ("None" if (slot_mask is None) else "Not none", "None" if (slot_vals is None) else "Not none", "None" if (slot_lengths is None) else "Not none"))

			if use_pos_encods and len(_input.shape) == 2 and self.positional_embedding_factor != 0:
				pos_indices = self.generate_pos_indices(slot_input, is_slot_embed)
				slot_embeds = slot_embeds + self.positional_embedding_factor * self.positional_embeddings(pos_indices)

			is_slot_embed = is_slot_embed.view(*(list(_input.shape) + [1]))
			embeds = (1 - is_slot_embed) * embeds + is_slot_embed * slot_embeds
			

		embeds = self.embedding_dropout(embeds)
		return embeds

	def generate_mask(self, _input):
		with torch.no_grad():
			is_slot_embed = (_input <= self.slot_start_index).long()
			slot_ids = is_slot_embed * 0
			slot_ids[:,0] = is_slot_embed[:,0]
			for i in range(1,_input.size(1)):
				slot_ids[:,i] += slot_ids[:,i-1] + is_slot_embed[:,i]
			slot_ids *= is_slot_embed
		return slot_ids.detach()

	def generate_positional_encoding(self, num_pos, embed_size=None):
		if embed_size is None:
			embed_size = self.embedding_size
		pos_embeds = np.zeros((num_pos, embed_size), dtype=np.float32)
		for p in range(0, num_pos):
			for i in range(0, embed_size, 2):
				pos_embeds[p,i] = np.sin(p / np.power(10000, i * 1.0 / embed_size))
				pos_embeds[p,i+1] = np.cos(p / np.power(10000, i * 1.0 / embed_size))
		
		return torch.from_numpy(pos_embeds)

	def generate_pos_indices(self, slot_input, is_slot_embed):
		batch_size = slot_input.size(0)
		seq_length = slot_input.size(1)

		slot_input = slot_input - (1 - is_slot_embed.long()) # -1 if it is not a slot
		slot_positions = slot_input.new_zeros(size=(batch_size, seq_length))
		slot_counter = slot_input.new_zeros(size=(batch_size, self.num_slots))
		slot_nums = torch.arange(start=0, end=self.num_slots, dtype=slot_input.dtype, device=slot_input.device)[None,:]

		for i in range(seq_length):
			slot_app = (slot_nums == slot_input[:,i:i+1]).long()
			slot_positions[:,i] = (slot_app * slot_counter).sum(dim=-1)
			slot_counter = slot_counter + slot_app

		return slot_positions

class SlotEncoderBOW(nn.Module):

	def __init__(self, model_params):
		super(SlotEncoderBOW, self).__init__()
		self.use_CBOW = get_param_val(model_params, "use_CBOW", False)
		hidden_size = get_param_val(model_params, "hidden_size", -1)	
		if self.use_CBOW:
			self.CBOW_layer = nn.Linear(hidden_size, hidden_size)	

	def forward(self, slot_vals, slot_lengths, embedding_module):
		slot_embed_words = embedding_module(slot_vals, skip_slots=True)
		if self.use_CBOW:
			slot_embed_words = self.CBOW_layer(slot_embed_words)
		mask = get_word_mask(slot_lengths, seq_length=slot_embed_words.size(2))
		slot_embeds = (mask * slot_embed_words).sum(dim=-2) / (mask.sum(dim=-2) + 1e-5)
		return slot_embeds




if __name__ == '__main__':
	# _, word2id_dict, wordvec_tensor = load_word2vec_from_file()
	# model_params = {"slot_embeddings": True}
	# module = EmbeddingModule(model_params, wordvec_tensor)

	# words = [["<name>", "<name>", "here", "<name>"], ["<food>","<name>","at","<addr>"]]
	# words = [[word2id_dict[w] for w in sent] for sent in words]
	# np_embed_ids = np.array(words, dtype=np.long)
	# _input_tensor = torch.from_numpy(np_embed_ids)
	# masks = module.generate_mask(_input_tensor)
	# print("Masks: %s" % (str(masks)))

	# module(_input_tensor, use_pos_encods=True)
	# print(PrototypeAttention._create_prototype_distributions(num_prototypes=5))

	dropout_mask = create_style_dropout_mask(torch.zeros((8,3)), 6, training=True, p=0.4, p_complete=0.0, size_to_augment=2)
	print(dropout_mask.squeeze(dim=-1))







