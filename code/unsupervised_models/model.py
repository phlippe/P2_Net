import torch
import torch.nn as nn
import numpy as np 
import sys
sys.path.append("../")
import os
import math
import time
from data import DATA_GLOVE, DATA_BERT
from vocab import get_SOS_index, get_UNK_index
from model_utils import *
from unsupervised_models.model_encoder import createEncoder, EncoderRNN
from unsupervised_models.model_decoder import createDecoder


class ModelUnsupervisedParaphrasingTemplate(ModelTemplate):

	def __init__(self, model_params, wordvec_tensor):
		super(ModelUnsupervisedParaphrasingTemplate, self).__init__()

		self.embedding_module = EmbeddingModule(model_params, wordvec_tensor)
		self.encoder_module = createEncoder(model_params["encoder_module"])
		self.decoder_module = createDecoder(model_params["decoder_module"], self.embedding_module)
		self.semantics_dropout = nn.Dropout(p=get_param_val(model_params, "semantics_dropout", 0.0))

	def forward(self, _input, teacher_forcing=False, switch_rate=1.0, labels_1=None, labels_2=None):
		"""
		Forward method for overall model. If labels are given, the decoder is performing teacher forcing.
		Otherwise, we perform Beam-search to find N best paraphrases
		"""
		par_1_words, par_1_lengths, par_1_mask, par_2_words, par_2_lengths, par_2_mask = _input

		# Masking out nouns that are replaced by "UNK" token
		par_1_words[DATA_GLOVE] = par_1_words[DATA_GLOVE] * (par_1_mask <= 0).long() + get_UNK_index() * (par_1_mask > 0).long()
		par_2_words = par_2_words * (par_2_mask <= 0).long() + get_UNK_index() * (par_2_mask > 0).long()

		par_1_words, par_1_word_embeds, par_1_lengths = self.encoder_module.prepare(par_1_words, par_1_lengths, self.embedding_module)
		par_2_words, par_2_word_embeds, par_2_lengths = self.encoder_module.prepare(par_2_words, par_2_lengths, self.embedding_module)

		par_1_semantics, par_1_style, par_1_UNK_tokens, par_1_UNK_lengths = self.encoder_module(par_1_words, par_1_word_embeds, par_1_lengths, par_1_mask)
		par_2_semantics, par_2_style, par_2_UNK_tokens, par_2_UNK_lengths = self.encoder_module(par_2_words, par_2_word_embeds, par_2_lengths, par_2_mask)
		par_1_style, par_1_style_mu, par_1_style_sigma = par_1_style
		par_2_style, par_2_style_mu, par_2_style_sigma = par_2_style
		par_1_semantics = self.semantics_dropout(par_1_semantics)
		par_2_semantics = self.semantics_dropout(par_2_semantics)

		# print("Par 1 UNK tokens: \n\tshape = %s\n\tvals = %s" % (str(par_1_UNK_tokens.shape), str(par_1_UNK_tokens[:,:,0])))
		# print("Par 2 UNK tokens: \n\tshape = %s\n\tvals = %s" % (str(par_2_UNK_tokens.shape), str(par_2_UNK_tokens[:,:,0])))
		# sys.exit(1)

		if switch_rate > 0.0:
			switch_vec = (par_1_semantics.new_zeros(par_1_semantics.shape[0],1).uniform_() > (1 - switch_rate)).float()
			switched_sem_1 = (1 - switch_vec) * par_1_semantics + switch_vec * par_2_semantics
			switched_sem_2 = (1 - switch_vec) * par_2_semantics + switch_vec * par_1_semantics
			switched_UNK_tokens_1 = (1 - switch_vec[:,:,None]) * par_1_UNK_tokens + switch_vec[:,:,None] * par_2_UNK_tokens 
			switched_UNK_tokens_2 = (1 - switch_vec[:,:,None]) * par_2_UNK_tokens + switch_vec[:,:,None] * par_1_UNK_tokens
			switched_UNK_lengths_1 = (1 - switch_vec.squeeze(dim=-1)) * par_1_UNK_lengths + switch_vec.squeeze(dim=-1) * par_2_UNK_lengths 
			switched_UNK_lengths_2 = (1 - switch_vec.squeeze(dim=-1)) * par_2_UNK_lengths + switch_vec.squeeze(dim=-1) * par_1_UNK_lengths
		else:
			switched_sem_1 = par_1_semantics
			switched_sem_2 = par_2_semantics
			switched_UNK_tokens_1 = par_1_UNK_tokens 
			switched_UNK_tokens_2 = par_2_UNK_tokens
			switched_UNK_lengths_1 = par_1_UNK_lengths
			switched_UNK_lengths_2 = par_2_UNK_lengths
		# print("Switch vector: %s" % (str(switch_vec)))
		# print("Switched UNK tokens 1: %s\nSwitched UNK tokens 2: %s" % (str(switched_UNK_tokens_1[:,:,0]), str(switched_UNK_tokens_2[:,:,0])))

		if labels_1 is None:
			labels_1 = par_1_words
		if labels_2 is None:
			labels_2 = par_2_words
		 # Check if the first label token is always start-of-sentence. If so, remove it 
		if (labels_1[:,0] == get_SOS_index()).byte().all():
			labels_1 = labels_1[:,1:]
		if (labels_2[:,0] == get_SOS_index()).byte().all():
			labels_2 = labels_2[:,1:]
		# Applies paraphrase module for text generation
		par_1_decoder_res = self.decoder_module(semantics = switched_sem_1, 
												styles = par_1_style,
												labels = labels_1 if teacher_forcing else None, 
												min_generation_steps = labels_1.shape[1], 
												max_generation_steps = labels_1.shape[1],
												UNK_embeds = switched_UNK_tokens_1,
												UNK_lengths = switched_UNK_lengths_1)
		par_2_decoder_res = self.decoder_module(semantics = switched_sem_2, 
												styles = par_2_style,
												labels = labels_2 if teacher_forcing else None, 
												min_generation_steps = labels_2.shape[1], 
												max_generation_steps = labels_2.shape[1],
												UNK_embeds = switched_UNK_tokens_2,
												UNK_lengths = switched_UNK_lengths_2)

		return par_1_decoder_res, par_2_decoder_res, (par_1_semantics, par_1_style_mu, par_1_style_sigma), (par_2_semantics, par_2_style_mu, par_2_style_sigma)

	def reconstruct(self, _input, teacher_forcing=False, labels=None):
		par_1_words, par_1_lengths, par_1_masks = _input
		par_1_words, par_1_word_embeds, par_1_lengths = self.encoder_module.prepare(par_1_words, par_1_lengths, self.embedding_module)
		
		par_1_semantics, par_1_style, par_1_UNK_tokens, par_1_UNK_lengths = self.encoder_module(par_1_words, par_1_word_embeds, par_1_lengths, par_1_masks)
		par_1_style, par_1_style_mu, par_1_style_sigma = par_1_style
		par_1_semantics = self.semantics_dropout(par_1_semantics)

		if labels is None:
			labels = par_1_words
		if (labels[:,0] == get_SOS_index()).byte().all():
			labels = labels[:,1:]
		# Applies paraphrase module for text generation
		par_1_decoder_res = self.decoder_module(semantics = par_1_semantics, 
												styles = par_1_style,
												labels = labels if teacher_forcing else None, 
												min_generation_steps = labels.shape[1], 
												max_generation_steps = labels.shape[1],
												UNK_embeds = par_1_UNK_tokens,
												UNK_lengths = par_1_UNK_lengths)
		
		return par_1_decoder_res, (par_1_semantics, par_1_style_mu, par_1_style_sigma)

	def sample_reconstruction_styles(self, _input, num_samples=4, max_generation_steps=30):
		with torch.no_grad():
			par_1_words, par_1_lengths, par_1_masks = _input
			par_1_words, par_1_word_embeds, par_1_lengths = self.encoder_module.prepare(par_1_words, par_1_lengths, self.embedding_module)
			
			par_1_semantics, par_1_style, par_1_UNK_tokens, par_1_UNK_lengths = self.encoder_module(par_1_words, par_1_word_embeds, par_1_lengths, par_1_masks)
			par_1_style, par_1_style_mu, par_1_style_sigma = par_1_style
			
			# Stack the different styles as different batch elements
			par_1_semantics = par_1_semantics[:,None,:].expand(-1, num_samples, -1).contiguous().view(-1, par_1_semantics.size(1))
			par_1_samp_styles = torch.randn(par_1_style.size(0), num_samples-1, par_1_style.size(1), device=get_device())
			par_1_samp_styles = torch.cat([par_1_style[:,None,:], par_1_samp_styles], dim=1).view(-1, par_1_style.size(1))
			par_1_UNK_tokens = par_1_UNK_tokens[:,None,:,:].expand(-1, num_samples,-1,-1).contiguous().view(-1, par_1_UNK_tokens.size(1), par_1_UNK_tokens.size(2))
			par_1_UNK_lengths = par_1_UNK_lengths[:,None].expand(-1, num_samples).contiguous().view(-1)

			# Applies paraphrase module for text generation
			par_1_decoder_res = self.decoder_module(semantics = par_1_semantics, 
													styles = par_1_samp_styles,
													labels = None, 
													max_generation_steps = max_generation_steps,
													UNK_embeds = par_1_UNK_tokens,
													UNK_lengths = par_1_UNK_lengths)
			# Reshape styles over batch elements to an extra dimension
			gen_outputs, gen_UNK_weights, gen_preds, gen_lengths = par_1_decoder_res
			gen_outputs = gen_outputs.view(par_1_words.size(0), num_samples, gen_outputs.size(1), gen_outputs.size(2))
			gen_UNK_weights = gen_UNK_weights.view(par_1_words.size(0), num_samples, gen_UNK_weights.size(1), gen_UNK_weights.size(2))
			gen_preds = gen_preds.view(par_1_words.size(0), num_samples, gen_preds.size(1))
			gen_lengths = gen_lengths.view(par_1_words.size(0), num_samples)

			return gen_outputs, gen_UNK_weights, gen_preds, gen_lengths

	def question_answer_switch(self, _input, teacher_forcing=False):
		"""
		Forward method for overall model. If labels are given, the decoder is performing teacher forcing.
		Otherwise, we perform Beam-search to find N best paraphrases
		"""
		quest_words, quest_lengths, quest_mask, answ_words, answ_lengths, answ_mask = _input

		# Masking out nouns that are replaced by "UNK" token
		quest_words[DATA_GLOVE] = quest_words[DATA_GLOVE] * (quest_mask <= 0).long() + get_UNK_index() * (quest_mask > 0).long()
		answ_words = answ_words * (answ_mask <= 0).long() + get_UNK_index() * (answ_mask > 0).long()

		quest_words, quest_word_embeds, quest_lengths = self.encoder_module.prepare(quest_words, quest_lengths, self.embedding_module)
		answ_words, answ_word_embeds, answ_lengths = self.encoder_module.prepare(answ_words, answ_lengths, self.embedding_module)

		_, quest_style, _, _ = self.encoder_module(quest_words, quest_word_embeds, quest_lengths, quest_mask)
		answ_semantics, _, answ_UNK_tokens, answ_UNK_lengths = self.encoder_module(answ_words, answ_word_embeds, answ_lengths, answ_mask)
		quest_style, quest_style_mu, quest_style_sigma = quest_style
		answ_semantics = self.semantics_dropout(answ_semantics)

		labels = answ_words
		if (labels[:,0] == get_SOS_index()).byte().all():
			labels = labels[:,1:]
		# Applies paraphrase module for text generation
		answ_decoder_res = self.decoder_module(semantics = answ_semantics, 
												styles = quest_style,
												labels = labels if teacher_forcing else None, 
												min_generation_steps = labels.shape[1], 
												max_generation_steps = labels.shape[1],
												UNK_embeds = answ_UNK_tokens,
												UNK_lengths = answ_UNK_lengths)

		return answ_decoder_res, (answ_semantics, quest_style_mu, quest_style_sigma)

	def get_loss_module(self, weight=None, ignore_index=-1, reduction='mean'):
		return self.decoder_module.get_loss_module(weight=weight, ignore_index=ignore_index, reduction=reduction)

	def requires_BERT_input(self):
		return (False, False) # For now, we do not use BERT encoders. Are too heavy encoders


class ModelUnsupervisedContextParaphrasingTemplate(ModelTemplate):

	def __init__(self, model_params, wordvec_tensor):
		super(ModelUnsupervisedContextParaphrasingTemplate, self).__init__()

		self.embedding_module = EmbeddingModule(model_params, wordvec_tensor)
		self.encoder_module = createEncoder(model_params["encoder_module"], hierarchical_model=True)
		self.decoder_module = createDecoder(model_params["decoder_module"], self.embedding_module)
		self.semantics_dropout = nn.Dropout(p=get_param_val(model_params, "semantics_dropout", 0.0))
		self.style_exponential_dropout = get_param_val(model_params, "style_exponential_dropout", 0.0)
		self.style_full_dropout = get_param_val(model_params, "style_full_dropout", 0.0)

	def forward(self, _input, teacher_forcing=False, switch_rate=1.0, labels_1=None, labels_2=None, semantic_full_dropout=0.0, teacher_forcing_ratio=1.0, additional_supervision=False, max_generation_steps=None, use_semantic_specific_attn=False, use_context_style=False, ignore_context=False, beams=-1, only_par_1=False, beam_search_method="diverse"):
		"""
		Forward method for overall model. If labels are given, the decoder is performing teacher forcing.
		Otherwise, we perform Beam-search to find N best paraphrases
		"""
		par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = _input

		par_1_words, par_1_word_embeds, par_1_lengths = self.encoder_module.prepare(par_1_words, par_1_lengths, self.embedding_module, slot_mask=par_1_masks, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths)
		par_2_words, par_2_word_embeds, par_2_lengths = self.encoder_module.prepare(par_2_words, par_2_lengths, self.embedding_module, slot_mask=par_2_masks, slot_vals=par_2_slots, slot_lengths=par_2_slot_lengths)
		contexts_1_words, contexts_1_word_embeds, contexts_1_lengths = self.encoder_module.prepare(contexts_1_words, contexts_1_lengths, self.embedding_module)
		contexts_2_words, contexts_2_word_embeds, contexts_2_lengths = self.encoder_module.prepare(contexts_2_words, contexts_2_lengths, self.embedding_module)
		# Note that the slots for both sentences are the same, but the order can be different. Important for later loss calculation
		slot_1_embeds, slot_1_lengths, slot_1_ids = EncoderRNN._gather_UNK_token_embeddings(par_1_word_embeds, par_1_masks, word_ids=par_1_words)
		slot_2_embeds, slot_2_lengths, slot_2_ids = EncoderRNN._gather_UNK_token_embeddings(par_2_word_embeds, par_2_masks, word_ids=par_2_words)

		par_1_semantics, par_1_style, par_1_attn_semantic, _, par_1_proto_dist = self.encoder_module(par_1_words, par_1_word_embeds, par_1_lengths, masks=None, debug=True, debug_proto=True)
		par_2_semantics, par_2_style, par_2_attn_semantic, _, par_2_proto_dist = self.encoder_module(par_2_words, par_2_word_embeds, par_2_lengths, masks=None, debug=True, debug_proto=True)
		par_1_style, par_1_style_mu, par_1_style_sigma = par_1_style
		par_2_style, par_2_style_mu, par_2_style_sigma = par_2_style
		
		if switch_rate > 0.0:
			switch_vec = (par_1_semantics.new_zeros(par_1_semantics.shape[0],1).uniform_() > (1 - switch_rate)).float()
			switched_sem_1 = (1 - switch_vec) * par_1_semantics + switch_vec * par_2_semantics
			switched_sem_2 = (1 - switch_vec) * par_2_semantics + switch_vec * par_1_semantics
			context_attn_vec_1 = (1 - switch_vec) * par_1_attn_semantic[0] + switch_vec * par_2_attn_semantic[0]
			context_attn_vec_2 = (1 - switch_vec) * par_2_attn_semantic[0] + switch_vec * par_1_attn_semantic[0]
		else:
			switched_sem_1 = par_1_semantics
			switched_sem_2 = par_2_semantics
			context_attn_vec_1 = par_1_attn_semantic[0]
			context_attn_vec_2 = par_2_attn_semantic[0]
		
		if not ignore_context:		
			if contexts_1_words.size(-2) == contexts_2_words.size(-2):
				batch_size = contexts_1_words.size(0)
				_, contexts_styles, contexts_proto_dists = self.encoder_module(embed_word_ids=torch.cat([contexts_1_words, contexts_2_words], dim=0),
																			   embed_words=torch.cat([contexts_1_word_embeds, contexts_2_word_embeds], dim=0),
																			   lengths=torch.cat([contexts_1_lengths, contexts_2_lengths], dim=0),
																			   masks=None,
																			   attn_context_vector=torch.cat([context_attn_vec_1, context_attn_vec_2], dim=0) if use_semantic_specific_attn else None,
																			   additional_input=torch.cat([switched_sem_1, switched_sem_2], dim=0),
																			   debug_proto=True)
				contexts_styles, contexts_styles_mu, contexts_styles_sigma = contexts_styles
				contexts_1_style = contexts_styles[:batch_size]
				contexts_2_style = contexts_styles[batch_size:]
				if contexts_styles_mu is not None and contexts_styles_sigma is not None:
					contexts_1_style_mu, contexts_1_style_sigma = contexts_styles_mu[:batch_size], contexts_styles_sigma[:batch_size]
					contexts_2_style_mu, contexts_2_style_sigma = contexts_styles_mu[batch_size:], contexts_styles_sigma[batch_size:]
				else:
					contexts_1_style_mu, contexts_1_style_sigma, contexts_2_style_mu, contexts_2_style_sigma = None, None, None, None
				if contexts_proto_dists is not None:
					context_1_proto_dist, context_2_proto_dist = contexts_proto_dists[:batch_size], contexts_proto_dists[batch_size:]
				else:
					context_1_proto_dist, context_2_proto_dist = None, None

			else:
				_, contexts_1_style, context_1_proto_dist = self.encoder_module(contexts_1_words, contexts_1_word_embeds, contexts_1_lengths, masks=None, attn_context_vector=context_attn_vec_1 if use_semantic_specific_attn else None, additional_input=switched_sem_1, debug_proto=True)
				_, contexts_2_style, context_2_proto_dist = self.encoder_module(contexts_2_words, contexts_2_word_embeds, contexts_2_lengths, masks=None, attn_context_vector=context_attn_vec_2 if use_semantic_specific_attn else None, additional_input=switched_sem_2, debug_proto=True)
				contexts_1_style, contexts_1_style_mu, contexts_1_style_sigma = contexts_1_style
				contexts_2_style, contexts_2_style_mu, contexts_2_style_sigma = contexts_2_style
		else:
			context_1_proto_dist, context_2_proto_dist = par_1_proto_dist.new_zeros(*(par_1_proto_dist.shape[:-1] + (self.encoder_module.style_size,))), par_2_proto_dist.new_zeros(*par_2_proto_dist.shape)
			contexts_1_style, contexts_1_style_mu, contexts_1_style_sigma = par_1_style.new_zeros(*(par_1_style.shape[:-1] + (self.encoder_module.style_size,))), None, None
			contexts_2_style, contexts_2_style_mu, contexts_2_style_sigma = par_2_style.new_zeros(*(par_2_style.shape[:-1] + (self.encoder_module.style_size,))), None, None

		switched_sem_1 = self.semantics_dropout(switched_sem_1)
		switched_sem_2 = self.semantics_dropout(switched_sem_2)

		if use_context_style:

			# par_1_style = self.encoder_module.prototype_style_layer.avg_prototypes(par_1_style.size(0))
			# par_2_style = self.encoder_module.prototype_style_layer.avg_prototypes(par_2_style.size(0))
			par_1_style = par_1_style * 0.0
			par_2_style = par_2_style * 0.0
			contexts_1_style = contexts_1_style * ((contexts_1_style.size(1) + par_1_style.size(1)) * 1.0 / contexts_1_style.size(1))
			contexts_2_style = contexts_2_style * ((contexts_2_style.size(1) + par_2_style.size(1)) * 1.0 / contexts_2_style.size(1))

		cat_contexts_1_style = torch.cat([par_1_style, contexts_1_style], dim=-1)
		cat_contexts_2_style = torch.cat([par_2_style, contexts_2_style], dim=-1)
		
		# cat_contexts_1_style = par_1_style
		# cat_contexts_2_style = par_2_style
		# elif not ignore_context:
		# 	cat_contexts_1_style = contexts_1_style
		# 	cat_contexts_2_style = contexts_2_style
		# else:
		# 	cat_contexts_1_style = par_1_style
		# 	cat_contexts_2_style = par_2_style

		if semantic_full_dropout > 0.0:
			dropout_vec = (switched_sem_1.new_zeros(switched_sem_1.shape[0],1).uniform_() <= semantic_full_dropout).float()
			switched_sem_1 = (1 - dropout_vec) * switched_sem_1
			switched_sem_2 = (1 - dropout_vec) * switched_sem_2
			dropped_contexts_1_style = cat_contexts_1_style * ((1 - dropout_vec) + dropout_vec * ((switched_sem_1.size(1) + cat_contexts_1_style.size(1)) / (1.0 * cat_contexts_1_style.size(1))))
			dropped_contexts_2_style = cat_contexts_2_style * ((1 - dropout_vec) + dropout_vec * ((switched_sem_2.size(1) + cat_contexts_2_style.size(1)) / (1.0 * cat_contexts_2_style.size(1))))
		else:
			dropped_contexts_1_style = cat_contexts_1_style
			dropped_contexts_2_style = cat_contexts_2_style

		if labels_1 is None:
			labels_1 = par_1_words
		if labels_2 is None:
			labels_2 = par_2_words
		 # Check if the first label token is always start-of-sentence. If so, remove it 
		if (labels_1[:,0] == get_SOS_index()).byte().all():
			labels_1 = labels_1[:,1:]
		if (labels_2[:,0] == get_SOS_index()).byte().all():
			labels_2 = labels_2[:,1:]

		if additional_supervision:
			raise NotImplementedError
			labels_1 = labels_1.repeat(2,1)
			labels_2 = labels_2.repeat(2,1)
			switched_sem_1 = switched_sem_1.repeat(2,1)
			switched_sem_2 = switched_sem_2.repeat(2,1)
			dropped_contexts_1_style = torch.cat([dropped_contexts_1_style, par_1_style], dim=0)
			dropped_contexts_2_style = torch.cat([dropped_contexts_2_style, par_2_style], dim=0)
			cont_slot_1_embeds = slot_1_embeds.repeat(2,1,1)
			cont_slot_2_embeds = slot_2_embeds.repeat(2,1,1)
			cont_slot_1_lengths = slot_1_lengths.repeat(2)
			cont_slot_2_lengths = slot_2_lengths.repeat(2)
			cont_slot_1_ids = slot_1_ids.repeat(2,1)
			cont_slot_2_ids = slot_2_ids.repeat(2,1)
		else:
			cont_slot_1_embeds, cont_slot_1_lengths, cont_slot_1_ids = slot_1_embeds, slot_1_lengths, slot_1_ids
			cont_slot_2_embeds, cont_slot_2_lengths, cont_slot_2_ids = slot_2_embeds, slot_2_lengths, slot_2_ids

		style_dropout_mask_1 = create_style_dropout_mask(dropped_contexts_1_style, labels_1.shape[1], 
														 training=self.training, 
														 size_to_augment=self.encoder_module.response_style_size,
														 p=1.0 if use_context_style else 1 - self.style_exponential_dropout,  
														 p_complete = 0.0 if use_context_style else self.style_full_dropout)
		style_dropout_mask_2 = create_style_dropout_mask(dropped_contexts_2_style, labels_2.shape[1], 
														 training=self.training, 
														 size_to_augment=self.encoder_module.response_style_size,
														 p=1.0 if use_context_style else 1 - self.style_exponential_dropout,
														 p_complete = 0.0 if use_context_style else self.style_full_dropout)

		if use_context_style:
			assert (style_dropout_mask_1 != 1.0).float().sum() == 0, "[!] ERROR: Dropout mask (1) is applied although pure context style is used."
			assert (style_dropout_mask_2 != 1.0).float().sum() == 0, "[!] ERROR: Dropout mask (2) is applied although pure context style is used."

		# Applies paraphrase module for text generation
		par_1_decoder_res = self.decoder_module(semantics = switched_sem_1, 
												styles = dropped_contexts_1_style,
												style_dropout_mask = style_dropout_mask_1,
												labels = labels_1 if teacher_forcing else None, 
												teacher_forcing_ratio = teacher_forcing_ratio,
												beams = beams,
												beam_search_method = beam_search_method,
												min_generation_steps = labels_1.shape[1], 
												max_generation_steps = labels_1.shape[1] if max_generation_steps is None else max_generation_steps,
												UNK_embeds = cont_slot_1_embeds,
												UNK_lengths = cont_slot_1_lengths,
												UNK_word_ids = cont_slot_1_ids)
		if only_par_1:
			par_2_decoder_res = None
		else:
			par_2_decoder_res = self.decoder_module(semantics = switched_sem_2, 
													styles = dropped_contexts_2_style,
													style_dropout_mask = style_dropout_mask_2,
													labels = labels_2 if teacher_forcing else None, 
													teacher_forcing_ratio = teacher_forcing_ratio,
													beams = beams,
													beam_search_method = beam_search_method,
													min_generation_steps = labels_2.shape[1], 
													max_generation_steps = labels_2.shape[1] if max_generation_steps is None else max_generation_steps,
													UNK_embeds = cont_slot_2_embeds,
													UNK_lengths = cont_slot_2_lengths,
													UNK_word_ids = cont_slot_2_ids)

		return par_1_decoder_res, \
			   par_2_decoder_res, \
			   (contexts_1_style, contexts_1_style_mu, contexts_1_style_sigma), \
			   (contexts_2_style, contexts_2_style_mu, contexts_2_style_sigma), \
			   (par_1_style, par_1_style_mu, par_1_style_sigma), \
			   (par_2_style, par_2_style_mu, par_2_style_sigma), \
			   (par_1_semantics, par_2_semantics), \
			   (slot_1_embeds, slot_1_lengths, slot_1_ids), \
			   (slot_2_embeds, slot_2_lengths, slot_2_ids), \
			   (par_1_proto_dist, par_2_proto_dist, context_1_proto_dist, context_2_proto_dist)

	def sample_reconstruction_styles(self, _input, num_samples=4, max_generation_steps=30, sample_gt=True, sample_context=False):
		with torch.no_grad():
			if len(_input) == 5:
				par_1_words, par_1_lengths, par_1_masks, par_1_slots, par_1_slot_lengths = _input
				contexts_1_words, contexts_1_lengths = None, None
				gt_1_words, gt_1_lengths = None, None
			elif len(_input) == 7:
				par_1_words, par_1_lengths, par_1_masks, par_1_slots, par_1_slot_lengths, contexts_1_words, contexts_1_lengths = _input
				gt_1_words, gt_1_lengths = None, None
			elif len(_input) == 9:
				par_1_words, par_1_lengths, par_1_masks, par_1_slots, par_1_slot_lengths, contexts_1_words, contexts_1_lengths, gt_1_words, gt_1_lengths = _input
			else:
				print("[!] ERROR: Unknown number of input arguments given: %i" % (len(_input)))
				sys.exit(1)
			par_1_words, par_1_word_embeds, par_1_lengths = self.encoder_module.prepare(par_1_words, par_1_lengths, self.embedding_module, slot_mask=par_1_masks, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths)
			
			par_1_semantics, par_1_style = self.encoder_module(par_1_words, par_1_word_embeds, par_1_lengths, par_1_masks)
			slot_1_embeds, slot_1_lengths, slot_1_ids = EncoderRNN._gather_UNK_token_embeddings(par_1_word_embeds, par_1_masks, word_ids=par_1_words)
		
			if not sample_context and contexts_1_words is not None:
				contexts_1_words, contexts_1_word_embeds, contexts_1_lengths = self.encoder_module.prepare(contexts_1_words, contexts_1_lengths, self.embedding_module)
				_, context_1_samp_styles = self.encoder_module(contexts_1_words, contexts_1_word_embeds, contexts_1_lengths, masks=None, attn_context_vector=None, additional_input=par_1_semantics, debug_proto=False)
				context_1_samp_styles, _, _ = context_1_samp_styles
				context_1_samp_styles = context_1_samp_styles[:,None,:].expand(-1, num_samples, -1).contiguous().view(-1, context_1_samp_styles.size(1))
			else:
				if not self.encoder_module.use_prototype_styles or self.encoder_module.no_prototypes_for_context:
					context_1_samp_styles = torch.randn(par_1_words.size(0) * num_samples, self.encoder_module.style_size, device=get_device())
				else:
					context_1_samp_styles = self.encoder_module.prototype_context_style_layer.sample_prototypes(par_1_words.size(0) * num_samples)

			if sample_gt:
				if self.encoder_module.use_prototype_styles:
					par_1_samp_styles = self.encoder_module.prototype_style_layer.sample_prototypes(par_1_words.size(0) * num_samples)
				else:
					par_1_samp_styles = torch.randn(par_1_words.size(0) * num_samples, self.encoder_module.response_style_size, device=get_device())
			else:
				par_1_samp_styles, _, _ = par_1_style
				par_1_samp_styles = par_1_samp_styles[:,None,:].expand(-1, num_samples, -1).contiguous().view(-1, par_1_samp_styles.size(1))
			par_1_samp_styles = torch.cat([par_1_samp_styles, context_1_samp_styles], dim=-1)
			
			# Stack the different styles as different batch elements
			par_1_semantics = par_1_semantics[:,None,:].expand(-1, num_samples, -1).contiguous().view(-1, par_1_semantics.size(1))
			slot_1_embeds = slot_1_embeds[:,None,:,:].expand(-1, num_samples,-1,-1).contiguous().view(-1, slot_1_embeds.size(1), slot_1_embeds.size(2))
			slot_1_ids = slot_1_ids[:,None,:].expand(-1, num_samples, -1).contiguous().view(-1, slot_1_ids.size(1))
			slot_1_lengths = slot_1_lengths[:,None].expand(-1, num_samples).contiguous().view(-1)

			# Applies paraphrase module for text generation
			par_1_decoder_res = self.decoder_module(semantics = par_1_semantics, 
													styles = par_1_samp_styles,
													labels = None, 
													max_generation_steps = max_generation_steps,
													UNK_embeds = slot_1_embeds,
													UNK_lengths = slot_1_lengths,
													UNK_word_ids = slot_1_ids)
			# Reshape styles over batch elements to an extra dimension
			gen_outputs, gen_slot_weights, gen_preds, gen_lengths = par_1_decoder_res
			gen_outputs = gen_outputs.view(par_1_words.size(0), num_samples, gen_outputs.size(1), gen_outputs.size(2))
			gen_slot_weights = gen_slot_weights.view(par_1_words.size(0), num_samples, gen_slot_weights.size(1), gen_slot_weights.size(2))
			gen_preds = gen_preds.view(par_1_words.size(0), num_samples, gen_preds.size(1))
			gen_lengths = gen_lengths.view(par_1_words.size(0), num_samples)

			return gen_outputs, gen_slot_weights, gen_preds, gen_lengths

	def generate_style_dist(self, _input, max_generation_steps=30):
		if not self.encoder_module.use_prototype_styles:
			return self.sample_reconstruction_styles(_input, max_generation_steps=max_generation_steps)
		else:
			with torch.no_grad():
				par_1_words, par_1_lengths, par_1_masks, par_1_slots, par_1_slot_lengths = _input
				par_1_words, par_1_word_embeds, par_1_lengths = self.encoder_module.prepare(par_1_words, par_1_lengths, self.embedding_module, slot_mask=par_1_masks, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths)
				
				par_1_semantics, _, par_1_proto_dist = self.encoder_module(par_1_words, par_1_word_embeds, par_1_lengths, par_1_masks, debug_proto=True)
				slot_1_embeds, slot_1_lengths, slot_1_ids = EncoderRNN._gather_UNK_token_embeddings(par_1_word_embeds, par_1_masks, word_ids=par_1_words)
			
				proto_samp_styles, proto_dists = self.encoder_module.prototype_style_layer.generate_prototype_dist()
				proto_context_samp_styles, _ = self.encoder_module.prototype_context_style_layer.generate_prototype_dist()
				proto_samp_styles = torch.cat([proto_samp_styles, proto_context_samp_styles], dim=-1)
				num_proto_dists = proto_samp_styles.size(0)
				proto_samp_styles = proto_samp_styles.unsqueeze(dim=0).expand(par_1_words.size(0), -1, -1).contiguous().view(-1, proto_samp_styles.size(1))
				par_1_samp_styles = proto_samp_styles

				# Stack the different styles as different batch elements
				par_1_semantics = par_1_semantics[:,None,:].expand(-1, num_proto_dists, -1).contiguous().view(-1, par_1_semantics.size(1))
				slot_1_embeds = slot_1_embeds[:,None,:,:].expand(-1, num_proto_dists,-1,-1).contiguous().view(-1, slot_1_embeds.size(1), slot_1_embeds.size(2))
				slot_1_ids = slot_1_ids[:,None,:].expand(-1, num_proto_dists, -1).contiguous().view(-1, slot_1_ids.size(1))
				slot_1_lengths = slot_1_lengths[:,None].expand(-1, num_proto_dists).contiguous().view(-1)

				# Applies paraphrase module for text generation
				par_1_decoder_res = self.decoder_module(semantics = par_1_semantics, 
														styles = par_1_samp_styles,
														labels = None, 
														max_generation_steps = max_generation_steps,
														UNK_embeds = slot_1_embeds,
														UNK_lengths = slot_1_lengths,
														UNK_word_ids = slot_1_ids)
				# Reshape styles over batch elements to an extra dimension
				gen_outputs, gen_slot_weights, gen_preds, gen_lengths = par_1_decoder_res
				gen_outputs = gen_outputs.view(par_1_words.size(0),  num_proto_dists, gen_outputs.size(1), gen_outputs.size(2))
				gen_slot_weights = gen_slot_weights.view(par_1_words.size(0),  num_proto_dists, gen_slot_weights.size(1), gen_slot_weights.size(2))
				gen_preds = gen_preds.view(par_1_words.size(0), num_proto_dists, gen_preds.size(1))
				gen_lengths = gen_lengths.view(par_1_words.size(0), num_proto_dists)

				return gen_outputs, gen_slot_weights, gen_preds, gen_lengths, proto_dists, par_1_proto_dist

	def language_modeling(self, _input, teacher_forcing=False, teacher_forcing_ratio=1.0):
		par_words, par_lengths, contexts_words, contexts_lengths = _input

		par_words, par_word_embeds, par_lengths = self.encoder_module.prepare(par_words, par_lengths, self.embedding_module, slot_mask=None, slot_vals=None, slot_lengths=None)
		contexts_words, contexts_word_embeds, contexts_lengths = self.encoder_module.prepare(contexts_words, contexts_lengths, self.embedding_module)
		par_masks = self.embedding_module.generate_mask(par_words)
		# Note that the slots for both sentences are the same, but the order can be different. Important for later loss calculation
		slot_embeds, slot_lengths, slot_ids = EncoderRNN._gather_UNK_token_embeddings(par_word_embeds, par_masks, word_ids=par_words)

		# par_semantic, par_style = self.encoder_module(par_words, par_word_embeds, par_lengths, masks=None)

		_, contexts_style = self.encoder_module(contexts_words, contexts_word_embeds, contexts_lengths, masks=None, additional_input=contexts_word_embeds.new_zeros((contexts_word_embeds.size(0), self.encoder_module.semantic_size)))
		contexts_style, contexts_style_mu, contexts_style_sigma = contexts_style
		dropped_contexts_style = contexts_style * ((self.encoder_module.semantic_size + contexts_style.size(1)) / (1.0 * contexts_style.size(1)))

		# Check if the first label token is always start-of-sentence. If so, remove it 
		labels = par_words
		if (labels[:,0] == get_SOS_index()).byte().all():
			labels = labels[:,1:]
		# Applies paraphrase module for text generation
		par_decoder_res = self.decoder_module(semantics = dropped_contexts_style.new_zeros(size=(dropped_contexts_style.size(0), self.encoder_module.semantic_size)), 
											  styles = dropped_contexts_style,
											  labels = labels if teacher_forcing else None, 
											  teacher_forcing_ratio = teacher_forcing_ratio,
											  min_generation_steps = labels.shape[1], 
											  max_generation_steps = labels.shape[1],
											  UNK_embeds = slot_embeds,
											  UNK_lengths = slot_lengths,
											  UNK_word_ids = slot_ids)

		return par_decoder_res, \
			   (contexts_style, contexts_style_mu, contexts_style_sigma)

	def contextless_paraphrasing(self, _input, teacher_forcing=False, teacher_forcing_ratio=1.0, switch_rate=0.5):
		"""
		Forward method for overall model. If labels are given, the decoder is performing teacher forcing.
		Otherwise, we perform Beam-search to find N best paraphrases
		"""
		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths = _input

		par_1_masks = self.embedding_module.generate_mask(par_1_words)
		par_2_masks = self.embedding_module.generate_mask(par_2_words)
		par_1_words, par_1_word_embeds, par_1_lengths = self.encoder_module.prepare(par_1_words, par_1_lengths, self.embedding_module, slot_mask=par_1_masks, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths)
		par_2_words, par_2_word_embeds, par_2_lengths = self.encoder_module.prepare(par_2_words, par_2_lengths, self.embedding_module, slot_mask=par_2_masks, slot_vals=par_2_slots, slot_lengths=par_2_slot_lengths)
		# Note that the slots for both sentences are the same, but the order can be different. Important for later loss calculation
		slot_1_embeds, slot_1_lengths, slot_1_ids = EncoderRNN._gather_UNK_token_embeddings(par_1_word_embeds, par_1_masks, word_ids=par_1_words)
		slot_2_embeds, slot_2_lengths, slot_2_ids = EncoderRNN._gather_UNK_token_embeddings(par_2_word_embeds, par_2_masks, word_ids=par_2_words)

		par_1_semantics, par_1_style = self.encoder_module(par_1_words, par_1_word_embeds, par_1_lengths, masks=None)
		par_2_semantics, par_2_style = self.encoder_module(par_2_words, par_2_word_embeds, par_2_lengths, masks=None)
		par_1_style, par_1_style_mu, par_1_style_sigma = par_1_style
		par_2_style, par_2_style_mu, par_2_style_sigma = par_2_style
		
		if switch_rate > 0.0:
			switch_vec = (par_1_semantics.new_zeros(par_1_semantics.shape[0],1).uniform_() > (1 - switch_rate)).float()
			switched_sem_1 = (1 - switch_vec) * par_1_semantics + switch_vec * par_2_semantics
			switched_sem_2 = (1 - switch_vec) * par_2_semantics + switch_vec * par_1_semantics
		else:
			switched_sem_1 = par_1_semantics
			switched_sem_2 = par_2_semantics
		
		switched_sem_1 = self.semantics_dropout(switched_sem_1)
		switched_sem_2 = self.semantics_dropout(switched_sem_2)

		labels_1 = par_1_words
		labels_2 = par_2_words
		 # Check if the first label token is always start-of-sentence. If so, remove it 
		if (labels_1[:,0] == get_SOS_index()).byte().all():
			labels_1 = labels_1[:,1:]
		if (labels_2[:,0] == get_SOS_index()).byte().all():
			labels_2 = labels_2[:,1:]
		# Applies paraphrase module for text generation
		par_1_decoder_res = self.decoder_module(semantics = switched_sem_1, 
												styles = par_1_style,
												labels = labels_1 if teacher_forcing else None, 
												teacher_forcing_ratio = teacher_forcing_ratio,
												min_generation_steps = labels_1.shape[1], 
												max_generation_steps = labels_1.shape[1],
												UNK_embeds = slot_1_embeds,
												UNK_lengths = slot_1_lengths,
												UNK_word_ids = slot_1_ids)
		par_2_decoder_res = self.decoder_module(semantics = switched_sem_2, 
												styles = par_2_style,
												labels = labels_2 if teacher_forcing else None, 
												teacher_forcing_ratio = teacher_forcing_ratio,
												min_generation_steps = labels_2.shape[1], 
												max_generation_steps = labels_2.shape[1],
												UNK_embeds = slot_2_embeds,
												UNK_lengths = slot_2_lengths,
												UNK_word_ids = slot_2_ids)

		return par_1_decoder_res, \
			   par_2_decoder_res, \
			   (par_1_style, par_1_style_mu, par_1_style_sigma), \
			   (par_2_style, par_2_style_mu, par_2_style_sigma), \
			   (par_1_semantics, par_2_semantics)

	def encode_sentence(self, _input):
		par_words, par_lengths, par_slots, par_slot_lengths = _input
		par_mask = self.embedding_module.generate_mask(par_words)
		par_words, par_word_embeds, par_lengths = self.encoder_module.prepare(par_words, par_lengths, self.embedding_module, slot_mask=par_mask, slot_vals=par_slots, slot_lengths=par_slot_lengths)
		par_semantics, par_style = self.encoder_module(par_words, par_word_embeds, par_lengths, masks=None)
		return par_semantics, par_style

	def encode_sent_context(self, _input):
		par_words, par_lengths, par_slots, par_slot_lengths, context_words, context_lengths = _input
		par_mask = self.embedding_module.generate_mask(par_words)
		par_words, par_word_embeds, par_lengths = self.encoder_module.prepare(par_words, par_lengths, self.embedding_module, slot_mask=par_mask, slot_vals=par_slots, slot_lengths=par_slot_lengths)
		par_semantics, par_style = self.encoder_module(par_words, par_word_embeds, par_lengths, masks=None)
		
		context_words, context_word_embeds, context_lengths = self.encoder_module.prepare(context_words, context_lengths, self.embedding_module)
		_, context_style, _, context_style_attn = self.encoder_module(context_words, context_word_embeds, context_lengths, masks=None, additional_input=par_semantics, debug=True)
		context_style_attn, _ = context_style_attn
		context_style, _, _ = context_style

		return par_semantics, par_style[0], context_style_attn, context_style

	def generate_new_style(self, _input, style_vecs):
		par_words, par_lengths, par_slots, par_slot_lengths = _input
		par_mask = self.embedding_module.generate_mask(par_words)
		par_words, par_word_embeds, par_lengths = self.encoder_module.prepare(par_words, par_lengths, self.embedding_module, slot_mask=par_mask, slot_vals=par_slots, slot_lengths=par_slot_lengths)
		par_semantics, _ = self.encoder_module(par_words, par_word_embeds, par_lengths, masks=None)

		slot_embeds, slot_lengths, slot_ids = EncoderRNN._gather_UNK_token_embeddings(par_word_embeds, par_mask, word_ids=par_words)

		par_decoder_res = self.decoder_module(semantics = par_semantics, 
											  styles = style_vecs,
											  labels = None, 
											  teacher_forcing_ratio = 0.0,
											  max_generation_steps = 50,
											  UNK_embeds = slot_embeds,
											  UNK_lengths = slot_lengths,
											  UNK_word_ids = slot_ids)
		return par_decoder_res

	def encode_gt(self, _input):
		par_1_words, par_1_lengths, par_1_slots, par_1_slot_lengths = _input
		par_1_masks = self.embedding_module.generate_mask(par_1_words)
		par_1_words, par_1_word_embeds, par_1_lengths = self.encoder_module.prepare(par_1_words, par_1_lengths, self.embedding_module, slot_mask=par_1_masks, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths)
		_, _, par_1_attn_semantic, par_1_attn_style, par_1_proto_dist = self.encoder_module(par_1_words, par_1_word_embeds, par_1_lengths, masks=None, debug=True, debug_proto=True)
		return par_1_attn_semantic[1], par_1_attn_style[1], par_1_proto_dist
		

	def get_loss_module(self, weight=None, ignore_index=-1, reduction='mean'):
		return self.decoder_module.get_loss_module(weight=weight, ignore_index=ignore_index, reduction=reduction)


