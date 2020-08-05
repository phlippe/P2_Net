import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import math
from random import shuffle, random
import os
import sys
import time
# Disable matplotlib screen support
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from statistics import mean, median

from data import DatasetTemplate, DatasetHandler, debug_level, DATA_GLOVE, DATA_BERT, reconstruct_sentences
from model_utils import get_device, get_param_val
from unsupervised_models.model_loss import *
from vocab import get_id2word_dict, get_UNK_index, get_SOS_index
from task import TaskTemplate
from scheduler_annealing_KL import create_KLScheduler
from metrics import *
from mutils import add_if_not_none


#########################
## TASK SPECIFIC TASKS ##
#########################

class UnsupervisedTask(TaskTemplate):


	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix="", dataset_fun=None):
		super(UnsupervisedTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name="UnsupervisedTask" + name_suffix, dataset_fun=dataset_fun)
		self.loss_module = self.model.get_loss_module()
		self.switch_rate = 1.0
		self.VAE_loss_scaling = get_param_val(model_params, "VAE_loss_scaling", 1.0)
		self.cosine_loss_scaling = get_param_val(model_params, "cosine_loss_scaling", 0.0)
		self.loss_module_UNK = nn.NLLLoss(ignore_index=-1, reduction='none')
		self.summary_dict = {"loss_rec": list(), 
							 "loss_UNK": list(), 
							 "loss_VAE": list(), 
							 "loss_cosine": list(), 
							 "loss_combined": list(), 
							 "loss_UNK_precision": list(), 
							 "loss_UNK_recall": list(), 
							 "acc_UNK": list(), 
							 "style_mu": list(), 
							 "style_sigma": list(), 
							 "UNK_word_dist": list()}


	def _load_datasets(self):
		self._get_datasets_from_handler()
		self.gen_batch = self.val_dataset.get_random_batch(4, toTorch=False, label_lengths=True, noun_mask=True, mask_prob=0.0)
		self.val_dataset.reset_index()
		self.id2word = get_id2word_dict()
		self.generated_before = False


	def _get_datasets_from_handler(self):
		raise NotImplementedError


	def _get_sents_of_batch(self, batch):
		context_words, context_lengths, template_words, template_lengths, par_2_words, par_2_lengths, template_masks, context_masks, par_2_masks = batch
		max_fun = np.max if isinstance(template_lengths[DATA_GLOVE], np.ndarray) else torch.max
		if template_words[DATA_GLOVE] is not None and max_fun(template_lengths[DATA_GLOVE]) > 0:
			par_1_words = template_words
			par_1_lengths = template_lengths
			par_1_masks = template_masks
		elif context_words[DATA_GLOVE] is not None and max_fun(context_lengths[DATA_GLOVE]) > 0:
			par_1_words = context_words
			par_1_lengths = context_lengths
			par_1_masks = context_masks
		else:
			print("[!] ERROR: Template and context words are None")
			sys.exit(1)
		return par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks


	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks = self._get_sents_of_batch(self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True, label_lengths=True, noun_mask=True, mask_prob=0.2))

		current_tf_ratio = self._get_tf_ratio(iteration)
		use_tf = (random() < current_tf_ratio)

		if par_1_words is not None and par_1_lengths is not None:
			par_1_res, par_2_res, par_1_embeds, par_2_embeds = self.model((par_1_words, par_1_lengths, par_1_masks[1], par_2_words, par_2_lengths, par_2_masks[1]), teacher_forcing=use_tf, switch_rate=self.switch_rate)
			par_1_words = par_1_words[DATA_GLOVE]
			loss_1, loss_UNK_1, loss_VAE_1, acc_1, acc_UNK_1 = self._calculate_loss(par_1_res, par_1_words, par_1_masks[1], par_1_embeds)
			loss_2, loss_UNK_2, loss_VAE_2, acc_2, acc_UNK_2 = self._calculate_loss(par_2_res, par_2_words, par_2_masks[1], par_2_embeds)
			loss_UNK_recall_1, loss_UNK_precision_1 = self._calculate_UNK_loss_metrics(par_1_res[1], par_1_res[3], par_1_masks[1])
			loss_UNK_recall_2, loss_UNK_precision_2 = self._calculate_UNK_loss_metrics(par_2_res[1], par_2_res[3], par_2_masks[1])
		
			loss = (loss_1 + loss_2) / 2.0
			loss_UNK = (loss_UNK_1 + loss_UNK_2) / 2.0
			loss_UNK_recall = (loss_UNK_recall_1 + loss_UNK_recall_2) / 2.0
			loss_UNK_precision = (loss_UNK_precision_1 + loss_UNK_precision_2) / 2.0
			loss_VAE = (loss_VAE_1 + loss_VAE_2) / 2.0
			loss_cos = (1 - F.cosine_similarity(par_1_embeds[0], par_2_embeds[0], dim=-1)).mean()
			acc = (acc_1 + acc_2) / 2.0
			acc_UNK = (acc_UNK_1 + acc_UNK_2) / 2.0

		else:
			#FIX RECONSTRUCTION
			par_res, par_embeds = self.model.reconstruct((par_2_words, par_2_lengths), teacher_forcing=use_tf)
			loss, loss_UNK, loss_VAE, acc, acc_UNK = self._calculate_loss(par_res, par_2_words, par_2_masks[0], par_embeds)
			loss_cos = torch.zeros(size=(1,))
			loss_UNK_recall, loss_UNK_precision = self._calculate_UNK_loss_metrics(par_res[1], par_res[3], par_2_masks[0])

		# (loss_UNK_recall + loss_UNK_precision) / 2.0 
		final_loss = loss + loss_UNK + loss_UNK_recall + loss_UNK_precision + loss_VAE * self.VAE_loss_scaling + loss_cos * self.cosine_loss_scaling

		self.summary_dict["loss_rec"].append(loss.item())
		self.summary_dict["loss_UNK"].append(loss_UNK.item())
		self.summary_dict["loss_UNK_recall"].append(loss_UNK_recall.item())
		self.summary_dict["loss_UNK_precision"].append(loss_UNK_precision.item())
		self.summary_dict["loss_VAE"].append(loss_VAE.item())
		self.summary_dict["loss_cosine"].append(loss_cos.item())
		self.summary_dict["acc_UNK"].append(acc_UNK.item())
		self.summary_dict["loss_combined"].append(final_loss.item())
		for dict_key, hist_tensors in zip(["UNK_word_dist", "style_mu", "style_sigma"], [[par_1_res[1][:,:,0], par_2_res[1][:,:,0]], [par_1_embeds[1], par_2_embeds[1]], [par_1_embeds[2], par_2_embeds[2]]]):
			new_vals = [t.detach().cpu().contiguous().view(-1).numpy().tolist() for t in hist_tensors]
			new_vals = [e for sublist in new_vals for e in sublist]
			self.summary_dict[dict_key].append(new_vals)
			while len(self.summary_dict[dict_key]) > 10:
				del self.summary_dict[dict_key][0]

		return final_loss, acc

	def _calculate_loss(self, par_res, batch_labels, par_masks, par_embeds):
		par_word_dist, UNK_word_dist, _, _ = par_res
		par_masks = par_masks[:,1:].contiguous() # First token is SOS
		# Remove unknown word labels from the loss
		if (batch_labels[:,0] == get_SOS_index()).byte().all():
			batch_labels = batch_labels[:,1:]
		unknown_label = (batch_labels == get_UNK_index()).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label
		## Loss reconstruction
		loss = self.loss_module(par_word_dist.view(-1, par_word_dist.shape[-1]), batch_labels.view(-1))
		UNK_word_dist = torch.log(UNK_word_dist + (UNK_word_dist == 0).float())
		loss_UNK = self.loss_module_UNK(UNK_word_dist.view(-1, UNK_word_dist.shape[-1]), par_masks.view(-1))
		loss_UNK[torch.isnan(loss_UNK)] = 0
		loss_UNK = loss_UNK * (par_masks.view(-1) >= 0).float()
		loss_UNK = (1 + (par_masks.view(-1) > 0).float()*5.0) * loss_UNK
		loss_UNK = loss_UNK.sum() / (par_masks >= 0).float().sum()
		## Accuracy calculation
		_, pred_labels = torch.max(par_word_dist, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / torch.sum(batch_labels != -1).float()
		_, pred_UNK = torch.max(UNK_word_dist, dim=-1)
		acc_UNK = torch.sum((pred_UNK == par_masks) & (pred_UNK > 0)).float() / (torch.sum(par_masks > 0).float() + 1e-10)
		## Loss VAE regularization
		semantic_embed, style_mu, style_std = par_embeds
		loss_VAE = torch.mean(- torch.log(style_std) + (style_std ** 2 - 1 + style_mu ** 2) / 2)

		return loss, loss_UNK, loss_VAE, acc, acc_UNK

	def _calculate_UNK_loss_metrics(self, UNK_word_dist, par_lengths, par_masks):
		par_masks = par_masks[:,1:].contiguous() # First token is SOS
		par_masks_recall = torch.max(par_masks - 1, par_masks.new_zeros(size=par_masks.shape)-1)
		UNK_word_dist_recall = UNK_word_dist[:,:,1:].contiguous()
		loss_UNK_recall = self.loss_module_UNK(UNK_word_dist_recall.view(-1, UNK_word_dist_recall.shape[-1]), par_masks_recall.view(-1))
		# print("Par masks: %s" % str(par_masks_recall[:,:4]))
		# print("UNK word dist recall: %s" % str(UNK_word_dist_recall[:,:4,:]))
		# print("Loss UNK recall: %s" % str(loss_UNK_recall))

		UNK_labels_one_hot = UNK_word_dist.new_zeros(size=UNK_word_dist.shape)
		UNK_labels_one_hot = UNK_labels_one_hot.scatter(2, (par_masks + (par_masks == -1).long())[:,:,None], 1)
		UNK_labels_one_hot = UNK_labels_one_hot[:,:,1:]
		valid_UNKs = (UNK_labels_one_hot.sum(dim=1) > 0.0).float()

		loss_UNK_recall_manual = - torch.log(UNK_word_dist_recall * UNK_labels_one_hot + (1 - UNK_labels_one_hot)).sum() / UNK_labels_one_hot.sum()
		# print("Loss UNK recall manual: %s" % str(loss_UNK_recall_manual))
		
		PREC_THRESHOLD = 0.1
		UNK_precision_sum = (torch.max(torch.zeros_like(UNK_word_dist_recall)+PREC_THRESHOLD, UNK_word_dist_recall * (1 - UNK_labels_one_hot)) - PREC_THRESHOLD).sum(dim=1) /  par_lengths.float().view(-1, 1)
		loss_UNK_precision = - torch.log(1 / (1 + UNK_precision_sum)).sum() / valid_UNKs.sum()

		return loss_UNK_recall_manual, loss_UNK_precision

	def _eval_batch(self, batch):
		par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks = self._get_sents_of_batch(batch)
		eval_swr = (1.0 if self.switch_rate > 0.0 else 0.0)
		p1_res, p2_res, _, _ = self.model((par_1_words, par_1_lengths, par_1_masks[int(eval_swr)], par_2_words, par_2_lengths, par_2_masks[int(eval_swr)]), teacher_forcing=True, switch_rate=eval_swr)
		p1_perplexity_probs, _, _, _ = p1_res
		p2_perplexity_probs, _, _, _ = p2_res
		p1_res_tf, p2_res_tf, _, _ = self.model((par_1_words, par_1_lengths, par_1_masks[int(eval_swr)], par_2_words, par_2_lengths, par_2_masks[int(eval_swr)]), teacher_forcing=False, switch_rate=eval_swr)
		_, _, p1_generated_words, p1_generated_lengths = p1_res_tf
		_, _, p2_generated_words, p2_generated_lengths = p2_res_tf

		p1_perplexity_probs = p1_perplexity_probs.detach()
		p1_generated_words = p1_generated_words.detach()
		p1_generated_lengths = p1_generated_lengths.detach()
		p2_perplexity_probs = p2_perplexity_probs.detach()
		p2_generated_words = p2_generated_words.detach()
		p2_generated_lengths = p2_generated_lengths.detach()

		# Remove unknown word labels from the evaluation
		batch_labels = par_2_words # [DATA_GLOVE]
		if (batch_labels[:,0] == get_SOS_index()).byte().all():
			batch_labels = batch_labels[:,1:]
		unknown_label = (batch_labels == get_UNK_index()).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label

		return batch_labels, p2_perplexity_probs, p2_generated_words, p2_generated_lengths # batch_labels, perplexity_probs, generated_words, generated_lengths 


	def eval(self, dataset=None, batch_size=64):
		return super().eval(dataset=dataset, batch_size=batch_size, label_lengths=True, noun_mask=True)


	def add_summary(self, writer, iteration):
		# TODO: Add some example generations here. Either run the model again for some random sentences, or save last training sentences
		writer.add_scalar("train_%s/teacher_forcing_ratio" % (self.name), self._get_tf_ratio(iteration), iteration)
		for key, val in self.summary_dict.items():
			if not isinstance(val, list):
				writer.add_scalar("train_%s/%s" % (self.name, key), val, iteration)
				self.summary_dict[key] = 0.0
			elif len(val) == 0:
				continue
			elif not isinstance(val[0], list):
				writer.add_scalar("train_%s/%s" % (self.name, key), mean(val), iteration)
				self.summary_dict[key] = list()
			else:
				val = [v for sublist in val for v in sublist]
				writer.add_histogram("train_%s/%s" % (self.name, key), np.array(val), iteration)
				self.summary_dict[key] = list()

		if iteration % 1 == 0:
			gen_list = self.generate_examples()
			for i in range(len(gen_list)):
				if not self.generated_before:
					writer.add_text(self.name + "_gen%i_input_phrase" % (i), gen_list[i][0], iteration)
					writer.add_text(self.name + "_gen%i_input_labels" % (i), gen_list[i][1], iteration)
				writer.add_text(self.name + "_gen%i_reconstructed_phrase" % (i), gen_list[i][2], iteration)
				writer.add_text(self.name + "_gen%i_reconstructed_phrase_tf" % (i), gen_list[i][3], iteration)

			gen_list = self.generate_random_style_samples()
			for i in range(len(gen_list)):
				if not self.generated_before:
					writer.add_text(self.name + "_samp%i_input_phrase" % (i), gen_list[i][0], iteration)
				for j in range(len(gen_list[i][1])):
					writer.add_text(self.name + "_samp%i_sample_%i" % (i, j), gen_list[i][1][j], iteration)

			self.generated_before = True


	@staticmethod
	def batch_to_torch(batch):
		new_batch = []
		for b in batch:
			if isinstance(b, dict):
				new_element = dict()
				for key in b.keys():
					new_element[key] = torch.LongTensor(b[key]).to(get_device()) if b[key] is not None else None
			elif isinstance(b, list):
				new_element = [torch.LongTensor(b_e).to(get_device()) for b_e in b]
			elif isinstance(b, tuple):
				new_element = tuple([torch.LongTensor(b_e).to(get_device()) for b_e in b])
			elif b is None:
				new_element = None
			else:
				new_element = torch.LongTensor(b).to(get_device())
			new_batch.append(new_element)
		return tuple(new_batch)


	def generate_examples(self):
		self.model.eval()
		# 1.) Put data on GPU
		batch_torch = UnsupervisedTask.batch_to_torch(self.gen_batch)
		eval_swr = (1.0 if self.switch_rate > 0.0 else 0.0)
		# 2.) Push data through network
		par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks = self._get_sents_of_batch(batch_torch) # self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True, label_lengths=True)
		with torch.no_grad():
			par_1_res, par_2_res, _, _ = self.model((par_1_words, par_1_lengths, par_1_masks[int(eval_swr)], par_2_words, par_2_lengths, par_2_masks[int(eval_swr)]), teacher_forcing=False, switch_rate=eval_swr)
			par_1_res_tf, par_2_res_tf, _, _ = self.model((par_1_words, par_1_lengths, par_1_masks[int(eval_swr)], par_2_words, par_2_lengths, par_2_masks[int(eval_swr)]), teacher_forcing=True, switch_rate=eval_swr)
		del batch_torch
		# 3.) Reconstruct generated answer and input
		generated_paraphrases = list()
		generated_paraphrases_tf = list()
		input_phrases = list()
		input_labels = list()

		gen_par_1_UNKdist = par_1_res[1].cpu().numpy()
		gen_par_1_words = par_1_res[2].cpu().numpy()
		gen_par_1_lengths = par_1_res[3].cpu().numpy()
		gen_par_2_UNKdist = par_1_res[1].cpu().numpy()
		gen_par_2_words = par_2_res[2].cpu().numpy()
		gen_par_2_lengths = par_2_res[3].cpu().numpy()

		gen_par_1_words_tf = par_1_res_tf[2].cpu().numpy()
		gen_par_1_lengths_tf = par_1_res_tf[3].cpu().numpy()
		gen_par_2_words_tf = par_2_res_tf[2].cpu().numpy()
		gen_par_2_lengths_tf = par_2_res_tf[3].cpu().numpy()

		par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks = self._get_sents_of_batch(self.gen_batch)
		par_1_words, par_1_lengths = par_1_words[DATA_GLOVE], par_1_lengths[DATA_GLOVE]

		for embeds, lengths, list_to_add, UNK_dist in zip([par_1_words, par_2_words, par_1_words if eval_swr == 0.0 else par_2_words, par_2_words if eval_swr == 0.0 else par_1_words, gen_par_1_words, gen_par_2_words, gen_par_1_words_tf, gen_par_2_words_tf],
														  [par_1_lengths, par_2_lengths, par_1_lengths if eval_swr == 0.0 else par_2_lengths, par_2_lengths if eval_swr == 0.0 else par_1_lengths, gen_par_1_lengths, gen_par_2_lengths, gen_par_1_lengths_tf, gen_par_2_lengths_tf],
														  [input_phrases, input_phrases, input_labels, input_labels, generated_paraphrases, generated_paraphrases, generated_paraphrases_tf, generated_paraphrases_tf],
														  [None, None, None, None, gen_par_1_UNKdist, gen_par_2_UNKdist, None, None]):
			for batch_index in range(embeds.shape[0]):
				p_words = list()
				if len(lengths.shape) == 1:
					for word_index in range(lengths[batch_index]):
						p_words.append(self.id2word[embeds[batch_index, word_index]])
						# if UNK_dist is not None:
						# 	p_words.append("{%i,%3.1f}" % (np.argmax(UNK_dist[batch_index,word_index]), 100.0*UNK_dist[batch_index,word_index,0]))
					sents = "[%i] " % (lengths[batch_index]) + " ".join(p_words)
				else:
					lengths = np.reshape(lengths, [lengths.shape[0], -1])
					for sent_index in range(lengths.shape[1]):
						s_words = ["[%i] " % (lengths[batch_index, sent_index])]
						for word_index in range(lengths[batch_index, sent_index]):
							s_words.append(self.id2word[embeds[batch_index, sent_index, word_index]])
							# if UNK_dist is not None:
							# 	p_words.append("{%i,%3.1f}" % (np.argmax(UNK_dist[batch_index,sent_index,word_index]), 100.0*UNK_dist[batch_index,sent_index,word_index,0]))
						p_words.append(" ".join(s_words))
					sents = p_words

				list_to_add.append(sents)

		# 5.) Put everything in a nice format
		gen_list = list(zip(input_phrases, input_labels, generated_paraphrases, generated_paraphrases_tf))
		self.model.train()
		return gen_list

	def generate_random_style_samples(self):
		self.model.eval()
		# 1.) Put data on GPU
		batch_torch = UnsupervisedTask.batch_to_torch(self.gen_batch)
		_, _, _, par_words, par_lengths, par_masks = self._get_sents_of_batch(batch_torch)
		with torch.no_grad():
			_, _, gen_par_words, gen_par_lengths = self.model.sample_reconstruction_styles((par_words, par_lengths, par_masks[0]), num_samples=8)
		del batch_torch
		# 3.) Reconstruct generated answer and input
		generated_paraphrases = list()
		input_phrases = list()

		gen_par_words = gen_par_words.cpu().numpy()
		gen_par_lengths = gen_par_lengths.cpu().numpy()

		par_words = self.gen_batch[4]
		par_lengths = self.gen_batch[5]

		for embeds, lengths, list_to_add in zip([par_words, gen_par_words],
												[par_lengths, gen_par_lengths],
												[input_phrases, generated_paraphrases]):
			for batch_index in range(embeds.shape[0]):
				p_words = list()
				if len(lengths.shape) == 1:
					for word_index in range(lengths[batch_index]):
						p_words.append(self.id2word[embeds[batch_index, word_index]])
					sents = "[%i] " % (lengths[batch_index]) + " ".join(p_words)
				else:
					lengths = np.reshape(lengths, [lengths.shape[0], -1])
					for sent_index in range(lengths.shape[1]):
						s_words = ["[%i] " % (lengths[batch_index, sent_index])]
						for word_index in range(lengths[batch_index, sent_index]):
							s_words.append(self.id2word[embeds[batch_index, sent_index, word_index]])
						p_words.append(" ".join(s_words))
					sents = p_words

				list_to_add.append(sents)

		# 5.) Put everything in a nice format
		gen_list = list(zip(input_phrases, generated_paraphrases))
		self.model.train()
		return gen_list


class LanguageModelingTask(UnsupervisedTask):

	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix="", switch_rate=0.0, dataset_fun=DatasetHandler.load_LM_Dialogue_datasets):
		super(LanguageModelingTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, dataset_fun=dataset_fun)
		self.name = "LanguageModeling" + self.train_dataset.dataset_name.replace(" ","_")
		self.switch_rate = switch_rate
		self.cosine_loss_scaling = 0.0

	def _get_datasets_from_handler(self):
		self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_fun(debug_dataset=self.debug)


class DialogueModelingTask(UnsupervisedTask):

	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix="", switch_rate=0.5, dataset_fun=DatasetHandler.load_LM_Dialogue_datasets):
		super(DialogueModelingTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, dataset_fun=dataset_fun)
		self.name = "DialogueModeling" + self.train_dataset.dataset_name.replace(" ","_")
		self.switch_rate = 0.0
		self.binary_switch_rate = 0.5
		self.cosine_loss_scaling = 0.0

	def _get_datasets_from_handler(self):
		self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_fun(debug_dataset=self.debug)

	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		quest_words, quest_lengths, quest_masks, answ_words, answ_lengths, answ_masks = self._get_sents_of_batch(self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True, label_lengths=True, noun_mask=True, mask_prob=0.2))

		current_tf_ratio = self._get_tf_ratio(iteration)
		use_tf = (random() < current_tf_ratio)
		switch_factor = (random() < self.binary_switch_rate)

		if switch_factor:
			answ_res, quest_embeds = self.model.question_answer_switch((quest_words, quest_lengths, quest_masks[0], answ_words, answ_lengths, answ_masks[0]), teacher_forcing=use_tf)
			loss, loss_UNK, loss_VAE, acc, acc_UNK = self._calculate_loss(answ_res, answ_words, answ_masks[0], quest_embeds)

		else:
			quest_res, quest_embeds = self.model.reconstruct((quest_words, quest_lengths, quest_masks[0]), teacher_forcing=use_tf)
			answ_res, answ_embeds = self.model.reconstruct((answ_words, answ_lengths, answ_masks[0]), teacher_forcing=use_tf)
			quest_words = quest_words[DATA_GLOVE]
			loss_quest, loss_UNK_quest, loss_VAE_quest, acc_quest, acc_UNK_quest = self._calculate_loss(quest_res, quest_words, quest_masks[0], quest_embeds)
			loss_answ, loss_UNK_answ, loss_VAE_answ, acc_answ, acc_UNK_answ = self._calculate_loss(answ_res, answ_words, answ_masks[0], answ_embeds)
			ANSW_FAC = 2.0 / 3.0
			QUEST_FAC = 1 - ANSW_FAC
			loss = loss_quest * QUEST_FAC + loss_answ * ANSW_FAC
			loss_VAE = loss_VAE_quest * QUEST_FAC + loss_VAE_answ * ANSW_FAC
			loss_UNK = loss_UNK_quest * QUEST_FAC + loss_UNK_answ * ANSW_FAC
			acc = acc_quest * QUEST_FAC + acc_answ * ANSW_FAC
			acc_UNK = acc_UNK_quest * QUEST_FAC + acc_UNK_answ * ANSW_FAC
			
		final_loss = loss + loss_UNK + loss_VAE * self.VAE_loss_scaling

		self.summary_dict["loss_rec"].append(loss.item())
		self.summary_dict["loss_UNK"].append(loss_UNK.item())
		self.summary_dict["loss_VAE"].append(loss_VAE.item())
		self.summary_dict["acc_UNK"].append(acc_UNK.item())
		self.summary_dict["loss_combined"].append(final_loss.item())
		for dict_key, hist_tensors in zip(["UNK_word_dist", "style_mu", "style_sigma"], [[answ_res[1][:,:,0]], [quest_embeds[1]], [quest_embeds[2]]]):
			new_vals = [t.detach().cpu().contiguous().view(-1).numpy().tolist() for t in hist_tensors]
			new_vals = [e for sublist in new_vals for e in sublist]
			self.summary_dict[dict_key].append(new_vals)
			while len(self.summary_dict[dict_key]) > 10:
				del self.summary_dict[dict_key][0]

		return final_loss, acc


class ParaphraseTask(UnsupervisedTask):

	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix="", switch_rate=0.8, dataset_fun=DatasetHandler.load_Microsoft_Video_Description_datasets):
		super(ParaphraseTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, dataset_fun=dataset_fun)
		self.name = "ParaphraseTask" + self.train_dataset.dataset_name.replace(" ","_")
		self.switch_rate = switch_rate

	def _get_datasets_from_handler(self):
		self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_fun(debug_dataset=self.debug)


class PretrainingTask(TaskTemplate):

	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix=""):
		super(PretrainingTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name="PretrainingTask" + name_suffix)
		if not debug:
			self.tasks = [
					DialogueModelingTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_LM_Dialogue_datasets),
					ParaphraseTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_Microsoft_Video_Description_datasets, switch_rate=0.75),
					ParaphraseTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_Quora_Paraphrase_datasets, switch_rate=0.6),
					ParaphraseTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_Wikipedia_Paraphrase_datasets, switch_rate=0.75)
				]
			self.task_frequency = [
					0.2,
					0.1,
					0.4,
					0.3
				]
		else:
			self.tasks = [
				ParaphraseTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_Wikipedia_Paraphrase_datasets, switch_rate=0.75)
				# DialogueModelingTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_LM_Dialogue_datasets)
			]
			self.task_frequency = [
				1.0
			]
		self._prepare_training()

	def _prepare_training(self):
		self.train_index = 0
		self.train_permut_order = []
		for task_index, p in enumerate(self.task_frequency):
			self.train_permut_order += [task_index]*int(p * 100)
		shuffle(self.train_permut_order)

	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		task_index = self.train_permut_order[self.train_index]
		self.train_index += 1
		if self.train_index >= len(self.train_permut_order):
			self.train_index = 0
			shuffle(self.train_permut_order)
		return self.tasks[task_index].train_step(batch_size=batch_size, loop_dataset=loop_dataset, iteration=iteration)

	def eval(self, dataset=None, batch_size=64):
		avg_acc = 0
		detailed_metrics = {}
		for t_ind, t in enumerate(self.tasks):
			acc_t, detailed_t = t.eval(dataset=dataset, batch_size=batch_size)
			detailed_metrics[t.name] = detailed_t 
			avg_acc += self.task_frequency[t_ind] * acc_t
		return avg_acc, detailed_metrics

	def add_summary(self, writer, iteration):
		for t in self.tasks:
			t.add_summary(writer, iteration)



class ContextAwarePretrainingTask(TaskTemplate):

	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix=""):
		super(ContextAwarePretrainingTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name="PretrainingTask" + name_suffix)
		
		self.tasks = [
				[
					ContextAwareParaphrasingTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_Quora_Paraphrase_datasets)
				],
				[	
					ContextAwareParaphrasingTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_Quora_Paraphrase_datasets),
					ContextAwareLanguageModelingTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_ContextLM_Book_datasets),
					ContextAwareDialogueTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_Dialogue_Paraphrase_datasets)
				],
				[
					ContextAwareDialogueTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_Dialogue_Paraphrase_datasets),
					ContextAwareDialogueTask(model, model_params, load_data=load_data, debug=debug, dataset_fun=DatasetHandler.load_Dialogue_Paraphrase_Small_datasets, name_suffix="Small")
				]
			]
		freq_second_task = get_param_val(model_params, "pretraining_second_task", 0.15)
		self.task_frequency = [
				[
					1.0
				],
				[
					0.3, # 0.6
					0.2, # 0.4
					0.5 # 0.0
				],
				[
					1.0 - freq_second_task,
					freq_second_task
				]
			]
		self.training_iterations = [
			0 if not get_param_val(model_params, "only_paraphrasing", False) else 100000,
			get_param_val(model_params, "pretraining_iterations", 15000)
		]
		self.training_stage = -1
		
		assert len(self.tasks) == len(self.task_frequency), "[!] ERROR: Both the tasks and the frequency need to be defined for the same number of training stages."
		assert all([len(self.tasks[i]) == len(self.task_frequency[i]) for i in range(len(self.tasks))]), "[!] ERROR: For each training stage, one frequency needs to be defined per task."
		assert len(self.training_iterations) == len(self.task_frequency) - 1, "[!] ERROR: A duration (number of training iterations) needs to be specified for every training stage except the last one."

		self._switch_training_stages()

	def _switch_training_stages(self):
		self.training_stage += 1
		print("-"*75)
		print("Switch training stage to %i" % (self.training_stage))
		print("-"*75)
		self.train_index = 0
		self.train_permut_order = []
		for task_index, p in enumerate(self.task_frequency[self.training_stage]):
			self.train_permut_order += [task_index]*int(p * 100)
		shuffle(self.train_permut_order)

	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		while self.training_stage < len(self.training_iterations) and \
			  iteration >= self.training_iterations[self.training_stage]:
			self._switch_training_stages()

		stage_iteration = iteration - (self.training_iterations[self.training_stage-1] if self.training_stage > 0 else 0)
		task_index = self.train_permut_order[self.train_index]
		self.train_index += 1
		if self.train_index >= len(self.train_permut_order):
			self.train_index = 0
			shuffle(self.train_permut_order)
		return self.tasks[self.training_stage][task_index].train_step(batch_size=batch_size, loop_dataset=loop_dataset, iteration=stage_iteration)

	def eval(self, dataset=None, batch_size=64):
		avg_acc = 0
		detailed_metrics = {}
		if len(self.tasks[self.training_stage]) > 1:
			for t_ind, t in enumerate(self.tasks[self.training_stage]):
				acc_t, detailed_t = t.eval(dataset=dataset, batch_size=batch_size)
				detailed_metrics[t.name] = detailed_t 
				if self.task_frequency[self.training_stage][t_ind] == max(self.task_frequency[self.training_stage]):
					avg_acc = acc_t
			return avg_acc, detailed_metrics
		else:
			return self.tasks[self.training_stage][0].eval(dataset=dataset, batch_size=batch_size)

	def add_summary(self, writer, iteration):
		for t in self.tasks[self.training_stage]:
			t.add_summary(writer, iteration)

	def finalize_summary(self, writer, iteration, checkpoint_path):
		for train_stage_tasks in self.tasks:
			for t in train_stage_tasks:
				t.finalize_summary(writer, iteration, checkpoint_path)

	def export_best_results(self, checkpoint_path, iteration):
		for t in self.tasks[self.training_stage]:
			t.export_best_results(checkpoint_path, iteration)



class ContextAwareDialogueTask(TaskTemplate):


	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix="", dataset_fun=DatasetHandler.load_Dialogue_Paraphrase_datasets):
		super(ContextAwareDialogueTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name="ContextAwareDialogueParaphrase" + name_suffix, dataset_fun=dataset_fun)
		self.loss_module = self.model.get_loss_module()
		self.switch_rate = get_param_val(model_params, "switch_rate", 0.8)
		self.semantic_full_dropout = get_param_val(model_params, "semantic_full_dropout", 0.0)
		self.semantic_full_dropout_eval = 0.0 if self.semantic_full_dropout < 1.0 else 1.0
		self.KL_scheduler = create_KLScheduler(scheduler_type = get_param_val(model_params, "VAE_scheduler", 1),
											   annealing_func_type = get_param_val(model_params, "VAE_annealing_func", 0),
											   loss_scaling = get_param_val(model_params, "VAE_loss_scaling", 1.0),
											   num_iters = get_param_val(model_params, "VAE_annealing_iters", 10000))
		self.cosine_loss_scaling = get_param_val(model_params, "cosine_loss_scaling", 0.0)
		self.cosine_counter_loss = get_param_val(model_params, "cosine_counter_loss", False)
		self.style_loss_scaling = get_param_val(model_params, "style_loss_scaling", 1.0)
		self.pure_style_loss = get_param_val(model_params, "pure_style_loss", False)
		self.use_semantic_specific_attn = get_param_val(model_params, "use_semantic_specific_attn", False)
		self.loss_module_slots = nn.NLLLoss(ignore_index=-1, reduction='none')
		self.eval_counter = 0
		# self.loss_module_style = LossStyleModule(style_size = get_param_val(model_params, "style_size", allow_default=False, error_location="ContextAwareDialogueTask - model_params"),
		# 										 response_style_size = get_param_val(model_params, "response_style_size", -1))
		# self.loss_module_style = LossStyleSimilarityModule(style_size = get_param_val(model_params, "style_size", allow_default=False, error_location="ContextAwareDialogueTask - model_params"))
		if get_param_val(model_params, "style_loss_module") == 0:
			self.loss_module_style = LossStylePrototypeSimilarityModule(style_size = get_param_val(model_params, "response_style_size", allow_default=False, error_location="ContextAwareDialogueTask - model_params"),
																		stop_grads = get_param_val(model_params, "style_loss_stop_grads", False))
		else:
			self.loss_module_style = LossStylePrototypeDistModule(style_size = get_param_val(model_params, "response_style_size", allow_default=False, error_location="ContextAwareDialogueTask - model_params"),
																  stop_grads = get_param_val(model_params, "style_loss_stop_grads", False))
		self.style_loss_scheduler = create_KLScheduler(scheduler_type=1, annealing_func_type=1, loss_scaling=self.style_loss_scaling,
													   num_iters=get_param_val(model_params, "style_loss_annealing_iters", -1))
		self.summary_dict = {"loss_rec": list(), 
							 "loss_slots": list(), 
							 "loss_VAE": list(), 
							 "loss_style": list(),
							 "KL_scheduler": list(),
							 "style_loss_scheduler": list(),
							 "loss_cosine": list(), 
							 "loss_cosine_to_others": list(),
							 "euclidean_dist": list(),
							 "euclidean_dist_to_others": list(),
							 "loss_combined": list(), 
							 "acc_slots": list(), 
							 "acc_style": list(),
							 "context_style_mu": list(), 
							 "context_style_sigma": list(), 
							 "par_style_mu": list(), 
							 "par_style_sigma": list(), 
							 "slots_word_dist": list()}
		for proto_index in range(self.model.encoder_module.num_prototypes):
			self.summary_dict["proto_%i_attn" % proto_index] = list()
			self.summary_dict["proto_%i_context_attn" % proto_index] = list()


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_fun(debug_dataset=self.debug, num_context_turns=get_param_val(self.model_params, "num_context_turns", 2))
		self.gen_batch = self.val_dataset.get_random_batch(16 if not self.debug else 2, toTorch=False, label_lengths=True, noun_mask=True, mask_prob=0.0)
		self.val_dataset.reset_index()
		self.id2word = get_id2word_dict()
		self.generated_before = False


	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)

		current_tf_ratio = self._get_tf_ratio(iteration)
		use_tf = (random() < current_tf_ratio)

		par_1_masks = self.model.embedding_module.generate_mask(par_1_words)
		par_2_masks = self.model.embedding_module.generate_mask(par_2_words)
		par_1_res, par_2_res, context_1_style, context_2_style, par_1_style, par_2_style, par_semantics, _, _, proto_dists = self.model(_input = (par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths), 
																																	    teacher_forcing = True,
																																	    teacher_forcing_ratio = current_tf_ratio, 
																																	    switch_rate = self.switch_rate,
																																	    semantic_full_dropout = self.semantic_full_dropout,
																																	    additional_supervision = False,
																																	    use_semantic_specific_attn = self.use_semantic_specific_attn,
																																	    ignore_context = self.model.encoder_module.use_prototype_styles and (self.style_loss_scaling == 0))
		loss_1, loss_slots_1, loss_VAE_1, acc_1, acc_slots_1 = self._calculate_loss(par_1_res, par_1_words, par_1_masks, context_1_style, par_1_style)
		loss_2, loss_slots_2, loss_VAE_2, acc_2, acc_slots_2 = self._calculate_loss(par_2_res, par_2_words, par_2_masks, context_2_style, par_2_style)
		loss_style, acc_style = self.loss_module_style(context_1_style, context_2_style, par_1_style, par_2_style, proto_dists)

		loss = (loss_1 + loss_2) / 2.0
		loss_slots = (loss_slots_1 + loss_slots_2) / 2.0
		loss_VAE = (loss_VAE_1 + loss_VAE_2) / 2.0
		loss_cosine_to_others = (F.cosine_similarity(par_semantics[0].unsqueeze(dim=0), par_semantics[1].unsqueeze(dim=1), dim=-1)) * (1 - torch.eye(par_semantics[0].size(0), device=par_semantics[0].device))
		loss_cosine_to_others = loss_cosine_to_others.sum() / (1.0 * par_semantics[0].size(0) * (par_semantics[0].size(0)-1))
		loss_cos = (1 - F.cosine_similarity(par_semantics[0], par_semantics[1], dim=-1)).mean()
		
		euclidean_dist_to_others = euclidean_distance(par_semantics[0].unsqueeze(dim=0), par_semantics[1].unsqueeze(dim=1)) * (1 - torch.eye(par_semantics[0].size(0), device=par_semantics[0].device))
		euclidean_dist_to_others = euclidean_dist_to_others.sum() / (1.0 * par_semantics[0].size(0) * (par_semantics[0].size(0)-1))
		euclidean_dist = euclidean_distance(par_semantics[0], par_semantics[1]).mean()

		acc = (acc_1 + acc_2) / 2.0
		acc_slots = (acc_slots_1 + acc_slots_2) / 2.0

		final_loss = loss + \
					 loss_slots + \
					 loss_VAE * self.KL_scheduler.get(iteration) + \
					 loss_cos * self.cosine_loss_scaling + \
					 ((loss_cosine_to_others * self.cosine_loss_scaling / 2.0) if self.cosine_counter_loss else 0.0) + \
					 loss_style * self.style_loss_scheduler.get(iteration)

		if self.pure_style_loss:
			final_loss = loss_style

		self.summary_dict["loss_rec"].append(loss.item())
		self.summary_dict["loss_slots"].append(loss_slots.item())
		self.summary_dict["loss_VAE"].append(loss_VAE.item())
		self.summary_dict["loss_style"].append(loss_style.item())
		self.summary_dict["KL_scheduler"] = [self.KL_scheduler.get(iteration)]
		self.summary_dict["style_loss_scheduler"] = [self.style_loss_scheduler.get(iteration)]
		self.summary_dict["loss_cosine"].append(loss_cos.item())
		self.summary_dict["loss_cosine_to_others"].append(loss_cosine_to_others.item())
		self.summary_dict["euclidean_dist"].append(euclidean_dist.item())
		self.summary_dict["euclidean_dist_to_others"].append(euclidean_dist_to_others.item())
		self.summary_dict["acc_slots"].append(acc_slots.item())
		self.summary_dict["acc_style"].append(acc_style.item())
		self.summary_dict["loss_combined"].append(final_loss.item())

		hist_summary_values = {
			"slots_word_dist": ([par_1_res[1][:,:,0], par_2_res[1][:,:,0]], None, 10),
			"context_style_mu": ([context_1_style[1], context_2_style[1]], 10, 10),
			"context_style_sigma": ([context_1_style[2], context_2_style[2]], 2, 10),
			"par_style_mu": ([par_1_style[1], par_2_style[1]], 10, 10),
			"par_style_sigma": ([par_1_style[2], par_2_style[2]], 2, 10)
		}
		if self.model.encoder_module.use_prototype_styles:
			for proto_index in range(proto_dists[0].size(1)):
				hist_summary_values["proto_%i_attn" % proto_index] = ([proto_dists[0][:,proto_index], proto_dists[1][:,proto_index]], None, 50)
				if not self.model.encoder_module.no_prototypes_for_context:
					hist_summary_values["proto_%i_context_attn" % proto_index] = ([proto_dists[2][:,proto_index], proto_dists[3][:,proto_index]], None, 50)
		
		for dict_key, dict_vals in hist_summary_values.items():
			hist_tensors, max_val, max_list_len = dict_vals
			new_vals = [t.detach().cpu().contiguous().view(-1).numpy().tolist() for t in hist_tensors if t is not None]
			new_vals = [e for sublist in new_vals for e in sublist]
			if max_val is not None:
				new_vals = [max(min(e,max_val),-max_val) for e in new_vals]
			self.summary_dict[dict_key].append(new_vals)
			while len(self.summary_dict[dict_key]) > max_list_len:
				del self.summary_dict[dict_key][0]

		return final_loss, acc

	def _calculate_loss(self, par_res, batch_labels, par_masks, context_style, par_style=None):
		par_word_dist, slot_dist, _, _ = par_res
		# Remove unknown word labels from the loss
		if (batch_labels[:,0] == get_SOS_index()).byte().all():
			batch_labels = batch_labels[:,1:]
			par_masks = par_masks[:,1:].contiguous() # First token is SOS
		else:
			print("[#] WARNING: Batch labels were not shortend. First token ids: \n%s \nSOS index: %i" % (str(batch_labels[:,0]), get_SOS_index()))
		unknown_label = ((batch_labels == get_UNK_index()) | (batch_labels < 0)).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label

		if par_word_dist.size(0) > batch_labels.size(0) and (par_word_dist.size(0) % batch_labels.size(0) == 0):
			extend_factor = int(par_word_dist.size(0) / batch_labels.size(0))
			batch_labels = batch_labels.repeat(extend_factor, 1)
			# print("[I] INFO: Extending batch labels by factor %i" % (extend_factor))
		if slot_dist.size(0) > par_masks.size(0) and (slot_dist.size(0) % par_masks.size(0) == 0):
			extend_factor = int(slot_dist.size(0) / par_masks.size(0))
			par_masks = par_masks.repeat(extend_factor, 1)
			# print("[I] INFO: Extending paraphrase masks by factor %i" % (extend_factor))

		## Loss reconstruction
		loss = self.loss_module(par_word_dist.view(-1, par_word_dist.shape[-1]), batch_labels.view(-1))
		slot_dist = torch.log(slot_dist + (slot_dist == 0).float())
		loss_slots = self.loss_module_slots(slot_dist.view(-1, slot_dist.shape[-1]), par_masks.view(-1))
		loss_slots[torch.isnan(loss_slots)] = 0
		loss_slots = loss_slots * (par_masks.view(-1) >= 0).float()
		loss_slots = loss_slots.sum() / (par_masks >= 0).float().sum()
		## Accuracy calculation
		_, pred_labels = torch.max(par_word_dist, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / torch.sum(batch_labels != -1).float()
		_, pred_slots = torch.max(slot_dist, dim=-1)
		acc_slots = torch.sum((pred_slots == par_masks) & (pred_slots > 0)).float() / (torch.sum(par_masks > 0).float() + 1e-10)
		## Loss VAE regularization
		_, style_mu, style_std = context_style
		loss_VAE = ContextAwareDialogueTask._calc_loss_VAE(style_mu, style_std)
		if par_style is not None:
			_, par_style_mu, par_style_std = par_style
			if par_style_mu is not None and par_style_std is not None:
				loss_VAE = loss_VAE / 2.0 + ContextAwareDialogueTask._calc_loss_VAE(par_style_mu, par_style_std) / 2.0

		return loss, loss_slots, loss_VAE, acc, acc_slots


	@staticmethod
	def _calc_loss_VAE(mu, std):
		if mu is None or std is None:
			return torch.tensor([0.0], dtype=torch.float32).to(get_device())
		return torch.mean(- torch.log(std) + (std ** 2 - 1 + mu ** 2) / 2)


	def _eval_batch(self, batch, use_context_style=False, perform_beamsearch=False):
		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = batch
		par_1_masks = self.model.embedding_module.generate_mask(par_1_words)
		par_2_masks = self.model.embedding_module.generate_mask(par_2_words)
		eval_swr = (1.0 if self.switch_rate > 0.0 else 0.0)
		p1_res, p2_res, context_1_style, context_2_style, par_1_style, par_2_style, _, _, _, proto_dists = self.model((par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths), 
																										  			  teacher_forcing=True, switch_rate=eval_swr, semantic_full_dropout=self.semantic_full_dropout_eval, use_semantic_specific_attn = self.use_semantic_specific_attn, use_context_style=use_context_style, 
																										  			  ignore_context = self.model.encoder_module.use_prototype_styles and (self.style_loss_scaling == 0))
		p1_perplexity_probs, _, _, _ = p1_res
		p2_perplexity_probs, _, _, _ = p2_res
		p1_perplexity_probs = p1_perplexity_probs.detach()
		p2_perplexity_probs = p2_perplexity_probs.detach()
		del p1_res 
		del p2_res

		_, acc_style = self.loss_module_style(context_1_style, context_2_style, par_1_style, par_2_style, proto_dists)
		acc_style = acc_style.detach()

		p1_res_tf, p2_res_tf, _, _, _, _, _, _, _, _ = self.model((par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths), 
																  teacher_forcing=False, switch_rate=eval_swr, semantic_full_dropout=self.semantic_full_dropout_eval, max_generation_steps=50, use_semantic_specific_attn = self.use_semantic_specific_attn, use_context_style=use_context_style, 
																  ignore_context = self.model.encoder_module.use_prototype_styles and (self.style_loss_scaling == 0))
		_, _, p1_generated_words, p1_generated_lengths = p1_res_tf
		_, _, p2_generated_words, p2_generated_lengths = p2_res_tf
		p1_generated_words = p1_generated_words.detach()
		p1_generated_lengths = p1_generated_lengths.detach()
		p2_generated_words = p2_generated_words.detach()
		p2_generated_lengths = p2_generated_lengths.detach()
		del p1_res_tf
		del p2_res_tf

		if perform_beamsearch:
			start_time = time.time()
			p1_res_beam, _, _, _, _, _, _, _, _, _ = self.model((par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths), 
																	      teacher_forcing=False, beams=8, switch_rate=eval_swr, semantic_full_dropout=self.semantic_full_dropout_eval, max_generation_steps=40, use_semantic_specific_attn = self.use_semantic_specific_attn, use_context_style=use_context_style, 
																	      ignore_context = self.model.encoder_module.use_prototype_styles and (self.style_loss_scaling == 0), only_par_1=True)
			_, _, p1_generated_words_beam, p1_generated_lengths_beam = p1_res_beam
			p1_generated_words_beam = p1_generated_words_beam.detach()
			p1_generated_lengths_beam = p1_generated_lengths_beam.detach()
			del p1_res_beam
			print("Completed beam search in %ss" % str(time.time() - start_time))
		else:
			p1_generated_words_beam, p1_generated_lengths_beam = None, None

		subbatches = 4
		subbatch_size = int(par_1_words.size(0) / subbatches)
		all_generated_words_styles, all_generated_lengths_styles = {"sample_all": [], "sample_gt": [], "sample_context": []}, {"sample_all": [], "sample_gt": [], "sample_context": []} 
		for subbatch_index in range(subbatches):
			bstart, bend = int(subbatch_index*subbatch_size), int((subbatch_index+1)*subbatch_size)
			for samp_cont, samp_gt, name in zip([True, True, False], [True, False, True], ["sample_all", "sample_context", "sample_gt"]):
				_, _, p1_generated_words_styles, p1_generated_lengths_styles = self.model.sample_reconstruction_styles((par_1_words[bstart:bend], par_1_lengths[bstart:bend], par_1_masks[bstart:bend], par_1_slots[bstart:bend], par_1_slot_lengths[bstart:bend], contexts_1_words[bstart:bend], contexts_1_lengths[bstart:bend]), 
																														num_samples=8, max_generation_steps=50, sample_gt=samp_gt, sample_context=samp_cont)
				p1_generated_words_styles = p1_generated_words_styles.detach()
				p1_generated_lengths_styles = p1_generated_lengths_styles.detach()
				all_generated_words_styles[name].append(p1_generated_words_styles)
				all_generated_lengths_styles[name].append(p1_generated_lengths_styles)
		p1_generated_lengths_styles = {k: torch.cat(val, dim=0) for k, val in all_generated_lengths_styles.items()}
		max_gen_len = {k: max([t.size(2) for t in val]) for k, val in all_generated_words_styles.items()}
		all_generated_words_styles = {k: [t if t.size(2) >= max_gen_len[k] else torch.cat([t, t.new_zeros(t.size(0), t.size(1), max_gen_len[k]-t.size(2))], dim=2) for t in val] for k, val in all_generated_words_styles.items()}
		p1_generated_words_styles = {k: torch.cat(val, dim=0) for k, val in all_generated_words_styles.items()}

		# Remove unknown word labels from the evaluation
		batch_labels = par_1_words
		if (batch_labels[:,0] == get_SOS_index()).byte().all():
			batch_labels = batch_labels[:,1:]
		unknown_label = ((batch_labels == get_UNK_index()) | (batch_labels == -1)).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label

		return batch_labels, p1_perplexity_probs, p1_generated_words, p1_generated_lengths, p1_generated_words_styles, p1_generated_lengths_styles, p1_generated_words_beam, p1_generated_lengths_beam, acc_style


	def eval(self, dataset=None, batch_size=64, label_lengths=False, noun_mask=False):
		# Default: if no dataset is specified, we use validation dataset
		self.eval_counter += 1
		if dataset is None:
			assert self.val_dataset is not None, "[!] ERROR: Validation dataset not loaded. Please load the dataset beforehand for evaluation."
			dataset = self.val_dataset

		self.model.eval()
		if not self.debug:
			batch_size = 128
		
		# Prepare metrics
		number_batches = int(math.ceil(dataset.get_num_examples() * 1.0 / batch_size))
		if self.debug:
			number_batches = min(8, number_batches)
		perplexity, perplexity_context = [], []
		acc_style = 0
		hypotheses, references = None, None
		hypotheses_context, references_context = None, None
		hypotheses_styles, hypotheses_styles_gt, hypotheses_styles_context, hypotheses_beams = None, None, None, None

		# Evaluation loop
		for batch_ind in range(number_batches):
			if debug_level() == 0:
				print("Evaluation process: %4.2f%% (batch %i of %i)" % (100.0 * batch_ind / number_batches, batch_ind+1, number_batches), end="\r")
			# Evaluate single batch
			with torch.no_grad():
				batch = dataset.get_batch(batch_size, loop_dataset=False, toTorch=True, label_lengths=label_lengths, noun_mask=noun_mask, mask_prob=0.0)
				batch_labels, perplexity_logits, generated_words, generated_lengths, generated_words_styles, generated_lengths_styles, generated_words_beams, generated_lengths_beams, loc_acc_style = self._eval_batch(batch, perform_beamsearch=False and (batch_ind < 4) and ((self.eval_counter % 5) == 1))
				if True or not (self.model.encoder_module.use_prototype_styles and (self.style_loss_scaling == 0)):
					_, perplexity_logits_context, generated_words_context, generated_lengths_context, _, _, _, _, _ = self._eval_batch(batch, use_context_style=True, perform_beamsearch=False)
				else:
					perplexity_logits_context = perplexity_logits
					generated_words_context = generated_words
					generated_lengths_context = generated_lengths
			# Perplexity calculation
			perplexity += TaskTemplate._eval_preplexity(perplexity_logits, batch_labels).cpu().numpy().tolist()
			perplexity_context += TaskTemplate._eval_preplexity(perplexity_logits_context, batch_labels).cpu().numpy().tolist()
			acc_style += loc_acc_style.item()

			hypotheses, references = add_if_not_none(TaskTemplate._preds_to_sents(batch_labels, generated_words, generated_lengths), (hypotheses, references))
			hypotheses_context, references_context = add_if_not_none(TaskTemplate._preds_to_sents(batch_labels, generated_words_context, generated_lengths_context), (hypotheses_context, references_context))
			hypotheses_styles = add_if_not_none(TaskTemplate._reconst_sents(generated_words_styles["sample_all"], generated_lengths_styles["sample_all"]), hypotheses_styles)
			hypotheses_styles_gt = add_if_not_none(TaskTemplate._reconst_sents(generated_words_styles["sample_gt"], generated_lengths_styles["sample_gt"]), hypotheses_styles_gt)
			hypotheses_styles_context = add_if_not_none(TaskTemplate._reconst_sents(generated_words_styles["sample_context"], generated_lengths_styles["sample_context"]), hypotheses_styles_context)
			if generated_words_beams is not None:
				hypotheses_beams = add_if_not_none(TaskTemplate._reconst_sents(generated_words_beams, generated_lengths_beams), hypotheses_beams)
			
		BLEU_score, prec_per_ngram = get_BLEU_score(hypotheses, references)
		BLEU_score_context, prec_per_ngram_context = get_BLEU_score(hypotheses_context, references_context)
		ROUGE_score = get_ROUGE_score(hypotheses, references)
		ROUGE_score_context = get_ROUGE_score(hypotheses_context, references_context)
		# Metric output
		avg_perplexity = sum(perplexity) / len(perplexity)
		avg_perplexity_context = sum(perplexity_context) / len(perplexity_context)
		median_perplexity = median(perplexity)
		unigram_variety, unigram_entropy = get_diversity_measure(hypotheses, n_gram=1)
		bigram_variety, bigram_entropy = get_diversity_measure(hypotheses, n_gram=2)
		unigram_variety_context, unigram_entropy_context = get_diversity_measure(hypotheses_context, n_gram=1)
		bigram_variety_context, bigram_entropy_context = get_diversity_measure(hypotheses_context, n_gram=2)
		unigram_variety_gt, unigram_entropy_gt = get_diversity_measure(references, n_gram=1)
		bigram_variety_gt, bigram_entropy_gt = get_diversity_measure(references, n_gram=2)
		unigram_variety_style, unigram_entropy_style = get_diversity_measure(hypotheses_styles, n_gram=1)
		bigram_variety_style, bigram_entropy_style = get_diversity_measure(hypotheses_styles, n_gram=2)
		unigram_variety_style_gt, unigram_entropy_style_gt = get_diversity_measure(hypotheses_styles_gt, n_gram=1)
		bigram_variety_style_gt, bigram_entropy_style_gt = get_diversity_measure(hypotheses_styles_gt, n_gram=2)
		unigram_variety_style_context, unigram_entropy_style_context = get_diversity_measure(hypotheses_styles_context, n_gram=1)
		bigram_variety_style_context, bigram_entropy_style_context = get_diversity_measure(hypotheses_styles_context, n_gram=2)
		if hypotheses_beams is None:
			unigram_variety_beams, unigram_entropy_beams, bigram_variety_beams, bigram_entropy_beams = 0.0, 0.0, 0.0, 0.0
		else:
			unigram_variety_beams, unigram_entropy_beams = get_diversity_measure(hypotheses_beams, n_gram=1)
			bigram_variety_beams, bigram_entropy_beams = get_diversity_measure(hypotheses_beams, n_gram=2)
		acc_style = acc_style / number_batches

		if self.semantic_full_dropout_eval == 1.0 and self.model.style_full_dropout == 1.0:
			# assert avg_perplexity == avg_perplexity_context, "[!] ERROR: Context perplexity is different from normal: %f vs %f" % (avg_perplexity, avg_perplexity_context)
			assert BLEU_score == BLEU_score_context, "[!] ERROR: BLEU scores with/without context is different although full dropout is applied: %f vs %f" % (BLEU_score, BLEU_score_context)
			assert all([r == r_c for r, r_c in zip(references, references_context)]), "[!] ERROR: References do not match"
			for p, p_c in zip(hypotheses, hypotheses_context):
				if p != p_c:
					print("-"*50+"\nPredictions do not fit.\nPrediction 1: %s\nPrediction 2: %s\n" % (str(p), str(p_c))+"-"*50)
			assert all([p == p_c for p, p_c in zip(hypotheses, hypotheses_context)]), "[!] ERROR: Hypotheses/predictions do not match"
			assert bigram_entropy_context == bigram_entropy, "[!] ERROR: Entropy for bigram differ"

		detailed_metrics = {
			"perplexity": avg_perplexity,
			"perplexity_context": avg_perplexity_context,
			"perplexity_median": median_perplexity,
			"diversity_unigram_entropy": unigram_entropy,
			"diversity_bigram_entropy": bigram_entropy,
			"diversity_unigram": unigram_variety,
			"diversity_bigram": bigram_variety,
			"diversity_unigram_entropy_context": unigram_entropy_context,
			"diversity_bigram_entropy_context": bigram_entropy_context,
			"diversity_unigram_context": unigram_variety_context,
			"diversity_bigram_context": bigram_variety_context,
			"diversity_unigram_entropy_gt": unigram_entropy_gt,
			"diversity_bigram_entropy_gt": bigram_entropy_gt,
			"diversity_unigram_gt": unigram_variety_gt,
			"diversity_bigram_gt": bigram_variety_gt,
			"diversity_unigram_style": unigram_variety_style,
			"diversity_bigram_style": bigram_variety_style,
			"diversity_unigram_entropy_style": unigram_entropy_style,
			"diversity_bigram_entropy_style": bigram_entropy_style,
			"diversity_unigram_style_gt": unigram_variety_style_gt,
			"diversity_bigram_style_gt": bigram_variety_style_gt,
			"diversity_unigram_entropy_style_gt": unigram_entropy_style_gt,
			"diversity_bigram_entropy_style_gt": bigram_entropy_style_gt,
			"diversity_unigram_style_context": unigram_variety_style_context,
			"diversity_bigram_style_context": bigram_variety_style_context,
			"diversity_unigram_entropy_style_context": unigram_entropy_style_context,
			"diversity_bigram_entropy_style_context": bigram_entropy_style_context,
			"diversity_unigram_beams": unigram_variety_beams,
			"diversity_bigram_beams": bigram_variety_beams,
			"diversity_unigram_entropy_beams": unigram_entropy_beams,
			"diversity_bigram_entropy_beams": bigram_entropy_beams,
			"BLEU": BLEU_score,
			"BLEU_context": BLEU_score_context,
			"acc_style": acc_style
		} 
		for n in range(len(prec_per_ngram)):
			detailed_metrics["BLEU_%i-gram" % (n+1)] = float(prec_per_ngram[n])
		for metric, results in ROUGE_score.items():
			if metric[-1] in ["1", "2", "3", "4"]:
				continue
			for sub_category, val in results.items():
				detailed_metrics[metric + "_" + sub_category] = val 

		self.model.train()
		dataset.reset_index()
		
		return BLEU_score_context, detailed_metrics


	def add_summary(self, writer, iteration):
		# TODO: Add some example generations here. Either run the model again for some random sentences, or save last training sentences
		writer.add_scalar("train_%s/teacher_forcing_ratio" % (self.name), self._get_tf_ratio(iteration), iteration)
		for key, val in self.summary_dict.items():
			if not isinstance(val, list):
				writer.add_scalar("train_%s/%s" % (self.name, key), val, iteration)
				self.summary_dict[key] = 0.0
			elif len(val) == 0:
				continue
			elif not isinstance(val[0], list):
				writer.add_scalar("train_%s/%s" % (self.name, key), mean(val), iteration)
				self.summary_dict[key] = list()
			else:
				if self.debug or iteration % 5000 == 0: # Histograms can take time to be exported
					val = [v for sublist in val for v in sublist]
					if len(val) == 0:
						continue
					writer.add_histogram("train_%s/%s" % (self.name, key), np.array(val), iteration)
				self.summary_dict[key] = list()

		if self.debug or iteration % 5000 == 0:
			gen_list = self.generate_examples()
			for i in range(min(4, len(gen_list))):
				if not self.generated_before:
					writer.add_text(self.name + "_gen%i_input_phrase" % (i), gen_list[i][0], iteration)
					writer.add_text(self.name + "_gen%i_input_labels" % (i), gen_list[i][1], iteration)
					writer.add_text(self.name + "_gen%i_input_contexts" % (i), gen_list[i][2], iteration)
				writer.add_text(self.name + "_gen%i_reconstructed_phrase" % (i), gen_list[i][3], iteration)
				writer.add_text(self.name + "_gen%i_reconstructed_phrase_context" % (i), gen_list[i][4], iteration)
				writer.add_text(self.name + "_gen%i_reconstructed_phrase_tf" % (i), gen_list[i][5], iteration)

			gen_list = self.generate_random_style_samples()
			for i in range(min(4, len(gen_list))):
				if not self.generated_before:
					writer.add_text(self.name + "_samp%i_input_phrase" % (i), gen_list[i][0], iteration)
				for j in range(len(gen_list[i][1])):
					writer.add_text(self.name + "_samp%i_sample_%i" % (i, j), gen_list[i][1][j], iteration)

			self.generate_slot_dist_images(writer, iteration)

			self.generated_before = True


	def generate_examples(self):
		self.model.eval()
		# 1.) Put data on GPU
		batch_torch = UnsupervisedTask.batch_to_torch(self.gen_batch)
		eval_swr = (1.0 if self.switch_rate > 0.0 else 0.0)
		# 2.) Push data through network
		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = batch_torch
		par_1_masks = self.model.embedding_module.generate_mask(par_1_words)
		par_2_masks = self.model.embedding_module.generate_mask(par_2_words)
		with torch.no_grad():
			par_1_res, par_2_res, _, _, _, _, _, _, _, _ = self.model((par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths),
																	  teacher_forcing=False, switch_rate=eval_swr, semantic_full_dropout=self.semantic_full_dropout_eval, use_semantic_specific_attn=self.use_semantic_specific_attn)
			par_1_res_context, par_2_res_context, _, _, _, _, _, _, _, _ = self.model((par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths),
																	  				  teacher_forcing=False, switch_rate=eval_swr, semantic_full_dropout=self.semantic_full_dropout_eval, use_semantic_specific_attn=self.use_semantic_specific_attn, use_context_style=True)
			par_1_res_tf, par_2_res_tf, _, _, _, _, _, _, _, _ = self.model((par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths),
																			teacher_forcing=True, switch_rate=eval_swr, semantic_full_dropout=self.semantic_full_dropout_eval, use_semantic_specific_attn=self.use_semantic_specific_attn)
		del batch_torch
		# 3.) Reconstruct generated answer and input
		generated_paraphrases = list()
		generated_paraphrases_context = list()
		generated_paraphrases_tf = list()
		input_phrases = list()
		input_labels = list()
		input_contexts = list()

		gen_par_1_UNKdist = par_1_res[1].cpu().numpy()
		gen_par_1_words = par_1_res[2].cpu().numpy()
		gen_par_1_lengths = par_1_res[3].cpu().numpy()
		gen_par_2_UNKdist = par_2_res[1].cpu().numpy()
		gen_par_2_words = par_2_res[2].cpu().numpy()
		gen_par_2_lengths = par_2_res[3].cpu().numpy()

		gen_par_1_context_UNKdist = par_1_res_context[1].cpu().numpy()
		gen_par_1_context_words = par_1_res_context[2].cpu().numpy()
		gen_par_1_context_lengths = par_1_res_context[3].cpu().numpy()
		gen_par_2_context_UNKdist = par_2_res_context[1].cpu().numpy()
		gen_par_2_context_words = par_2_res_context[2].cpu().numpy()
		gen_par_2_context_lengths = par_2_res_context[3].cpu().numpy()

		gen_par_1_words_tf = par_1_res_tf[2].cpu().numpy()
		gen_par_1_lengths_tf = par_1_res_tf[3].cpu().numpy()
		gen_par_2_words_tf = par_2_res_tf[2].cpu().numpy()
		gen_par_2_lengths_tf = par_2_res_tf[3].cpu().numpy()

		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = self.gen_batch

		for embeds, lengths, list_to_add, slot_vals, slot_lengths, slot_preds in zip([contexts_1_words, contexts_2_words, par_1_words if eval_swr == 0 else par_2_words, par_2_words if eval_swr == 0 else par_1_words, par_1_words, par_2_words, gen_par_1_words, gen_par_2_words, gen_par_1_context_words, gen_par_2_context_words, gen_par_1_words_tf, gen_par_2_words_tf],
																					 [contexts_1_lengths, contexts_2_lengths, par_1_lengths if eval_swr == 0 else par_2_lengths, par_2_lengths if eval_swr == 0 else par_1_lengths, par_1_lengths, par_2_lengths, gen_par_1_lengths, gen_par_2_lengths, gen_par_1_context_lengths, gen_par_2_context_lengths, gen_par_1_lengths_tf, gen_par_2_lengths_tf],
																					 [input_contexts, input_contexts, input_phrases, input_phrases, input_labels, input_labels, generated_paraphrases, generated_paraphrases, generated_paraphrases_context, generated_paraphrases_context, generated_paraphrases_tf, generated_paraphrases_tf],
																					 [None, None, par_1_slots if eval_swr == 0 else par_2_slots, par_2_slots if eval_swr == 0 else par_1_slots, par_1_slots, par_2_slots, par_1_slots, par_2_slots, par_1_slots, par_2_slots, None, None],
																					 [None, None, par_1_slot_lengths if eval_swr == 0 else par_2_slot_lengths, par_2_slot_lengths if eval_swr == 0 else par_1_slot_lengths, par_1_slot_lengths, par_2_slot_lengths, par_1_slot_lengths, par_2_slot_lengths, par_1_slot_lengths, par_2_slot_lengths, None, None],
																					 [None, None, None, None, None, None, gen_par_1_UNKdist[:,:,1:], gen_par_2_UNKdist[:,:,1:], gen_par_1_context_UNKdist[:,:,1:], gen_par_2_context_UNKdist[:,:,1:], None, None]):
			reconstruct_sentences(embeds, lengths, slot_vals=slot_vals, slot_lengths=slot_lengths, slot_preds=slot_preds, list_to_add=list_to_add)	

		# 5.) Put everything in a nice format
		gen_list = list(zip(input_phrases, input_labels, input_contexts, generated_paraphrases, generated_paraphrases_context, generated_paraphrases_tf))
		self.model.train()
		return gen_list

	def generate_random_style_samples(self):
		self.model.eval()
		# 1.) Put data on GPU
		batch_torch = UnsupervisedTask.batch_to_torch(self.gen_batch)
		par_words, par_lengths, _, _, par_slots, par_slot_lengths, _, _, context_words, context_lengths, _, _ = batch_torch
		par_masks = self.model.embedding_module.generate_mask(par_words)
		with torch.no_grad():
			_, gen_par_UNK_weigths, gen_par_words, gen_par_lengths = self.model.sample_reconstruction_styles((par_words, par_lengths, par_masks, par_slots, par_slot_lengths, context_words, context_lengths), num_samples=8)
		del batch_torch
		# 3.) Reconstruct generated answer and input
		generated_paraphrases = list()
		input_phrases = list()

		gen_par_UNK_weigths = gen_par_UNK_weigths.cpu().numpy()
		gen_par_words = gen_par_words.cpu().numpy()
		gen_par_lengths = gen_par_lengths.cpu().numpy()

		par_words = self.gen_batch[0]
		par_lengths = self.gen_batch[1]
		par_slots = self.gen_batch[4]
		par_slot_lengths = self.gen_batch[5]

		for embeds, lengths, list_to_add, add_sents_up, slot_preds in zip([par_words, gen_par_words],
																		  [par_lengths, gen_par_lengths],
																		  [input_phrases, generated_paraphrases],
																		  [True, False],
																		  [None, gen_par_UNK_weigths[:,:,1:]]):
			reconstruct_sentences(embeds, lengths, slot_vals=par_slots, slot_lengths=par_slot_lengths, slot_preds=slot_preds, list_to_add=list_to_add, add_sents_up=add_sents_up)

		# 5.) Put everything in a nice format
		gen_list = list(zip(input_phrases, generated_paraphrases))
		self.model.train()
		return gen_list


	def generate_style_dist(self):
		self.model.eval()
		# 1.) Put data on GPU
		batch_torch = UnsupervisedTask.batch_to_torch(self.gen_batch)
		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, _, _, _, _ = batch_torch
		par_1_masks = self.model.embedding_module.generate_mask(par_1_words)
		par_2_masks = self.model.embedding_module.generate_mask(par_2_words)
		with torch.no_grad():
			_, gen_par_1_UNK_weigths, gen_par_1_words, gen_par_1_lengths, gen_proto_dists, par_1_proto_dist = self.model.generate_style_dist((par_1_words, par_1_lengths, par_1_masks, par_1_slots, par_1_slot_lengths), max_generation_steps=40)
			_, gen_par_2_UNK_weigths, gen_par_2_words, gen_par_2_lengths, _, par_2_proto_dist = self.model.generate_style_dist((par_2_words, par_2_lengths, par_2_masks, par_2_slots, par_2_slot_lengths), max_generation_steps=40)
		del batch_torch
		# 3.) Reconstruct generated answer and input
		generated_paraphrases = list()
		input_phrases = list()
		phrase_proto_dist = list()
		proto_dist_descriptions = list()

		gen_par_1_UNK_weigths = gen_par_1_UNK_weigths.cpu().numpy()
		gen_par_1_words = gen_par_1_words.cpu().numpy()
		gen_par_1_lengths = gen_par_1_lengths.cpu().numpy()
		gen_par_2_UNK_weigths = gen_par_2_UNK_weigths.cpu().numpy()
		gen_par_2_words = gen_par_2_words.cpu().numpy()
		gen_par_2_lengths = gen_par_2_lengths.cpu().numpy()
		gen_proto_dists = gen_proto_dists.cpu().numpy()
		par_1_proto_dist = par_1_proto_dist.cpu().numpy()
		par_2_proto_dist = par_2_proto_dist.cpu().numpy()
		par_proto_dist = np.concatenate([par_1_proto_dist, par_2_proto_dist], axis=0)

		par_1_words = self.gen_batch[0]
		par_1_lengths = self.gen_batch[1]
		par_2_words = self.gen_batch[2]
		par_2_lengths = self.gen_batch[3]
		par_1_slots = self.gen_batch[4]
		par_1_slot_lengths = self.gen_batch[5]
		par_2_slots = self.gen_batch[6]
		par_2_slot_lengths = self.gen_batch[7]

		for embeds, lengths, par_slots, par_slot_lengths, list_to_add, add_sents_up, slot_preds in zip([par_1_words, par_2_words, gen_par_1_words, gen_par_2_words],
																									   [par_1_lengths, par_2_lengths, gen_par_1_lengths, gen_par_2_lengths],
																									   [par_1_slots, par_2_slots, par_1_slots, par_2_slots],
																									   [par_1_slot_lengths, par_2_slot_lengths, par_1_slot_lengths, par_2_slot_lengths],
																									   [input_phrases, input_phrases, generated_paraphrases, generated_paraphrases],
																									   [True, True, False, False],
																									   [None, None, gen_par_1_UNK_weigths[:,:,1:], gen_par_2_UNK_weigths[:,:,1:]]):
			reconstruct_sentences(embeds, lengths, slot_vals=par_slots, slot_lengths=par_slot_lengths, slot_preds=slot_preds, list_to_add=list_to_add, add_sents_up=add_sents_up)

		for proto_index in range(gen_proto_dists.shape[0]):
			s = ""
			if np.max(gen_proto_dists[proto_index]) == 1.0:
				s = "Proto %i" % np.argmax(gen_proto_dists[proto_index])
			elif np.sum(gen_proto_dists[proto_index] == 0.5) == 2:
				proto_indices = np.where(gen_proto_dists[proto_index] == 0.5)[0]
				s = "Proto %i and %i" % (proto_indices[0], proto_indices[1])
			elif np.all(gen_proto_dists[proto_index] == gen_proto_dists[proto_index,0]):
				s = "All proto"
			else:
				s = "Unknown proto"
			proto_dist_descriptions.append(s)

		for par_index in range(par_proto_dist.shape[0]):
			proto_dist_string = "[%s]" % (", ".join(["%4.2f%%" % (par_proto_dist[par_index,i]*100.0) for i in range(par_proto_dist.shape[1])]))
			phrase_proto_dist.append(proto_dist_string)
		# 5.) Put everything in a nice format
		gen_list = list(zip(input_phrases, generated_paraphrases, phrase_proto_dist))
		self.model.train()
		return gen_list, proto_dist_descriptions


	def generate_beamsearch_batchwise(self, batch_torch, beam_search_method="diverse"):
		self.model.eval()
		# 1.) Put data on GPU
		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = batch_torch
		par_1_masks = self.model.embedding_module.generate_mask(par_1_words)
		par_2_masks = self.model.embedding_module.generate_mask(par_2_words)
		with torch.no_grad():
			p1_res_beam, p2_res_beam, _, _, _, _, _, _, _, _ = self.model((par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths), 
															      teacher_forcing=False, beams=8, switch_rate=1.0, semantic_full_dropout=self.semantic_full_dropout_eval, max_generation_steps=40, use_semantic_specific_attn = self.use_semantic_specific_attn, use_context_style=True, 
															      ignore_context = self.model.encoder_module.use_prototype_styles and (self.style_loss_scaling == 0), only_par_1=False, beam_search_method=beam_search_method)
			_, gen_par_1_UNK_weigths, gen_par_1_words, gen_par_1_lengths = p1_res_beam
			_, gen_par_2_UNK_weigths, gen_par_2_words, gen_par_2_lengths = p2_res_beam
			
		# 3.) Reconstruct generated answer and input
		generated_paraphrases = list()
		input_phrases = list()

		def pad_UNK_weights(UNK_weights):
			max_shape_1 = max([g.shape[1] for g in UNK_weights])
			max_shape_2 = max([g.shape[2] for g in UNK_weights])
			UNK_weights = np.stack([np.concatenate([np.concatenate([g, np.zeros((g.shape[0], max_shape_1-g.shape[1], g.shape[2]))], axis=1), np.zeros((g.shape[0], max_shape_1, max_shape_2-g.shape[2]))], axis=2) for g in UNK_weights], axis=0)
			return UNK_weights

		gen_par_1_UNK_weigths = [g.cpu().numpy() for g in gen_par_1_UNK_weigths]
		print([g.shape for g in gen_par_1_UNK_weigths])
		gen_par_1_UNK_weigths = pad_UNK_weights(gen_par_1_UNK_weigths)
		
		gen_par_1_words = gen_par_1_words.cpu().numpy()
		gen_par_1_lengths = gen_par_1_lengths.cpu().numpy()
		gen_par_2_UNK_weigths = [g.cpu().numpy() for g in gen_par_2_UNK_weigths]
		gen_par_2_UNK_weigths = pad_UNK_weights(gen_par_2_UNK_weigths)
		gen_par_2_words = gen_par_2_words.cpu().numpy()
		gen_par_2_lengths = gen_par_2_lengths.cpu().numpy()

		par_1_words = batch_torch[0].cpu().numpy()
		par_1_lengths = batch_torch[1].cpu().numpy()
		par_2_words = batch_torch[2].cpu().numpy()
		par_2_lengths = batch_torch[3].cpu().numpy()
		par_1_slots = batch_torch[4].cpu().numpy()
		par_1_slot_lengths = batch_torch[5].cpu().numpy()
		par_2_slots = batch_torch[6].cpu().numpy()
		par_2_slot_lengths = batch_torch[7].cpu().numpy()

		for embeds, lengths, par_slots, par_slot_lengths, list_to_add, add_sents_up, slot_preds in zip([par_1_words, par_2_words, gen_par_1_words, gen_par_2_words],
																									   [par_1_lengths, par_2_lengths, gen_par_1_lengths, gen_par_2_lengths],
																									   [par_1_slots, par_2_slots, par_1_slots, par_2_slots],
																									   [par_1_slot_lengths, par_2_slot_lengths, par_1_slot_lengths, par_2_slot_lengths],
																									   [input_phrases, input_phrases, generated_paraphrases, generated_paraphrases],
																									   [True, True, False, False],
																									   [None, None, gen_par_1_UNK_weigths[:,:,1:], gen_par_2_UNK_weigths[:,:,1:]]):
			reconstruct_sentences(embeds, lengths, slot_vals=par_slots, slot_lengths=par_slot_lengths, slot_preds=slot_preds, list_to_add=list_to_add, add_sents_up=add_sents_up)

		# 5.) Put everything in a nice format
		gen_list = list(zip(input_phrases, generated_paraphrases))
		self.model.train()
		return gen_list


	def generate_style_dist_batchwise(self, batch_torch):
		self.model.eval()
		# 1.) Put data on GPU
		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, _, _, _, _ = batch_torch
		par_1_masks = self.model.embedding_module.generate_mask(par_1_words)
		par_2_masks = self.model.embedding_module.generate_mask(par_2_words)
		with torch.no_grad():
			_, gen_par_1_UNK_weigths, gen_par_1_words, gen_par_1_lengths, gen_proto_dists, par_1_proto_dist = self.model.generate_style_dist((par_1_words, par_1_lengths, par_1_masks, par_1_slots, par_1_slot_lengths), max_generation_steps=40)
			_, gen_par_2_UNK_weigths, gen_par_2_words, gen_par_2_lengths, _, par_2_proto_dist = self.model.generate_style_dist((par_2_words, par_2_lengths, par_2_masks, par_2_slots, par_2_slot_lengths), max_generation_steps=40)
		
		# 3.) Reconstruct generated answer and input
		generated_paraphrases = list()
		input_phrases = list()
		phrase_proto_dist = list()
		proto_dist_descriptions = list()

		gen_par_1_UNK_weigths = gen_par_1_UNK_weigths.cpu().numpy()
		gen_par_1_words = gen_par_1_words.cpu().numpy()
		gen_par_1_lengths = gen_par_1_lengths.cpu().numpy()
		gen_par_2_UNK_weigths = gen_par_2_UNK_weigths.cpu().numpy()
		gen_par_2_words = gen_par_2_words.cpu().numpy()
		gen_par_2_lengths = gen_par_2_lengths.cpu().numpy()
		gen_proto_dists = gen_proto_dists.cpu().numpy()
		par_1_proto_dist = par_1_proto_dist.cpu().numpy()
		par_2_proto_dist = par_2_proto_dist.cpu().numpy()
		par_proto_dist = np.concatenate([par_1_proto_dist, par_2_proto_dist], axis=0)

		print("Shape gen_par_1_UNK_weigths", gen_par_1_UNK_weigths.shape)

		par_1_words = batch_torch[0].cpu().numpy()
		par_1_lengths = batch_torch[1].cpu().numpy()
		par_2_words = batch_torch[2].cpu().numpy()
		par_2_lengths = batch_torch[3].cpu().numpy()
		par_1_slots = batch_torch[4].cpu().numpy()
		par_1_slot_lengths = batch_torch[5].cpu().numpy()
		par_2_slots = batch_torch[6].cpu().numpy()
		par_2_slot_lengths = batch_torch[7].cpu().numpy()

		for embeds, lengths, par_slots, par_slot_lengths, list_to_add, add_sents_up, slot_preds in zip([par_1_words, par_2_words, gen_par_1_words, gen_par_2_words],
																									   [par_1_lengths, par_2_lengths, gen_par_1_lengths, gen_par_2_lengths],
																									   [par_1_slots, par_2_slots, par_1_slots, par_2_slots],
																									   [par_1_slot_lengths, par_2_slot_lengths, par_1_slot_lengths, par_2_slot_lengths],
																									   [input_phrases, input_phrases, generated_paraphrases, generated_paraphrases],
																									   [True, True, False, False],
																									   [None, None, gen_par_1_UNK_weigths[:,:,1:], gen_par_2_UNK_weigths[:,:,1:]]):
			reconstruct_sentences(embeds, lengths, slot_vals=par_slots, slot_lengths=par_slot_lengths, slot_preds=slot_preds, list_to_add=list_to_add, add_sents_up=add_sents_up)

		for proto_index in range(gen_proto_dists.shape[0]):
			s = ""
			if np.max(gen_proto_dists[proto_index]) == 1.0:
				s = "Proto %i" % np.argmax(gen_proto_dists[proto_index])
			elif np.sum(gen_proto_dists[proto_index] == 0.5) == 2:
				proto_indices = np.where(gen_proto_dists[proto_index] == 0.5)[0]
				s = "Proto %i and %i" % (proto_indices[0], proto_indices[1])
			elif np.all(gen_proto_dists[proto_index] == gen_proto_dists[proto_index,0]):
				s = "All proto"
			else:
				s = "Unknown proto"
			proto_dist_descriptions.append(s)

		for par_index in range(par_proto_dist.shape[0]):
			proto_dist_string = "[%s]" % (", ".join(["%4.2f%%" % (par_proto_dist[par_index,i]*100.0) for i in range(par_proto_dist.shape[1])]))
			phrase_proto_dist.append(proto_dist_string)
		# 5.) Put everything in a nice format
		gen_list = list(zip(input_phrases, generated_paraphrases, phrase_proto_dist))
		self.model.train()
		return gen_list, proto_dist_descriptions


	def generate_styles_batchwise(self, batch_torch):
		self.model.eval()
		# 1.) Put data on GPU
		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = batch_torch
		par_1_masks = self.model.embedding_module.generate_mask(par_1_words)
		par_2_masks = self.model.embedding_module.generate_mask(par_2_words)
		with torch.no_grad():
			_, gen_par_1_UNK_weigths, gen_par_1_words, gen_par_1_lengths = self.model.sample_reconstruction_styles((par_1_words, par_1_lengths, par_1_masks, par_1_slots, par_1_slot_lengths, contexts_1_words, contexts_1_lengths), max_generation_steps=40, num_samples=8, sample_context=False, sample_gt=True)
			_, gen_par_2_UNK_weigths, gen_par_2_words, gen_par_2_lengths = self.model.sample_reconstruction_styles((par_2_words, par_2_lengths, par_2_masks, par_2_slots, par_2_slot_lengths, contexts_2_words, contexts_2_lengths), max_generation_steps=40, num_samples=8, sample_context=False, sample_gt=True)
		
		# 3.) Reconstruct generated answer and input
		generated_paraphrases = list()
		input_phrases = list()

		gen_par_1_UNK_weigths = gen_par_1_UNK_weigths.cpu().numpy()
		gen_par_1_words = gen_par_1_words.cpu().numpy()
		gen_par_1_lengths = gen_par_1_lengths.cpu().numpy()
		gen_par_2_UNK_weigths = gen_par_2_UNK_weigths.cpu().numpy()
		gen_par_2_words = gen_par_2_words.cpu().numpy()
		gen_par_2_lengths = gen_par_2_lengths.cpu().numpy()

		par_1_words = batch_torch[0].cpu().numpy()
		par_1_lengths = batch_torch[1].cpu().numpy()
		par_2_words = batch_torch[2].cpu().numpy()
		par_2_lengths = batch_torch[3].cpu().numpy()
		par_1_slots = batch_torch[4].cpu().numpy()
		par_1_slot_lengths = batch_torch[5].cpu().numpy()
		par_2_slots = batch_torch[6].cpu().numpy()
		par_2_slot_lengths = batch_torch[7].cpu().numpy()

		for embeds, lengths, par_slots, par_slot_lengths, list_to_add, add_sents_up, slot_preds in zip([par_1_words, par_2_words, gen_par_1_words, gen_par_2_words],
																									   [par_1_lengths, par_2_lengths, gen_par_1_lengths, gen_par_2_lengths],
																									   [par_1_slots, par_2_slots, par_1_slots, par_2_slots],
																									   [par_1_slot_lengths, par_2_slot_lengths, par_1_slot_lengths, par_2_slot_lengths],
																									   [input_phrases, input_phrases, generated_paraphrases, generated_paraphrases],
																									   [True, True, False, False],
																									   [None, None, gen_par_1_UNK_weigths[:,:,1:], gen_par_2_UNK_weigths[:,:,1:]]):
			reconstruct_sentences(embeds, lengths, slot_vals=par_slots, slot_lengths=par_slot_lengths, slot_preds=slot_preds, list_to_add=list_to_add, add_sents_up=add_sents_up)

		# 5.) Put everything in a nice format
		gen_list = list(zip(input_phrases, generated_paraphrases))
		self.model.train()
		return gen_list


	def extract_gt_attn(self, batch_torch):
		self.model.eval()
		# 1.) Put data on GPU
		par_1_words, par_1_lengths, _, _, par_1_slots, par_1_slot_lengths, _, _, _, _, _, _ = batch_torch
		with torch.no_grad():
			par_1_attn_semantic, par_1_attn_style, par_1_proto_dist = self.model.encode_gt(_input=(par_1_words, par_1_lengths, par_1_slots, par_1_slot_lengths))
		# 3.) Reconstruct generated answer and input
		input_phrases = list()

		par_1_words = batch_torch[0].cpu().numpy()
		par_1_lengths = batch_torch[1].cpu().numpy()
		par_1_slots = batch_torch[4].cpu().numpy()
		par_1_slot_lengths = batch_torch[5].cpu().numpy()

		reconstruct_sentences(par_1_words, par_1_lengths, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths, slot_preds=None, list_to_add=input_phrases, add_sents_up=True)

		# 5.) Put everything in a nice format
		attn_info = []
		print("Attention semantic", par_1_attn_semantic)
		print("Style attention", par_1_attn_style)
		for sent_index in range(len(input_phrases)):
			info = ["Semantic attention: " + str(par_1_attn_semantic[sent_index].cpu().numpy()),
					"Style attention: " + str(par_1_attn_style[sent_index].cpu().numpy()),
					"Prototype distribution: " + str(par_1_proto_dist[sent_index].cpu().numpy())]
			attn_info.append(info)
		gen_list = list(zip(input_phrases, attn_info))
		self.model.train()
		return gen_list



	def generate_slot_dist_images(self, writer, iteration):
		
		self.model.eval()
		# 1.) Put data on GPU
		batch_torch = UnsupervisedTask.batch_to_torch(self.gen_batch)
		eval_swr = (1.0 if self.switch_rate > 0.0 else 0.0)
		# 2.) Push data through network
		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = batch_torch
		par_1_masks = self.model.embedding_module.generate_mask(par_1_words)
		par_2_masks = self.model.embedding_module.generate_mask(par_2_words)
		with torch.no_grad():
			par_1_res, par_2_res, _, _, _, _, _, slot_1_res, slot_2_res, _ = self.model((par_1_words, par_1_lengths, par_1_masks, par_2_words, par_2_lengths, par_2_masks, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths), 
																						teacher_forcing=False, switch_rate=eval_swr, semantic_full_dropout=self.semantic_full_dropout_eval, max_generation_steps=40, use_semantic_specific_attn = self.use_semantic_specific_attn)
		del batch_torch

		_, par_1_UNK_weights, par_1_preds, par_1_lengths = par_1_res
		_, slot_1_lengths, slot_1_ids = slot_1_res

		par_1_UNK_weights = par_1_UNK_weights.cpu().numpy()
		par_1_preds = par_1_preds.cpu().numpy()
		par_1_lengths = par_1_lengths.cpu().numpy()
		par_1_slots = par_1_slots.cpu().numpy()
		par_1_slot_lengths = par_1_slot_lengths.cpu().numpy()
		slot_1_lengths = slot_1_lengths.cpu().numpy()
		slot_1_ids = slot_1_ids.cpu().numpy()

		id2word = get_id2word_dict()
		fig = plt.figure()

		slot_sents = reconstruct_sentences(slot_1_ids, slot_1_lengths, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths, add_sents_up=False)
		pred_sents = reconstruct_sentences(par_1_preds, par_1_lengths, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths, slot_preds=par_1_UNK_weights[:,:,1:], add_sents_up=False)
		
		for batch_index in range(par_1_UNK_weights.shape[0]):

			if slot_1_lengths[batch_index] == 0 or par_1_lengths[batch_index] == 0:
				continue

			fig = plt.figure()
			ax = fig.add_subplot(111)
			
			sent_attention_map = par_1_UNK_weights[batch_index, :par_1_lengths[batch_index], :slot_1_lengths[batch_index]+1]
			
			sent_attention_map = np.concatenate((sent_attention_map[:,0:1], sent_attention_map), axis=1)
			sent_attention_map[:,0] = (sent_attention_map[:,0] < 0.5)
			sent_attention_map[:,1] = 1 - sent_attention_map[:,1]
			sent_attention_map[:,2:] /= np.maximum(np.sum(sent_attention_map[:,2:], axis=-1, keepdims=True), 1e-5)

			sent_attention_map = np.transpose(sent_attention_map)

			cax = ax.matshow(sent_attention_map, cmap=plt.cm.gray, vmin=0, vmax=1)
			ax.set_yticklabels(["use slot (bin)", "use slot"] + slot_sents[batch_index])
			ax.set_xticklabels(pred_sents[batch_index], rotation=90)
			ax.set_yticks(range(2 + slot_1_lengths[batch_index]))
			ax.set_xticks(range(par_1_lengths[batch_index]))
			ax.set_yticks(np.arange(-.5, 2 + slot_1_lengths[batch_index], 1), minor=True)
			ax.set_xticks(np.arange(-.5, par_1_lengths[batch_index], 1), minor=True)
			ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
			
			# Add rectangle for the chosen slots
			for seq_index in range(sent_attention_map.shape[1]):
				if sent_attention_map[0,seq_index] == 1:
					best_ind = np.argmax(sent_attention_map[2:,seq_index]) + 2
					ax.add_patch(Rectangle((seq_index-0.5,best_ind-0.5),1,1,linewidth=2,edgecolor=(0.5,1.0,0.5),facecolor='none'))

			plt.tight_layout()
			writer.add_figure(tag="train/%s_sample_attention_maps_%i" % (self.name, batch_index), figure=fig, global_step=iteration)
		plt.close()

		self.model.train()

	def finalize_summary(self, writer, iteration, checkpoint_path):
		# if not self.debug:
		# 	self.create_tSNE_embeddings(self.val_dataset, writer, iteration, prefix="val_")
		# 	self.create_tSNE_embeddings(self.train_dataset, writer, iteration, prefix="train_")
		self.export_whole_dataset(self.train_dataset, checkpoint_path, prefix="train_")
		self.export_whole_dataset(self.val_dataset, checkpoint_path, prefix="val_")

	def export_whole_dataset(self, dataset, checkpoint_path, prefix="", batch_size=128):
		self.model.eval()
		data, data_indices = dataset.get_all_sentences()
		
		num_batches = int(math.ceil(len(data) * 1.0 / batch_size))
		if self.debug:
			batch_size = 4
			num_batches = min(num_batches, 2)

		par_semantic_vecs, par_style_vecs, context_style_attn_vecs, context_style_vecs = None, None, None, None
		
		par_words = ["%i\t%s" % (d_index, " ".join(d.par_1_words)) for d, d_index in zip(data, data_indices)]
		context_words = ["\t\t".join([" ".join(c) for c in d.context_1_words]) for d in data]

		for n in range(num_batches):
			batch_data = data[n*batch_size:min((n+1)*batch_size, len(data))]
			with torch.no_grad():
				batch_data_input = dataset._data_to_batch(batch_data, toTorch=True)
				batch_par, batch_par_length, _, _, batch_slots, batch_slot_lengths, _, _, batch_context, batch_context_length, _, _ = batch_data_input
				par_semantics, par_style, context_style_attn, context_style = self.model.encode_sent_context((batch_par, batch_par_length, batch_slots, batch_slot_lengths, batch_context, batch_context_length))
			par_semantics = par_semantics.cpu().numpy()
			par_style = par_style.cpu().numpy()
			context_style_attn = context_style_attn.cpu().numpy()
			context_style = context_style.cpu().numpy()

			if n == 0:
				par_semantic_vecs, par_style_vecs, context_style_attn_vecs, context_style_vecs = par_semantics, par_style, context_style_attn, context_style
			else:
				par_semantic_vecs = np.concatenate([par_semantic_vecs, par_semantics], axis=0)
				par_style_vecs = np.concatenate([par_style_vecs, par_style], axis=0)
				context_style_attn_vecs = np.concatenate([context_style_attn_vecs, context_style_attn], axis=0)
				context_style_vecs = np.concatenate([context_style_vecs, context_style], axis=0)

		dir_path = os.path.join(checkpoint_path, prefix+self.name+"_export")
		os.makedirs(dir_path, exist_ok=True)
		np.savez_compressed(os.path.join(dir_path, "par_semantic_vecs.npz"), par_semantic_vecs)
		np.savez_compressed(os.path.join(dir_path, "par_style_vecs.npz"), par_style_vecs)
		np.savez_compressed(os.path.join(dir_path, "context_style_attn_vecs.npz"), context_style_attn_vecs)
		np.savez_compressed(os.path.join(dir_path, "context_style_vecs.npz"), context_style_vecs)

		with open(os.path.join(dir_path, "responses.txt"), "w") as f:
			f.write("\n".join(par_words))
		with open(os.path.join(dir_path, "contexts.txt"), "w") as f:
			f.write("\n".join(context_words))



		
	def create_tSNE_embeddings(self, dataset, writer, iteration, prefix="", batch_size=64, max_number_batches=15):
		# tSNE embeddings of styles and semantics
		self.model.eval()
		
		# Prepare metrics
		number_batches = int(math.ceil(dataset.get_num_examples() * 1.0 / batch_size))
		number_batches = min(number_batches, max_number_batches)
		context_style_embed_list = list()
		response_style_embed_list = list()
		semantic_embed_list = list()
		original_response_list = list()
		original_context_list = list()

		# Evaluation loop
		for batch_ind in range(number_batches):
			if debug_level() == 0:
				print("Saving %sembeddings for tSNE: %4.2f%% (batch %i of %i)" % (prefix, 100.0 * batch_ind / number_batches, batch_ind+1, number_batches), end="\r")

			par_1_words, par_1_lengths, _, _, par_1_slots, par_1_slot_lengths, _, _, contexts_1_words, contexts_1_lengths, _, _ = dataset.get_batch(batch_size, loop_dataset=False, toTorch=True, label_lengths=False, noun_mask=None, mask_prob=0.0)
			par_semantics, par_style = self.model.encode_sentence((par_1_words, par_1_lengths, par_1_slots, par_1_slot_lengths))
			_, context_style = self.model.encode_sentence((contexts_1_words, contexts_1_lengths, None, None))

			semantic_embed_list.append(par_semantics)
			response_style_embed_list.append(par_style[0])
			context_style_embed_list.append(context_style[0])

			par_1_words = par_1_words.cpu().numpy()
			par_1_lengths = par_1_lengths.cpu().numpy()
			par_1_slots = par_1_slots.cpu().numpy()
			par_1_slot_lengths = par_1_slot_lengths.cpu().numpy()
			contexts_1_words = contexts_1_words.cpu().numpy()
			contexts_1_lengths = contexts_1_lengths.cpu().numpy()

			reconstruct_sentences(par_1_words, par_1_lengths, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths, list_to_add=original_response_list)
			reconstruct_sentences(contexts_1_words[:,-1], contexts_1_lengths[:,-1], list_to_add=original_context_list) # -1 to only use last sentence
		
		semantic_embeds = torch.cat(semantic_embed_list, dim=0)
		response_style_embeds = torch.cat(response_style_embed_list, dim=0)
		context_style_embeds = torch.cat(context_style_embed_list, dim=0)

		print("Semantic embeds: " + str(semantic_embeds.shape))
		print("Response style embeds: " + str(response_style_embeds.shape))
		print("Context style embeds: " + str(context_style_embeds.shape))
		print("Original response length: " + str(len(original_response_list)))
		print("Original context length: " + str(len(original_context_list)))

		writer.add_embedding(semantic_embeds, metadata=original_response_list, tag=prefix + "Semantic_" + self.name, global_step=iteration)
		writer.add_embedding(response_style_embeds, metadata=original_response_list, tag=prefix + "Response_Style_" + self.name, global_step=iteration)
		writer.add_embedding(context_style_embeds, metadata=original_context_list, tag=prefix + "Context_Style_" + self.name, global_step=iteration)
		self.model.train()

	def export_best_results(self, checkpoint_path, iteration):
		if False and (not self.debug and iteration < 10000): # Reduce amount of times this function is called. Do not expect best results before 20k
			return

		if self.name == "ContextAwareDialogueParaphraseSmall":
			return

		print("Exporting best results of %s..." % self.name)

		TOK_WIDTH = 200

		gen_list = self.generate_examples()
		results_gen = "-- Iteration %i --\n" % iteration
		for g in gen_list:
			results_gen += "\n\n" + "="*TOK_WIDTH + "\n"
			results_gen += g[2] + "\n"
			results_gen += "-"*TOK_WIDTH + "\n"
			results_gen += "Generated           | %s\n" % (g[3])
			results_gen += "Generated (context) | %s\n" % (g[4])
			results_gen += "Ground truth        | %s\n" % (g[1])
			results_gen += "="*TOK_WIDTH + "\n"

		with open(os.path.join(checkpoint_path, self.name.lower() + "_generation.txt"), "w") as f:
			f.write(results_gen)

		gen_list = self.generate_random_style_samples()
		results_random_styles = "-- Iteration %i --\n" % iteration
		for g in gen_list:
			results_random_styles += "\n\n" + "="*TOK_WIDTH + "\n"
			results_random_styles += "Input:\t%s\n" % (g[0])
			results_random_styles += "-"*TOK_WIDTH + "\n"
			results_random_styles += "\n".join(["Gen (%i):\t%s" % (i, e) for i, e in enumerate(g[1])])
			results_random_styles += "\n" + "="*TOK_WIDTH + "\n"

		with open(os.path.join(checkpoint_path, self.name.lower() + "_random_styles.txt"), "w") as f:
			f.write(results_random_styles)

		if True or (self.model.encoder_module.use_prototype_styles and not self.model.encoder_module.no_prototypes_for_context):
			gen_list, proto_dist_list = [], []
			gen_style_list = []
			gt_attention_maps = []
			gen_beam_std_list, gen_beam_sto_list, gen_beam_div_list = [], [], []
			batch_size = 64
			number_batches = int(math.ceil(self.test_dataset.get_num_examples() * 1.0 / batch_size))
			if self.debug:
				number_batches = min(number_batches, 2)

			def export_list(gen_resp_list, name):
				res = "-- Iteration %i --\n" % iteration
				for g in gen_resp_list:
					res += "\n\n" + "="*TOK_WIDTH + "\n"
					res += "Input:\t%s\n" % (g[0])
					res += "-"*TOK_WIDTH + "\n"
					res += "\n".join(["(%i):\t%s" % (i, e) for i, e in enumerate(g[1])])

				with open(os.path.join(checkpoint_path, "%s_%s.txt" % (self.name.lower(), name)), "w") as f:
					f.write(res)

			for batch_index in range(number_batches):
				batch_data_input = self.test_dataset.get_batch(batch_size, loop_dataset=False, toTorch=True, label_lengths=False, noun_mask=False, mask_prob=0.0)
				glist, protolist = self.generate_style_dist_batchwise(batch_data_input)
				gslist = self.generate_styles_batchwise(batch_data_input)
				gt_attn_maps = self.extract_gt_attn(batch_data_input)
				
				# start_time = time.time()
				# gbstdlist = self.generate_beamsearch_batchwise(batch_data_input, beam_search_method="standard")
				# print("Finished batch of standard beam search in %4.2fs" % (time.time() - start_time))

				# start_time = time.time()
				# gbstolist = self.generate_beamsearch_batchwise(batch_data_input, beam_search_method="stochastic")
				# print("Finished batch of stochastic beam search in %4.2fs" % (time.time() - start_time))

				# start_time = time.time()
				# gbdivlist = self.generate_beamsearch_batchwise(batch_data_input, beam_search_method="diverse")
				# print("Finished batch of diverse beam search in %4.2fs" % (time.time() - start_time))

				gt_attention_maps += gt_attn_maps
				gen_list += glist
				proto_dist_list += protolist
				# gen_beam_std_list += gbstdlist
				# gen_beam_sto_list += gbstolist
				# gen_beam_div_list += gbdivlist
				gen_style_list += gslist

				export_list(gt_attention_maps, "gt_attention_maps")
				export_list(gen_beam_std_list, "beam_standard")
				export_list(gen_beam_sto_list, "beam_stochastic")
				export_list(gen_beam_div_list, "beam_diverse")
				export_list(gen_style_list, "styles")

			# gen_list, proto_dist_list = self.generate_style_dist()
			results_style_dist = "-- Iteration %i --\n" % iteration
			for g in gen_list:
				results_style_dist += "\n\n" + "="*TOK_WIDTH + "\n"
				results_style_dist += "Input:\t%s\n" % (g[0])
				results_style_dist += "Proto distribution:\t%s\n" % (g[2])
				results_style_dist += "-"*TOK_WIDTH + "\n"
				results_style_dist += "\n".join(["(%s):\t%s" % (name, e) for name, e in zip(proto_dist_list, g[1])])
				results_style_dist += "\n" + "="*TOK_WIDTH + "\n"
			
			with open(os.path.join(checkpoint_path, self.name.lower() + "_style_dist.txt"), "w") as f:
				f.write(results_style_dist)




class ContextAwareLanguageModelingTask(TaskTemplate):


	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix="", dataset_fun=DatasetHandler.load_ContextLM_Book_datasets):
		super(ContextAwareLanguageModelingTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name="ContextAwareLanguageModeling" + name_suffix, dataset_fun=dataset_fun)
		self.loss_module = self.model.get_loss_module()
		self.KL_scheduler = create_KLScheduler(scheduler_type = get_param_val(model_params, "VAE_scheduler", 1),
											   annealing_func_type = get_param_val(model_params, "VAE_annealing_func", 0),
											   loss_scaling = get_param_val(model_params, "VAE_loss_scaling", 1.0),
											   num_iters = get_param_val(model_params, "VAE_annealing_iters", 10000))
		self.summary_dict = {"loss_rec": list(), 
							 "loss_VAE": list(), 
							 "KL_scheduler": list(),
							 "loss_combined": list(), 
							 "style_mu": list(), 
							 "style_sigma": list()}


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_fun(debug_dataset=self.debug, num_context_sents=get_param_val(self.model_params, "num_context_turns", 2)*2-1)


	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		par_words, par_lengths, contexts_words, contexts_lengths = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)

		current_tf_ratio = self._get_tf_ratio(iteration)

		par_res, context_style = self.model.language_modeling(_input = (par_words, par_lengths, contexts_words, contexts_lengths), 
															  teacher_forcing = True, 
															  teacher_forcing_ratio = current_tf_ratio)
		loss, loss_VAE, acc = self._calculate_loss(par_res, par_words, context_style)
		final_loss = loss + loss_VAE * self.KL_scheduler.get(iteration)

		self.summary_dict["loss_rec"].append(loss.item())
		self.summary_dict["loss_VAE"].append(loss_VAE.item())
		self.summary_dict["KL_scheduler"] = [self.KL_scheduler.get(iteration)]
		self.summary_dict["loss_combined"].append(final_loss.item())
		for dict_key, hist_tensors in zip(["style_mu", "style_sigma"], [[context_style[1]], [context_style[2]]]):
			new_vals = [t.detach().cpu().contiguous().view(-1).numpy().tolist() for t in hist_tensors if t is not None]
			new_vals = [e for sublist in new_vals for e in sublist]
			self.summary_dict[dict_key].append(new_vals)
			while len(self.summary_dict[dict_key]) > 10:
				del self.summary_dict[dict_key][0]

		return final_loss, acc

	def _calculate_loss(self, par_res, batch_labels, context_style, par_style=None):
		par_word_dist, slot_dist, _, _ = par_res
		# Remove unknown word labels from the loss
		if (batch_labels[:,0] == get_SOS_index()).byte().all():
			batch_labels = batch_labels[:,1:]
		else:
			print("[#] WARNING: Batch labels were not shortend. First token ids: \n%s \nSOS index: %i" % (str(batch_labels[:,0]), get_SOS_index()))
		unknown_label = ((batch_labels == get_UNK_index()) | (batch_labels < 0)).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label

		## Loss reconstruction
		loss = self.loss_module(par_word_dist.view(-1, par_word_dist.shape[-1]), batch_labels.view(-1))
		## Accuracy calculation
		_, pred_labels = torch.max(par_word_dist, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / torch.sum(batch_labels != -1).float()
		## Loss VAE regularization
		_, style_mu, style_std = context_style
		if par_style is not None:
			_, par_style_mu, par_style_std = par_style
			style_mu = torch.cat([style_mu, par_style_mu], dim=-1)
			style_std = torch.cat([style_std, par_style_std], dim=-1)
		loss_VAE = ContextAwareDialogueTask._calc_loss_VAE(style_mu, style_std)

		return loss, loss_VAE, acc

	def eval(self, dataset=None, batch_size=64, label_lengths=False, noun_mask=False):
		return float('nan'), dict()

	def add_summary(self, writer, iteration):
		# TODO: Add some example generations here. Either run the model again for some random sentences, or save last training sentences
		writer.add_scalar("train_%s/teacher_forcing_ratio" % (self.name), self._get_tf_ratio(iteration), iteration)
		for key, val in self.summary_dict.items():
			if not isinstance(val, list):
				writer.add_scalar("train_%s/%s" % (self.name, key), val, iteration)
				self.summary_dict[key] = 0.0
			elif len(val) == 0:
				continue
			elif not isinstance(val[0], list):
				writer.add_scalar("train_%s/%s" % (self.name, key), mean(val), iteration)
				self.summary_dict[key] = list()
			else:
				val = [v for sublist in val for v in sublist]
				if len(val) == 0:
					continue
				writer.add_histogram("train_%s/%s" % (self.name, key), np.array(val), iteration)
				self.summary_dict[key] = list()
		

class ContextAwareParaphrasingTask(TaskTemplate):


	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix="", dataset_fun=DatasetHandler.load_Quora_Paraphrase_datasets):
		super(ContextAwareParaphrasingTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name="ContextAwareParaphrasing" + name_suffix, dataset_fun=dataset_fun)
		self.loss_module = self.model.get_loss_module()
		self.KL_scheduler = create_KLScheduler(scheduler_type = get_param_val(model_params, "VAE_scheduler", 1),
											   annealing_func_type = get_param_val(model_params, "VAE_annealing_func", 0),
											   loss_scaling = get_param_val(model_params, "VAE_loss_scaling", 1.0),
											   num_iters = get_param_val(model_params, "VAE_annealing_iters", 10000))
		self.cosine_loss_scaling = get_param_val(model_params, "cosine_loss_scaling", 0.0)
		self.switch_rate = get_param_val(model_params, "switch_rate", 0.8)
		self.summary_dict = {"loss_rec": list(), 
							 "loss_cosine": list(), 
							 "loss_VAE": list(), 
							 "KL_scheduler": list(),
							 "loss_combined": list(), 
							 "style_mu": list(), 
							 "style_sigma": list()}


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_fun(debug_dataset=self.debug)
		self.gen_batch = self.val_dataset.get_random_batch(16 if not self.debug else 2, toTorch=False, label_lengths=True, noun_mask=False, mask_prob=0.0)
		self.val_dataset.reset_index()
		self.id2word = get_id2word_dict()
		self.generated_before = False


	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		_, _, par_1_words, par_1_lengths, par_2_words, par_2_lengths = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True, label_lengths=True)
		par_1_words = par_1_words[DATA_GLOVE]
		par_1_lengths = par_1_lengths[DATA_GLOVE]

		current_tf_ratio = self._get_tf_ratio(iteration)

		par_1_res, par_2_res, par_1_style, par_2_style, par_semantics = self.model.contextless_paraphrasing(_input = (par_1_words, par_1_lengths, par_2_words, par_2_lengths, None, None, None, None), 
																											teacher_forcing = True, 
																											teacher_forcing_ratio = current_tf_ratio,
																											switch_rate = self.switch_rate)
		loss_1, loss_VAE_1, acc_1 = self._calculate_loss(par_1_res, par_1_words, par_1_style)
		loss_2, loss_VAE_2, acc_2 = self._calculate_loss(par_2_res, par_2_words, par_2_style)
		
		loss = (loss_1 + loss_2) / 2.0
		loss_VAE = (loss_VAE_1 + loss_VAE_2) / 2.0
		loss_cos = (1 - F.cosine_similarity(par_semantics[0], par_semantics[1], dim=-1)).mean()
		acc = (acc_1 + acc_2) / 2.0
		final_loss = loss + loss_VAE * self.KL_scheduler.get(iteration) + loss_cos * self.cosine_loss_scaling

		self.summary_dict["loss_rec"].append(loss.item())
		self.summary_dict["loss_cosine"].append(loss_cos.item())
		self.summary_dict["loss_VAE"].append(loss_VAE.item())
		self.summary_dict["KL_scheduler"] = [self.KL_scheduler.get(iteration)]
		self.summary_dict["loss_combined"].append(final_loss.item())
		for dict_key, hist_tensors in zip(["style_mu", "style_sigma"], [[par_1_style[1], par_2_style[1]], [par_1_style[2], par_2_style[2]]]):
			new_vals = [t.detach().cpu().contiguous().view(-1).numpy().tolist() for t in hist_tensors if t is not None]
			new_vals = [e for sublist in new_vals for e in sublist]
			self.summary_dict[dict_key].append(new_vals)
			while len(self.summary_dict[dict_key]) > 10:
				del self.summary_dict[dict_key][0]

		return final_loss, acc

	def _calculate_loss(self, par_res, batch_labels, par_style):
		par_word_dist, slot_dist, _, _ = par_res
		# Remove unknown word labels from the loss
		if (batch_labels[:,0] == get_SOS_index()).byte().all():
			batch_labels = batch_labels[:,1:]
		else:
			print("[#] WARNING: Batch labels were not shortend. First token ids: \n%s \nSOS index: %i" % (str(batch_labels[:,0]), get_SOS_index()))
		unknown_label = ((batch_labels == get_UNK_index()) | (batch_labels < 0)).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label

		## Loss reconstruction
		loss = self.loss_module(par_word_dist.view(-1, par_word_dist.shape[-1]), batch_labels.view(-1))
		## Accuracy calculation
		_, pred_labels = torch.max(par_word_dist, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / torch.sum(batch_labels != -1).float()
		## Loss VAE regularization
		_, style_mu, style_std = par_style
		loss_VAE = ContextAwareDialogueTask._calc_loss_VAE(style_mu, style_std)

		return loss, loss_VAE, acc

	def _eval_batch(self, batch, use_context_style=False):
		_, _, par_1_words, par_1_lengths, par_2_words, par_2_lengths = batch
		par_1_words = par_1_words[DATA_GLOVE]
		par_1_lengths = par_1_lengths[DATA_GLOVE]
		eval_swr = (1.0 if self.switch_rate > 0.0 else 0.0)
		p1_res, p2_res, _, _, _ = self.model.contextless_paraphrasing(_input = (par_1_words, par_1_lengths, par_2_words, par_2_lengths, None, None, None, None), 
																											teacher_forcing = True, 
																											teacher_forcing_ratio = 1.0,
																											switch_rate = eval_swr)
		p1_perplexity_probs, _, _, _ = p1_res
		p2_perplexity_probs, _, _, _ = p2_res
		p1_res_tf, p2_res_tf, _, _, _ = self.model.contextless_paraphrasing(_input = (par_1_words, par_1_lengths, par_2_words, par_2_lengths, None, None, None, None), 
																			teacher_forcing = False, 
																			teacher_forcing_ratio = 0.0,
																			switch_rate = eval_swr)
		_, _, p1_generated_words, p1_generated_lengths = p1_res_tf
		_, _, p2_generated_words, p2_generated_lengths = p2_res_tf

		p1_perplexity_probs = p1_perplexity_probs.detach()
		p1_generated_words = p1_generated_words.detach()
		p1_generated_lengths = p1_generated_lengths.detach()
		p2_perplexity_probs = p2_perplexity_probs.detach()
		p2_generated_words = p2_generated_words.detach()
		p2_generated_lengths = p2_generated_lengths.detach()

		# Remove unknown word labels from the evaluation
		batch_labels = par_1_words
		if (batch_labels[:,0] == get_SOS_index()).byte().all():
			batch_labels = batch_labels[:,1:]
		unknown_label = ((batch_labels == get_UNK_index()) | (batch_labels == -1)).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label

		return batch_labels, p1_perplexity_probs, p1_generated_words, p1_generated_lengths


	def eval(self, dataset=None, batch_size=64):
		# Default: if no dataset is specified, we use validation dataset
		if dataset is None:
			assert self.val_dataset is not None, "[!] ERROR: Validation dataset not loaded. Please load the dataset beforehand for evaluation."
			dataset = self.val_dataset

		self.model.eval()
		if not self.debug:
			batch_size = 128
		
		# Prepare metrics
		number_batches = int(math.ceil(dataset.get_num_examples() * 1.0 / batch_size))
		if self.debug:
			number_batches = min(8, number_batches)
		perplexity = []
		hypotheses, references = None, None

		# Evaluation loop
		for batch_ind in range(number_batches):
			if debug_level() == 0:
				print("Evaluation process: %4.2f%% (batch %i of %i)" % (100.0 * batch_ind / number_batches, batch_ind+1, number_batches), end="\r")
			# Evaluate single batch
			with torch.no_grad():
				batch = dataset.get_batch(batch_size, loop_dataset=False, toTorch=True, label_lengths=True, noun_mask=False, mask_prob=0.0)
				batch_labels, perplexity_logits, generated_words, generated_lengths = self._eval_batch(batch)
			# Perplexity calculation
			perplexity += TaskTemplate._eval_preplexity(perplexity_logits, batch_labels).cpu().numpy().tolist()

			hypotheses, references = add_if_not_none(TaskTemplate._preds_to_sents(batch_labels, generated_words, generated_lengths), (hypotheses, references))
			
		BLEU_score, prec_per_ngram = get_BLEU_score(hypotheses, references)
		ROUGE_score = get_ROUGE_score(hypotheses, references)
		# Metric output
		avg_perplexity = sum(perplexity) / len(perplexity)
		median_perplexity = median(perplexity)
		unigram_variety, unigram_entropy = get_diversity_measure(hypotheses, n_gram=1)
		bigram_variety, bigram_entropy = get_diversity_measure(hypotheses, n_gram=2)
		unigram_variety_gt, unigram_entropy_gt = get_diversity_measure(references, n_gram=1)
		bigram_variety_gt, bigram_entropy_gt = get_diversity_measure(references, n_gram=2)

		detailed_metrics = {
			"perplexity": avg_perplexity,
			"perplexity_median": median_perplexity,
			"diversity_unigram_entropy": unigram_entropy,
			"diversity_bigram_entropy": bigram_entropy,
			"diversity_unigram": unigram_variety,
			"diversity_bigram": bigram_variety,
			"diversity_unigram_entropy_gt": unigram_entropy_gt,
			"diversity_bigram_entropy_gt": bigram_entropy_gt,
			"diversity_unigram_gt": unigram_variety_gt,
			"diversity_bigram_gt": bigram_variety_gt,
			"BLEU": BLEU_score
		} 
		for n in range(len(prec_per_ngram)):
			detailed_metrics["BLEU_%i-gram" % (n+1)] = float(prec_per_ngram[n])
		for metric, results in ROUGE_score.items():
			if metric[-1] in ["1", "2", "3", "4"]:
				continue
			for sub_category, val in results.items():
				detailed_metrics[metric + "_" + sub_category] = val 

		self.model.train()
		dataset.reset_index()
		
		return BLEU_score, detailed_metrics


	def add_summary(self, writer, iteration):
		# TODO: Add some example generations here. Either run the model again for some random sentences, or save last training sentences
		writer.add_scalar("train_%s/teacher_forcing_ratio" % (self.name), self._get_tf_ratio(iteration), iteration)
		for key, val in self.summary_dict.items():
			if not isinstance(val, list):
				writer.add_scalar("train_%s/%s" % (self.name, key), val, iteration)
				self.summary_dict[key] = 0.0
			elif len(val) == 0:
				continue
			elif not isinstance(val[0], list):
				writer.add_scalar("train_%s/%s" % (self.name, key), mean(val), iteration)
				self.summary_dict[key] = list()
			else:
				val = [v for sublist in val for v in sublist]
				if len(val) == 0:
					continue
				writer.add_histogram("train_%s/%s" % (self.name, key), np.array(val), iteration)
				self.summary_dict[key] = list()

		if self.debug or iteration % 1000 == 0:
			gen_list = self.generate_random_style_samples()
			for i in range(len(gen_list)):
				if not self.generated_before:
					writer.add_text(self.name + "_samp%i_input_phrase" % (i), gen_list[i][0], iteration)
				for j in range(len(gen_list[i][1])):
					writer.add_text(self.name + "_samp%i_sample_%i" % (i, j), gen_list[i][1][j], iteration)
			self.generated_before = True


	def generate_random_style_samples(self):
		self.model.eval()
		# 1.) Put data on GPU
		batch_torch = UnsupervisedTask.batch_to_torch(self.gen_batch)
		_, _, par_words, par_lengths, _, _ = batch_torch
		par_words = par_words[DATA_GLOVE]
		par_lengths = par_lengths[DATA_GLOVE]
		par_masks = self.model.embedding_module.generate_mask(par_words)
		with torch.no_grad():
			_, _, gen_par_words, gen_par_lengths = self.model.sample_reconstruction_styles((par_words, par_lengths, par_masks, None, None), num_samples=12)
		del batch_torch
		# 3.) Reconstruct generated answer and input
		generated_paraphrases = list()
		input_phrases = list()

		gen_par_words = gen_par_words.cpu().numpy()
		gen_par_lengths = gen_par_lengths.cpu().numpy()

		par_words = self.gen_batch[2][DATA_GLOVE]
		par_lengths = self.gen_batch[3][DATA_GLOVE]

		for embeds, lengths, list_to_add, add_sents_up in zip([par_words, gen_par_words],
															  [par_lengths, gen_par_lengths],
															  [input_phrases, generated_paraphrases],
															  [True, False]):
			reconstruct_sentences(embeds, lengths, list_to_add=list_to_add, add_sents_up=add_sents_up)

		# 5.) Put everything in a nice format
		gen_list = list(zip(input_phrases, generated_paraphrases))
		self.model.train()
		return gen_list
		
