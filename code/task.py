import torch 
import torch.nn as nn
import numpy as np
import math
from random import shuffle, random
import os
import sys

from data import reconstruct_sentences, DatasetTemplate, DatasetHandler, debug_level, DATA_GLOVE, DATA_BERT
from model_utils import get_device, get_param_val
from vocab import get_id2word_dict, get_UNK_index, get_EOS_index
from metrics import get_BLEU_batch_stats, get_BLEU_score




class TaskTemplate:

	def __init__(self, model, model_params, name, load_data=True, debug=False, dataset_fun=None):
		self.name = name 
		self.model = model
		self.model_params = model_params
		self.debug = debug
		self.dataset_fun = dataset_fun
		self.loss_module = None
		self.train_dataset = None 
		self.val_dataset = None 
		self.test_dataset = None
		self.teacher_forcing_ratio = get_param_val(model_params, "teacher_forcing_ratio", 1.0)
		self.teacher_forcing_annealing = get_param_val(model_params, "teacher_forcing_annealing", -1)
		if load_data:
			self._load_datasets()


	def _get_tf_ratio(self, iteration):
		if self.teacher_forcing_annealing > 0:
			current_tf_ratio = self.teacher_forcing_ratio ** (1.0 + iteration/self.teacher_forcing_annealing)
		else:
			current_tf_ratio = self.teacher_forcing_ratio
		return current_tf_ratio


	def _load_datasets(self):
		raise NotImplementedError


	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		# Function to perform single step given the batch size; returns the loss and the accuracy of the batch
		raise NotImplementedError


	def _eval_batch(self, batch):
		raise NotImplementedError


	def eval(self, dataset=None, batch_size=64, label_lengths=False, noun_mask=False):
		# Default: if no dataset is specified, we use validation dataset
		if dataset is None:
			assert self.val_dataset is not None, "[!] ERROR: Validation dataset not loaded. Please load the dataset beforehand for evaluation."
			dataset = self.val_dataset

		self.model.eval()
		
		# Prepare metrics
		number_batches = int(math.ceil(dataset.get_num_examples() * 1.0 / batch_size))
		perplexity = []
		diversity_unigram, diversity_bigram = None, None

		# Evaluation loop
		for batch_ind in range(number_batches):
			if debug_level() == 0:
				print("Evaluation process: %4.2f%%" % (100.0 * batch_ind / number_batches), end="\r")
			# Evaluate single batch
			batch = dataset.get_batch(batch_size, loop_dataset=False, toTorch=True, label_lengths=label_lengths, noun_mask=noun_mask, mask_prob=0.0)
			batch_labels, perplexity_logits, generated_words, generated_lengths = self._eval_batch(batch)
			# Perplexity calculation
			perplexity += TaskTemplate._eval_preplexity(perplexity_logits, batch_labels).cpu().numpy().tolist()
			loc_div_uni, loc_div_bi = TaskTemplate._eval_diversity(generated_words, generated_lengths, num_classes=perplexity_logits.shape[-1])
			if diversity_unigram is None or diversity_bigram is None:
				diversity_unigram, diversity_bigram = loc_div_uni, loc_div_bi
			else:
				diversity_unigram += loc_div_uni
				diversity_bigram += loc_div_bi

		diversity_unigram = diversity_unigram.cpu().numpy()
		diversity_bigram = diversity_bigram.cpu().numpy()
		# Metric output
		avg_perplexity = sum(perplexity) / len(perplexity)
		div_uni_probs = diversity_unigram / max(np.sum(diversity_unigram), 1e-5)
		div_bi_probs = diversity_bigram / max(np.sum(diversity_bigram), 1e-5)
		unigram_entropy = - (div_uni_probs * np.log(np.maximum(div_uni_probs, 1e-10))).sum()
		bigram_entropy = - (div_bi_probs * np.log(np.maximum(div_bi_probs, 1e-10))).sum()
		unigram_variety = int(np.sum(diversity_unigram > 0))
		bigram_variety = int(np.sum(diversity_bigram > 0))

		detailed_metrics = {
			"perplexity": avg_perplexity,
			"unigram_entropy": unigram_entropy,
			"bigram_entropy": bigram_entropy,
			"unigram_variety": unigram_variety,
			"bigram_variety": bigram_variety
		} 

		self.model.train()
		
		return avg_perplexity, detailed_metrics


	def add_to_summary(self, writer, iteration):
		pass


	def eval_metric(self, eval_dict):
		return eval_dict["perplexity"]


	def finalize_summary(self, writer, iteration, checkpoint_path):
		pass


	def export_best_results(self, checkpoint_path, iteration):
		pass


	@staticmethod
	def _eval_preplexity(perplexity_logits, batch_labels):
		valid_labels = (batch_labels >= 0).float()
		prob_logits = torch.gather(perplexity_logits, -1, torch.max(batch_labels[:,:,None], other=batch_labels.new_zeros(size=(batch_labels.shape[0], batch_labels.shape[1], 1)))).squeeze(-1) - perplexity_logits.exp().sum(dim=-1).log()
		prob_logits = valid_labels * prob_logits # Set probability to 1 (or logit to 0) for not used characters
		prob_logits = prob_logits.sum(dim=1) # Sum over sequence length
		prob_logits = prob_logits * (-1 / valid_labels.sum(dim=-1)) # Divide by 
		prob_vals = prob_logits.exp() # Convert logits to actual probability values
		return prob_vals

	@staticmethod
	def _eval_diversity(generated_words, generated_lengths, num_classes=0):
		word_positions = torch.arange(start=0, end=generated_words.shape[1], dtype=generated_lengths.dtype, device=generated_lengths.device)
		mask = ((word_positions.reshape(shape=[1, -1]) < generated_lengths.reshape([-1, 1])) & (generated_words >= 0)).long()
		
		generated_words[generated_words < 0] = 0
		diversity_unigram = generated_words.view(-1).bincount(minlength=num_classes, weights=mask.view(-1))
		diversity_bigram = diversity_unigram * 0 # TODO: Implement bigram diversity

		return diversity_unigram, diversity_bigram

	@staticmethod
	def _preds_to_sents(batch_labels, generated_words, generated_lengths):
		generated_words = generated_words.cpu().numpy()
		generated_lengths = generated_lengths.cpu().numpy()
		batch_labels = batch_labels.cpu().numpy()
		batch_labels[batch_labels == -1] = get_UNK_index()
		batch_lengths = ((get_EOS_index() == batch_labels) * np.arange(start=1, stop=batch_labels.shape[1]+1)[None,:]).sum(axis=-1)
		
		generated_sentences = reconstruct_sentences(generated_words, generated_lengths, add_sents_up=False, make_pretty=False)
		ground_truth_sentences = reconstruct_sentences(batch_labels, batch_lengths, add_sents_up=False, make_pretty=False)

		return generated_sentences, ground_truth_sentences

	@staticmethod
	def _reconst_sents(generated_words, generated_lengths):
		generated_words = generated_words.cpu().numpy()
		generated_lengths = generated_lengths.cpu().numpy()
		
		generated_sentences = reconstruct_sentences(generated_words, generated_lengths, add_sents_up=False, make_pretty=False)
		generated_sentences = [[[w for w in s.split(" ") if (w.replace(" ","") != "")] for s in sent_set] for sent_set in generated_sentences]

		return generated_sentences



