import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import math
from random import shuffle, random
import os
import sys
# Disable matplotlib screen support
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from statistics import mean, median

from data import DatasetTemplate, DatasetHandler, debug_level, DATA_GLOVE, DATA_BERT, reconstruct_sentences
from model_utils import get_device, get_param_val
from unsupervised_models.model_loss import LossStyleModule, LossStyleSimilarityModule
from vocab import get_id2word_dict, get_UNK_index, get_SOS_index
from task import TaskTemplate
from scheduler_annealing_KL import create_KLScheduler
from metrics import get_BLEU_batch_stats, get_BLEU_score, get_ROUGE_score, euclidean_distance
from mutils import add_if_not_none
from unsupervised_task import UnsupervisedTask


#########################
## TASK SPECIFIC TASKS ##
#########################

class DiscriminatorTask(TaskTemplate):


	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix="", dataset_fun=DatasetHandler.load_Dialogue_Paraphrase_datasets):
		super(DiscriminatorTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name="DiscriminatorTask" + name_suffix, dataset_fun=dataset_fun)
		self.loss_module = self.model.get_loss_module()
		self.use_VAE = get_param_val(model_params, "use_VAE", False)
		self.use_semantic_specific_attn = get_param_val(model_params, "use_semantic_specific_attn", False)
		self.summary_dict = {"loss_discriminator": list(),
							 "loss_VAE": list(),
							 "acc_discriminator": list()}


	def _load_datasets(self):
		self.train_dataset, self.val_dataset, self.test_dataset = self.dataset_fun(debug_dataset=self.debug, num_context_turns=get_param_val(self.model_params, "num_context_turns", 2))
		self.gen_batch = self.val_dataset.get_random_batch(8, toTorch=False, label_lengths=True, noun_mask=True, mask_prob=0.0)
		self.val_dataset.reset_index()


	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		# if iteration == 10000:
		# 	self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_Dialogue_Paraphrase_Small_datasets(debug_dataset=self.debug, num_context_turns=get_param_val(self.model_params, "num_context_turns", 2))
		
		train_batch = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)

		discriminator_predictions, labels, context_1_style, context_2_style = self.model(_input=train_batch, use_VAE=self.use_VAE, use_semantic_specific_attn=self.use_semantic_specific_attn)
		loss_disc = self.loss_module(discriminator_predictions, labels)
		acc = ((discriminator_predictions > 0.5).float() == labels).float().mean()
		loss_VAE = (self._get_VAE_loss(context_1_style) + self._get_VAE_loss(context_2_style)) / 2.0

		loss = loss_disc
		if self.use_VAE:
			loss += 0.1 * loss_VAE
		
		self.summary_dict["loss_discriminator"].append(loss_disc.item())
		self.summary_dict["loss_VAE"].append(loss_VAE.item())
		self.summary_dict["acc_discriminator"].append(acc.item())

		return loss, acc


	def _get_VAE_loss(self, style_tuple):
		_, style_mu, style_std = style_tuple
		return torch.mean(- torch.log(style_std) + (style_std ** 2 - 1 + style_mu ** 2) / 2)


	def _eval_batch(self, batch):
		
		discriminator_predictions, labels, _, _ = self.model(_input=batch, use_VAE=self.use_VAE, use_semantic_specific_attn=self.use_semantic_specific_attn)
		loss = self.loss_module(discriminator_predictions, labels)
		discrete_preds = (discriminator_predictions > 0.5).float()
		TP = ((discrete_preds == 1) & (labels == 1)).float().sum()
		FP = ((discrete_preds == 1) & (labels == 0)).float().sum()
		FN = ((discrete_preds == 0) & (labels == 1)).float().sum()
		TN = ((discrete_preds == 0) & (labels == 0)).float().sum()
		acc = (discrete_preds == labels).float().mean()
		recall = TP / (TP + FN + 1e-5)
		precision = TP / (TP + FP + 1e-5)
		f1 = 2 * recall * precision / (recall + precision + 1e-5)

		metric_dict = {
			"eval_acc": acc,
			"eval_recall": recall,
			"eval_precision": precision,
			"eval_f1": f1
		}

		return loss, metric_dict


	def eval(self, dataset=None, batch_size=64):
		if dataset is None:
			assert self.val_dataset is not None, "[!] ERROR: Validation dataset not loaded. Please load the dataset beforehand for evaluation."
			dataset = self.val_dataset

		self.model.eval()
		
		# Prepare metrics
		number_batches = int(math.ceil(dataset.get_num_examples() * 1.0 / batch_size))
		eval_metrics = None
		eval_loss = []
		num_counter = []

		# Evaluation loop
		with torch.no_grad():
			for batch_ind in range(number_batches):
				if debug_level() == 0:
					print("Evaluation process: %4.2f%%" % (100.0 * batch_ind / number_batches), end="\r")
				# Evaluate single batch
				batch = dataset.get_batch(batch_size, loop_dataset=False, toTorch=True)
				batch_loss, additional_metrics = self._eval_batch(batch)
				if eval_metrics is None:
					eval_metrics = {metric_name: [metric_val.item()] for metric_name, metric_val in additional_metrics.items()}
				else:
					[eval_metrics[metric_name].append(metric_val.item()) for metric_name, metric_val in additional_metrics.items()]
				eval_loss.append(batch_loss.item())
				num_counter.append(batch[0].size(0))

		mean_loss = sum([n * l for n, l in zip(num_counter, eval_loss)]) / sum(num_counter)
		detailed_metrics = {metric_name: sum([n * l for n, l in zip(num_counter, metric_vals)]) / sum(num_counter) for metric_name, metric_vals in eval_metrics.items()}

		detailed_metrics["eval_loss"] = mean_loss
		

		self.model.train()
		
		return mean_loss, detailed_metrics


	def add_summary(self, writer, iteration):
		# TODO: Add some example generations here. Either run the model again for some random sentences, or save last training sentences
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

		if iteration % 1000 == 0:
			self.visualize_attentions(writer, iteration)


	def visualize_attentions(self, writer, iteration):
		self.model.eval()
		# 1.) Put data on GPU
		batch_torch = UnsupervisedTask.batch_to_torch(self.gen_batch)
		# 2.) Push data through network
		with torch.no_grad():
			disc_preds, labels, _, _, context_attn_style_1, context_attn_style_2, par_1_attn_semantic, par_2_attn_semantic = self.model(_input = batch_torch, 
																																		use_semantic_specific_attn = self.use_semantic_specific_attn,
																																		debug = True)
		
		del batch_torch

		context_attn_style_1 = context_attn_style_1[1].cpu().numpy()
		context_attn_style_2 = context_attn_style_2[1].cpu().numpy()
		par_1_attn_semantic = par_1_attn_semantic[1].cpu().numpy()
		par_2_attn_semantic = par_2_attn_semantic[1].cpu().numpy()
		par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = self.gen_batch

		fig = plt.figure()

		par_1_sents = reconstruct_sentences(par_1_words, par_1_lengths, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths, add_sents_up=False, make_pretty=False)
		par_2_sents = reconstruct_sentences(par_2_words, par_2_lengths, slot_vals=par_2_slots, slot_lengths=par_2_slot_lengths, add_sents_up=False, make_pretty=False)
		context_1_sents = reconstruct_sentences(contexts_1_words, contexts_1_lengths, add_sents_up=False, make_pretty=False)
		context_2_sents = reconstruct_sentences(contexts_2_words, contexts_2_lengths, add_sents_up=False, make_pretty=False)
		context_1_sents = [[s.strip().split(" ") for s in con] for con in context_1_sents]
		context_2_sents = [[s.strip().split(" ") for s in con] for con in context_2_sents]

		for sent, sent_length, sent_attn, sent_name in zip([par_1_sents, par_2_sents, context_1_sents, context_2_sents], 
														   [par_1_lengths, par_2_lengths, contexts_1_lengths, contexts_2_lengths], 
														   [par_1_attn_semantic, par_2_attn_semantic, context_attn_style_1, context_attn_style_2],
														   ["sent_1", "sent_2", "context_1", "context_2"]):
			if len(sent_length.shape) == 1:
				sent = [[s] for s in sent]
				sent_length = sent_length[:,None]
				sent_attn = sent_attn[:,None,:]

			for batch_index in range(len(par_1_sents)):

				fig = plt.figure()
				for i in range(sent_length.shape[1]):
					ax = fig.add_subplot(sent_length.shape[1], 1, i+1)
					if sent_length[batch_index, i] <= 0:
						continue
					sent_attention_map = sent_attn[batch_index,i:i+1,:sent_length[batch_index,i]]
					cax = ax.matshow(sent_attention_map, cmap=plt.cm.gray)
					ax.set_yticklabels(["Attention (max %.2f)" % (np.max(sent_attention_map))])
					ax.set_xticklabels(sent[batch_index][i], rotation=90)
					ax.set_yticks(range(1))
					ax.set_xticks(range(sent_length[batch_index,i]))
					ax.set_yticks(np.arange(-.5, 1, 1), minor=True)
					ax.set_xticks(np.arange(-.5, len(sent[batch_index][i]), 1), minor=True)
					ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
				
				plt.tight_layout()
				writer.add_figure(tag="train/%s_%i_%s_attention" % (self.name, batch_index, sent_name), figure=fig, global_step=iteration)
		plt.close()

		self.model.train()


	def export_best_results(self, checkpoint_path):
		self.model.eval()
		
		# Prepare metrics
		batch_size = 64
		number_batches = int(math.ceil(self.val_dataset.get_num_examples() * 1.0 / batch_size))
		number_batches = min(5, number_batches)
		true_positive_sents = list()
		false_positive_sents = list()
		true_negative_sents = list()
		false_negative_sents = list()

		# Evaluation loop
		with torch.no_grad():
			for batch_ind in range(number_batches):
				if debug_level() == 0:
					print("Evaluation process: %4.2f%%" % (100.0 * batch_ind / number_batches), end="\r")
				# Evaluate single batch
				batch = self.val_dataset.get_batch(batch_size, loop_dataset=False, toTorch=True)
				discriminator_predictions, labels, _, _ = self.model(_input=batch, use_VAE=self.use_VAE, use_semantic_specific_attn=self.use_semantic_specific_attn)
				positive_predictions = (discriminator_predictions > 0.5).float().cpu().numpy()
				labels = labels.cpu().numpy()

				batch = tuple([tensor.cpu().numpy() for tensor in batch])
				par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = batch
				reconstructed_sents_1 = reconstruct_sentences(par_1_words, par_1_lengths, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths)
				reconstructed_sents_2 = reconstruct_sentences(par_2_words, par_2_lengths, slot_vals=par_2_slots, slot_lengths=par_2_slot_lengths)
				reconstructed_contexts_1 = reconstruct_sentences(contexts_1_words, contexts_1_lengths)
				reconstructed_contexts_2 = reconstruct_sentences(contexts_2_words, contexts_2_lengths)

				loc_batch_size = par_1_words.shape[0]

				for b in range(positive_predictions.shape[0]):
					semantic_sents = reconstructed_sents_1[b%loc_batch_size] if b < loc_batch_size or b >= loc_batch_size*3 else reconstructed_sents_2[b%loc_batch_size]
					context_sents = reconstructed_contexts_1[b%loc_batch_size] if b < loc_batch_size*2 else reconstructed_contexts_2[b%loc_batch_size]
					s = "\n" + "="*100 + "\n" + context_sents + "\n" + "-"*100 + "\nResponse: " + semantic_sents + "\n" + "="*100 + "\n"
					if positive_predictions[b] == 1 and labels[b] == 1:
						true_positive_sents.append(s)
					elif positive_predictions[b] == 1 and labels[b] == 0:
						false_positive_sents.append(s)
					elif positive_predictions[b] == 0 and labels[b] == 1:
						false_negative_sents.append(s)
					elif positive_predictions[b] == 0 and labels[b] == 0:
						true_negative_sents.append(s)
					else:
						print("[!] ERROR: Something went wrong. Prediction is not any of TP, FP, FN, and TN...")
						sys.exit(1)

		for sents, filename in zip([true_positive_sents, false_positive_sents, false_negative_sents, true_negative_sents],
									["true_positives", "false_positives", "false_negatives", "true_negatives"]):
			sents = list(set(sents))
			with open(os.path.join(checkpoint_path, "%s_%s.txt" % (self.name, filename)), "w") as f:
				f.write("\n".join(sents))

		self.model.train()

