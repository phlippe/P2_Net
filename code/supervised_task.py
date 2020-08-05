import torch 
import torch.nn as nn
import argparse
import numpy as np
import math
from random import shuffle, random
import os
import sys

from data import DatasetTemplate, DatasetHandler, debug_level, DATA_GLOVE, DATA_BERT
from model_utils import get_device, get_param_val
from vocab import get_id2word_dict, get_UNK_index
from task import TaskTemplate


#########################
## TASK SPECIFIC TASKS ##
#########################

class ParaphraseTask(TaskTemplate):


	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix=""):
		super(ParaphraseTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name="ParaphraseTask" + name_suffix)
		self.loss_module = self.model.get_loss_module()


	def _load_datasets(self):
		self._get_datasets_from_handler()
		self.gen_batch = self.val_dataset.get_random_batch(8, toTorch=False )
		self.val_dataset.reset_index()
		self.id2word = get_id2word_dict()
		self.generated_before = False


	def _get_datasets_from_handler(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_AH_Paraphrase_datasets()


	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		# TODO: Implement
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		dialog_embeds, dialog_lengths, template_embeds, template_lengths, batch_labels = self.train_dataset.get_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)
		
		current_tf_ratio = self._get_tf_ratio(iteration)

		if random() < current_tf_ratio:
			paraphrases_word_dist, _, _ = self.model((dialog_embeds, dialog_lengths, template_embeds, template_lengths), labels=batch_labels)
		else:
			paraphrases_word_dist, _, _ = self.model((dialog_embeds, dialog_lengths, template_embeds, template_lengths), labels=None, beams=1, min_generation_steps=batch_labels.size(1), max_generation_steps=batch_labels.size(1))

		# Remove unknown word labels from the loss
		unknown_label = (batch_labels == get_UNK_index()).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label

		loss = self.loss_module(paraphrases_word_dist.view(-1, paraphrases_word_dist.shape[-1]), batch_labels.view(-1))
		_, pred_labels = torch.max(paraphrases_word_dist, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / torch.sum(batch_labels != -1).float()

		return loss, acc


	def _eval_batch(self, batch):
		dialog_embeds, dialog_lengths, template_embeds, template_lengths, batch_labels = batch
		perplexity_probs, _, _ = self.model((dialog_embeds, dialog_lengths, template_embeds, template_lengths), labels=batch_labels)
		_, generated_words, generated_lengths = self.model((dialog_embeds, dialog_lengths, template_embeds, template_lengths), labels=None, beams=1)
		
		perplexity_probs = perplexity_probs.detach()
		generated_words = generated_words.detach()
		generated_lengths = generated_lengths.detach()

		# Remove unknown word labels from the evaluation
		unknown_label = (batch_labels == get_UNK_index()).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label

		return batch_labels, perplexity_probs, generated_words, generated_lengths 


	def eval(self, dataset=None, batch_size=64):
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
			batch = dataset.get_batch(batch_size, loop_dataset=False, toTorch=True)
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


	def add_summary(self, writer, iteration):
		# TODO: Add some example generations here. Either run the model again for some random sentences, or save last training sentences
		writer.add_scalar("train_%s/teacher_forcing_ratio" % (self.name), self._get_tf_ratio(iteration), iteration)

		if iteration % 1000 == 0:
			gen_list = self.generate_examples()
			for i in range(len(gen_list)):
				if not self.generated_before:
					writer.add_text(self.name + "_%i_input_dialogue" % (i), gen_list[i][0], iteration)
					writer.add_text(self.name + "_%i_input_template" % (i), gen_list[i][1], iteration)
					writer.add_text(self.name + "_%i_input_labels" % (i), gen_list[i][2], iteration)
				if isinstance(gen_list[i][3], list):
					for sent_index in range(len(gen_list[i][3])):
						writer.add_text(self.name + "_%i_generated_paraphrase_%i" % (i, sent_index), gen_list[i][3][sent_index], iteration)
				else:
					writer.add_text(self.name + "_%i_generated_paraphrase" % (i), gen_list[i][3], iteration)
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
		batch_torch = PretrainingTask.batch_to_torch(self.gen_batch)
		# 2.) Push data through network
		dialog_embeds, dialog_lengths, template_embeds, template_lengths, batch_labels = batch_torch
		with torch.no_grad():
			paraphrases_word_dist, paraphrase_word_preds, paraphrase_lengths = self.model((dialog_embeds, dialog_lengths, template_embeds, template_lengths), labels=None, beams=5, max_generation_steps=30)
		del batch_torch
		# 3.) Reconstruct generated answer and input
		generated_paraphrases = list()
		input_dialogues = list()
		input_templates = list()
		input_labels = list()

		paraphrase_words = paraphrase_word_preds.cpu().numpy()
		paraphrase_lengths = paraphrase_lengths.cpu().numpy()

		dialog_embeds = self.gen_batch[0][DATA_GLOVE]
		dialog_lengths = self.gen_batch[1][DATA_GLOVE]
		template_embeds = self.gen_batch[2][DATA_GLOVE]
		template_lengths = self.gen_batch[3][DATA_GLOVE]
		batch_labels = self.gen_batch[4]
		batch_lengths = (batch_labels != -1).sum(axis=-1)

		for embeds, lengths, list_to_add in zip([paraphrase_words, dialog_embeds, template_embeds, batch_labels],
												[paraphrase_lengths, dialog_lengths, template_lengths, batch_lengths],
												[generated_paraphrases, input_dialogues, input_templates, input_labels]):
			for batch_index in range(embeds.shape[0]):
				p_words = list()
				if len(lengths.shape) == 1:
					for word_index in range(lengths[batch_index]):
						p_words.append(self.id2word[embeds[batch_index, word_index]])
					sents = " ".join(p_words)
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
		gen_list = list(zip(input_dialogues, input_templates, input_labels, generated_paraphrases))
		self.model.train()
		return gen_list


class LanguageModelingTask(ParaphraseTask):

	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix=""):
		super(LanguageModelingTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug)
		self.name = "LanguageModeling" + name_suffix

	def _get_datasets_from_handler(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_LM_Wikitext_datasets()


class MicrosoftParaphraseTask(ParaphraseTask):

	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix=""):
		super(MicrosoftParaphraseTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug)
		self.name = "MicrosoftParaphraseTask" + name_suffix

	def _get_datasets_from_handler(self):
		self.train_dataset, self.val_dataset, self.test_dataset = DatasetHandler.load_Microsoft_Paraphrase_datasets()


class PretrainingTask(ParaphraseTask):


	def __init__(self, model, model_params, load_data=True, debug=False, name_suffix=""):
		super(PretrainingTask, self).__init__(model=model, model_params=model_params, load_data=load_data, debug=debug, name_suffix="")
		self.name = "PretrainingTask" + name_suffix
		self.loss_module = self.model.get_loss_module(reduction='none')


	def _get_datasets_from_handler(self):
		self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
		all_loading_methods = [
				('LM', DatasetHandler.load_LM_Wikitext_datasets),
				('LM', DatasetHandler.load_LM_Book_datasets),
				('LM', DatasetHandler.load_LM_Dialogue_datasets),
				('Par', DatasetHandler.load_Microsoft_Paraphrase_datasets),
				('Par', DatasetHandler.load_Microsoft_Video_Description_datasets)
			]
		for _, load_method in all_loading_methods:
			train_data, val_data, test_data = load_method(debug_dataset=self.debug)
			self.train_dataset.append(train_data)
			self.val_dataset.append(val_data)
			self.test_dataset.append(test_data)

		train_LM = 0.6
		train_Par = 1.0 - train_LM

		num_LM_examples = sum([d.get_num_examples() for d_index, d in enumerate(self.train_dataset) if all_loading_methods[d_index][0] == 'LM'])
		self.train_proportions = []
		for i in range(len(self.train_dataset)):
			if all_loading_methods[i][0] == 'LM':
				weight_factor = train_LM * self.train_dataset[i].get_num_examples() / num_LM_examples
			else:
				weight_factor = train_Par * 1.0 / sum([1 for dtype, _ in all_loading_methods if dtype == 'Par'])
			self.train_proportions.append(weight_factor)

		self.train_index = 0
		self.train_permut_order = []
		for dataset_index, p in enumerate(self.train_proportions):
			self.train_permut_order += [dataset_index]*int(p * 5000)
		shuffle(self.train_permut_order)

		self.avg_loss_per_dataset = np.zeros(shape=(len(self.train_dataset),2), dtype=np.float32) # Stores sum of losses *and* number of losses/examples


	def _load_datasets(self):
		self._get_datasets_from_handler()
		self.gen_batch, self.gen_batch_datasets = self.stack_batch([val_data.get_random_batch(2, toTorch=False ) for val_data in self.val_dataset])
		[val_data.reset_index() for val_data in self.val_dataset]
		self.id2word = get_id2word_dict()
		self.generated_before = False


	def stack_batch(self, batch):
		stacked_batch = list()
		batch_size = None
		for element_index in range(len(batch[0])):
			if isinstance(batch[0][element_index], dict):
				combined_elements = dict()
				for key in batch[0][element_index].keys():
					combined_elements[key] = PretrainingTask._stack_tensor_list([b[element_index][key] for b in batch])
					batch_size = combined_elements[key].shape[0] if combined_elements[key] is not None else batch_size
			elif isinstance(batch[0][element_index], tuple):
				batch_elements = [[b[element_index][i] for b in sub_batch] for i in range(len(batch[0][element_index]))]
				combined_elements = [PretrainingTask._stack_tensor_list(sub_batch) for sub_batch in batch_elements]
				combined_elements = tuple(combined_elements)
				batch_size = combined_elements[0].shape[0] if combined_elements[0] is not None else batch_size
			else:
				batch_elements = [b[element_index] for b in batch]
				combined_elements = PretrainingTask._stack_tensor_list(batch_elements)
				batch_size = combined_elements.shape[0]  if combined_elements is not None else batch_size
			stacked_batch.append(combined_elements)

		dataset_assignment = np.zeros(shape=(batch_size,), dtype=np.int32)
		s_index = 0
		for b_index, b in enumerate(batch):
			s_index += b[-1].shape[0]
			dataset_assignment[s_index:] += 1

		return tuple(stacked_batch), dataset_assignment


	@staticmethod
	def _stack_tensor_list(batch_elements):
		if isinstance(batch_elements[0], np.ndarray): # Numpy array
			if len(batch_elements[0].shape) > 1: # Contain sequence length or not
				max_length = max([b.shape[1] for b in batch_elements])
				batch_elements = [np.concatenate([b, np.zeros(shape=(b.shape[0], max_length - b.shape[1]), dtype=np.int32)-1], axis=1) for b in batch_elements]
			combined_elements = np.concatenate([b for b in batch_elements], axis=0)
		elif isinstance(batch_elements[0], torch.Tensor): # Torch tensor
			# print([b.shape for b in batch_elements]) # Print statement for debugging if needed
			if max([len(b.shape) for b in batch_elements]) > 1:
				max_length = max([b.shape[1] for b in batch_elements if len(b.shape) > 1])
				batch_elements = [torch.cat([b, b.new_zeros(size=(b.shape[0], max_length - b.shape[1]))-1], dim=1) if (len(b.shape) > 1 and b.shape[1] < max_length and b.shape[0] > 0) else b for b in batch_elements]
			combined_elements = torch.cat([b for b in batch_elements if b.shape[0] > 0], dim=0)
		else:
			combined_elements = None
		return combined_elements


	def get_train_batch(self, batch_size, loop_dataset, toTorch):
		# Determine how many example should be used from each dataset
		batch_dataset_indices = []
		for _ in range(batch_size):
			batch_dataset_indices.append(self.train_permut_order[self.train_index])
			self.train_index += 1
			if self.train_index >= len(self.train_permut_order):
				self.train_index = 0
				shuffle(self.train_permut_order)
		batch_sizes = [batch_dataset_indices.count(i) for i in range(len(self.train_dataset))]
		# Load batches from each dataset and stack those
		dataset_batches = [train_data.get_batch(batch_sizes[i], loop_dataset=loop_dataset, toTorch=toTorch ) for i, train_data in enumerate(self.train_dataset)]
		training_batch, dataset_assignment = self.stack_batch(dataset_batches)
		return training_batch, dataset_assignment


	def train_step(self, batch_size, loop_dataset=True, iteration=0):
		assert self.train_dataset is not None, "[!] ERROR: Training dataset not loaded. Please load the dataset beforehand for training."
		
		batch_train, dataset_assignment = self.get_train_batch(batch_size, loop_dataset=loop_dataset, toTorch=True)
		dialog_embeds, dialog_lengths, template_embeds, template_lengths, batch_labels = batch_train

		current_tf_ratio = self._get_tf_ratio(iteration)

		if random() < current_tf_ratio:
			paraphrases_word_dist, _, _ = self.model((dialog_embeds, dialog_lengths, template_embeds, template_lengths), labels=batch_labels)
		else:
			paraphrases_word_dist, _, _ = self.model((dialog_embeds, dialog_lengths, template_embeds, template_lengths), labels=None, beams=1, min_generation_steps=batch_labels.size(1), max_generation_steps=batch_labels.size(1))

		# Remove unknown word labels from the loss
		unknown_label = (batch_labels == get_UNK_index()).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label
		# We do not average anymore over all valid labels directly because we want to record loss per task
		elementwise_loss = self.loss_module(paraphrases_word_dist.view(-1, paraphrases_word_dist.shape[-1]), batch_labels.view(-1)).view(*batch_labels.shape)
		elementwise_loss[torch.isnan(elementwise_loss)] = 0
		valid_labels = (batch_labels >= 0).float()
		loss = (elementwise_loss * valid_labels).sum() / (valid_labels.sum() + 1e-10)
		_, pred_labels = torch.max(paraphrases_word_dist, dim=-1)
		acc = torch.sum(pred_labels == batch_labels).float() / torch.sum(batch_labels != -1).float()
		if torch.isnan(loss):
			print("Elementwise loss: " + str(elementwise_loss))
			print("Valid labels: " + str(valid_labels))
			print("Batch labels: " + str(batch_labels))
			print("Predictions: " + str(pred_labels))
		# Determine per batch loss
		elementwise_loss = elementwise_loss.detach()
		loss_per_sentence = ((elementwise_loss * valid_labels).sum(dim=-1) / valid_labels.sum(dim=-1)).cpu().numpy()
		for data_index in range(len(self.train_dataset)):
			dataset_mask = (dataset_assignment == data_index)
			self.avg_loss_per_dataset[data_index,0] += np.sum(loss_per_sentence * dataset_mask) 
			self.avg_loss_per_dataset[data_index,1] += np.sum(dataset_mask)

		return loss, acc


	def eval(self, dataset=None, batch_size=64):
		# Default: if no dataset is specified, we use validation dataset
		if dataset is not None:
			return super().eval(dataset=dataset, batch_size=batch_size)
		else:
			# Go over all validation datasets and take the average perplexity
			perplexities = list()
			detailed_metrics = dict()
			for val_data in self.val_dataset:
				val_perp, val_metrics = super().eval(dataset=val_data, batch_size=batch_size)
				perplexities.append(val_perp)
				detailed_metrics[val_data.dataset_name] = val_metrics
			avg_perplexity = sum(perplexities) / len(perplexities)
			return avg_perplexity, detailed_metrics


	def add_summary(self, writer, iteration):
		writer.add_scalar("train_%s/teacher_forcing_ratio" % (self.name), self._get_tf_ratio(iteration), iteration)

		gen_list = self.generate_examples()
		for i in range(len(gen_list)):
			data_name = self.train_dataset[self.gen_batch_datasets[i]].dataset_name
			if not self.generated_before:
				writer.add_text(self.name + "_%i_%s_input_dialogue" % (i, data_name), gen_list[i][0], iteration)
				writer.add_text(self.name + "_%i_%s_input_template" % (i, data_name), gen_list[i][1], iteration)
				writer.add_text(self.name + "_%i_%s_input_labels" % (i, data_name), gen_list[i][2], iteration)
			if isinstance(gen_list[i][3], list):
				for sent_index in range(len(gen_list[i][3])):
					writer.add_text(self.name + "_%i_%s_generated_paraphrase_%i" % (i, data_name, sent_index), gen_list[i][3][sent_index], iteration)
			else:
				writer.add_text(self.name + "_%i_%s_generated_paraphrase" % (i, data_name), gen_list[i][3], iteration)
		self.generated_before = True
		
		for dataset_index, train_data in enumerate(self.train_dataset):
			writer.add_scalar("train/%s_loss" % (train_data.dataset_name), self.avg_loss_per_dataset[dataset_index,0]/(1e-10+self.avg_loss_per_dataset[dataset_index,1]), iteration)
			writer.add_scalar("train/%s_num_examples" % (train_data.dataset_name), self.avg_loss_per_dataset[dataset_index,1], iteration)
		self.avg_loss_per_dataset[:,:] = 0



if __name__ == '__main__':

	# Testing perplexity
	torch.manual_seed(42)
	logits = torch.randn(4, 4, 4)
	labels = torch.randint(logits.shape[2], size=(logits.shape[0], logits.shape[1])).long()

	probs = logits.exp() / logits.exp().sum(dim=-1)[:,:,None]
	sel_probs = torch.gather(probs, -1, labels[:,:,None]).squeeze(-1)
	perplexity = sel_probs.log().sum(dim=-1).exp()
	perplexity = torch.pow(perplexity, -1/logits.shape[-1])

	print("Logits: " + str(logits))
	print("Labels: " + str(labels))
	print("Probs: " + str(probs))
	print("Selected probs: " + str(sel_probs))
	print("Perplexity: " + str(perplexity))

	perplexity_2 = TaskTemplate._eval_preplexity(logits, labels)

	print("Perplexity: " + str(perplexity_2))
	assert torch.sum(torch.abs(perplexity - perplexity_2)).item() < 1e-5, "Solutions are deviating!"




