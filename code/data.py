import os
import numpy as np
import torch
import json
import re
import sys
import math
import random
from random import shuffle, randint
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

from vocab import load_word2vec_from_file, get_num_slot_tokens, get_slot_tokens, get_slot_token_start_index, get_id2word_dict, EOS_TOKEN, SOS_TOKEN, UNK_TOKEN
from model_utils import get_device
from statistics import mean, median
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import sent_tokenize, word_tokenize
from pytorch_pretrained_bert.tokenization import BertTokenizer

# 0 => Full debug
# 1 => Reduced output
# 2 => No output at all (on cluster)
DEBUG_LEVEL = 0
DATA_GLOVE = "Glove"
DATA_BERT = "BERT"
SLOT_REGEX = re.compile(r"<[a-zA-Z_]*=[a-zA-Z_0-9.:,\\\"'&\-;? ]*>")

def set_debug_level(level):
	global DEBUG_LEVEL
	DEBUG_LEVEL = level

def debug_level():
	global DEBUG_LEVEL
	return DEBUG_LEVEL


###############################
## Dataset class definitions ##
###############################

class DatasetHandler:

	### Tokenizer for BERT models ###
	# Once set, it will be used for all already loaded and/or future datasets
	BERT_TOKENIZER = None

	### Known datasets to load ###
	# Once a dataset is loaded, it can be reused (speeds up loading significantly)
	AH_PARAPHRASE_DATASETS = None
	LM_WIKITEXT_DATASETS = None
	LM_BOOK_DATASETS = None
	LM_DIALOGUE_DATASETS = None
	MICROSOFT_PARAPHRASE_DATASETS = None
	MICROSOFT_VIDEO_DESC_DATASETS = None
	SNLI_PARAPHRASE_DATASETS = None
	WIKIPEDIA_PARAPHRASE_DATASETS = None
	QUORA_PARAPHRASE_DATASETS = None
	DIALOGUE_PARAPHRASE_DATASETS = None
	DIALOGUE_PARAPHRASE_SMALL_DATASETS = None
	CONTEXT_LM_BOOK_DATASETS = None


	@staticmethod
	def _load_all_type_datasets(dataset_fun, debug_dataset=False, data_types=None, data_path=None, name=None, params=None):
		"""
		Loads the training, validation and test split of a dataset.

		Inputs:
			`dataset_fun`: Function with which the dataset can be loaded. Mostly the constructor of an `DatasetTemplate` object.
						   Requires to take the parameters `shuffle_data`, `data_path` and `name`
			`debug_dataset`: If true, then the training dataset will be replaced by the validation. Speeds up loading for large datasets.
							 Note that this should only be used for debugging purposes and *not* for training.
			`data_types`: Which dataset splits to load. By default, they are 'train', 'val' and 'test'
			`data_path`: Path to the dataset if not set by defaut in `dataset_fun`
			`name`: The name that should be set for the dataset (parameter in `dataset_fun`)

		Outputs:
			`dataset_list`: List containing all dataset splits loaded according to the inputs above
		"""
		_, word2id_dict, _ = load_word2vec_from_file()
		dataset_list = list()
		if data_types is None:
			data_types = ['train' if not debug_dataset else 'val', 'val', 'test']
		elif debug_dataset:
			data_types[0] = data_types[1]
		for data_type in data_types:
			if data_path is None:
				dataset = dataset_fun(data_type, shuffle_data=('train' in data_type), params=params)
			else:
				dataset = dataset_fun(data_type, data_path=data_path, shuffle_data=('train' in data_type), name=name, params=params)
			dataset.set_vocabulary(word2id_dict)
			if DatasetHandler.BERT_TOKENIZER is not None:
				dataset.set_BERT_tokenizer(DatasetHandler.BERT_TOKENIZER)
			dataset.print_statistics()
			dataset_list.append(dataset)
		return dataset_list

	@staticmethod
	def load_AH_Paraphrase_datasets(debug_dataset=False):
		if DatasetHandler.AH_PARAPHRASE_DATASETS is None:
			DatasetHandler.AH_PARAPHRASE_DATASETS = DatasetHandler._load_all_type_datasets(ParaphraseDataset, debug_dataset=debug_dataset)
		return DatasetHandler.AH_PARAPHRASE_DATASETS[0], DatasetHandler.AH_PARAPHRASE_DATASETS[1], DatasetHandler.AH_PARAPHRASE_DATASETS[2]

	@staticmethod
	def load_LM_Wikitext_datasets(debug_dataset=False):
		if DatasetHandler.LM_WIKITEXT_DATASETS is None:
			DatasetHandler.LM_WIKITEXT_DATASETS = DatasetHandler._load_all_type_datasets(LMDataset, data_path="../data/LanguageModeling/wikitext-2", debug_dataset=debug_dataset, name="LM Wikitext")
		return DatasetHandler.LM_WIKITEXT_DATASETS[0], DatasetHandler.LM_WIKITEXT_DATASETS[1], DatasetHandler.LM_WIKITEXT_DATASETS[2]

	@staticmethod
	def load_LM_Book_datasets(debug_dataset=False):
		if DatasetHandler.LM_BOOK_DATASETS is None:
			DatasetHandler.LM_BOOK_DATASETS = DatasetHandler._load_all_type_datasets(LMDataset, data_path="../data/LanguageModeling/books", debug_dataset=debug_dataset, name="LM Books")
		return DatasetHandler.LM_BOOK_DATASETS[0], DatasetHandler.LM_BOOK_DATASETS[1], DatasetHandler.LM_BOOK_DATASETS[2]

	@staticmethod
	def load_LM_Dialogue_datasets(debug_dataset=False):
		if DatasetHandler.LM_DIALOGUE_DATASETS is None:
			DatasetHandler.LM_DIALOGUE_DATASETS = DatasetHandler._load_all_type_datasets(LMDataset, data_path="../data/LanguageModeling/dialogues", debug_dataset=debug_dataset, name="LM Dialogues")
		return DatasetHandler.LM_DIALOGUE_DATASETS[0], DatasetHandler.LM_DIALOGUE_DATASETS[1], DatasetHandler.LM_DIALOGUE_DATASETS[2]

	@staticmethod
	def load_Microsoft_Paraphrase_datasets(debug_dataset=False):
		if DatasetHandler.MICROSOFT_PARAPHRASE_DATASETS is None:
			DatasetHandler.MICROSOFT_PARAPHRASE_DATASETS = DatasetHandler._load_all_type_datasets(PairwiseParaphraseDataset, data_path="../data/Paraphrasing/MicrosoftParaphrase", debug_dataset=debug_dataset, name="Paraphrasing Microsoft Pairwise")
		return DatasetHandler.MICROSOFT_PARAPHRASE_DATASETS[0], DatasetHandler.MICROSOFT_PARAPHRASE_DATASETS[1], DatasetHandler.MICROSOFT_PARAPHRASE_DATASETS[2]

	@staticmethod
	def load_Microsoft_Video_Description_datasets(debug_dataset=False):
		if DatasetHandler.MICROSOFT_VIDEO_DESC_DATASETS is None:
			DatasetHandler.MICROSOFT_VIDEO_DESC_DATASETS = DatasetHandler._load_all_type_datasets(MultiParaphraseDataset, data_path="../data/Paraphrasing/MicrosoftVideoDescription", debug_dataset=debug_dataset, name="Paraphrasing Microsoft Video Description")
		return DatasetHandler.MICROSOFT_VIDEO_DESC_DATASETS[0], DatasetHandler.MICROSOFT_VIDEO_DESC_DATASETS[1], DatasetHandler.MICROSOFT_VIDEO_DESC_DATASETS[2]

	@staticmethod
	def load_SNLI_Paraphrase_datasets(debug_dataset=False):
		if DatasetHandler.SNLI_PARAPHRASE_DATASETS is None:
			DatasetHandler.SNLI_PARAPHRASE_DATASETS = DatasetHandler._load_all_type_datasets(PairwiseParaphraseDataset, data_path="../data/Paraphrasing/SNLI", debug_dataset=debug_dataset, name="Paraphrasing SNLI")
		return DatasetHandler.SNLI_PARAPHRASE_DATASETS[0], DatasetHandler.SNLI_PARAPHRASE_DATASETS[1], DatasetHandler.SNLI_PARAPHRASE_DATASETS[2]

	@staticmethod
	def load_Wikipedia_Paraphrase_datasets(debug_dataset=False):
		if DatasetHandler.WIKIPEDIA_PARAPHRASE_DATASETS is None:
			DatasetHandler.WIKIPEDIA_PARAPHRASE_DATASETS = DatasetHandler._load_all_type_datasets(PairwiseParaphraseDataset, data_path="../data/Paraphrasing/Wikipedia", debug_dataset=debug_dataset, name="Paraphrasing Wikipedia")
		return DatasetHandler.WIKIPEDIA_PARAPHRASE_DATASETS[0], DatasetHandler.WIKIPEDIA_PARAPHRASE_DATASETS[1], DatasetHandler.WIKIPEDIA_PARAPHRASE_DATASETS[2]

	@staticmethod
	def load_Quora_Paraphrase_datasets(debug_dataset=False):
		if DatasetHandler.QUORA_PARAPHRASE_DATASETS is None:
			DatasetHandler.QUORA_PARAPHRASE_DATASETS = DatasetHandler._load_all_type_datasets(PairwiseParaphraseDataset, data_path="../data/Paraphrasing/QuoraQuestions", debug_dataset=debug_dataset, name="Paraphrasing Quora")
		return DatasetHandler.QUORA_PARAPHRASE_DATASETS[0], DatasetHandler.QUORA_PARAPHRASE_DATASETS[1], DatasetHandler.QUORA_PARAPHRASE_DATASETS[2]

	@staticmethod
	def load_Dialogue_Paraphrase_datasets(debug_dataset=False, num_context_turns=2):
		if DatasetHandler.DIALOGUE_PARAPHRASE_DATASETS is None:
			DatasetHandler.DIALOGUE_PARAPHRASE_DATASETS = DatasetHandler._load_all_type_datasets(DialogueParaphraseDataset, data_path="../data/LanguageModeling/dialogues", debug_dataset=debug_dataset, name="Paraphrasing Dialogues MULTIWOZ", params={"num_context_turns": num_context_turns})
		return DatasetHandler.DIALOGUE_PARAPHRASE_DATASETS[0], DatasetHandler.DIALOGUE_PARAPHRASE_DATASETS[1], DatasetHandler.DIALOGUE_PARAPHRASE_DATASETS[2]
	
	@staticmethod
	def load_Dialogue_Paraphrase_Small_datasets(debug_dataset=False, num_context_turns=2):
		if DatasetHandler.DIALOGUE_PARAPHRASE_SMALL_DATASETS is None:
			DatasetHandler.DIALOGUE_PARAPHRASE_SMALL_DATASETS = DatasetHandler._load_all_type_datasets(DialogueParaphraseDataset, data_path="../data/DialogueParaphrasing/simulated_dialogues", debug_dataset=debug_dataset, name="Paraphrasing Dialogues Small", params={"num_context_turns": num_context_turns})
		return DatasetHandler.DIALOGUE_PARAPHRASE_SMALL_DATASETS[0], DatasetHandler.DIALOGUE_PARAPHRASE_SMALL_DATASETS[1], DatasetHandler.DIALOGUE_PARAPHRASE_SMALL_DATASETS[2]

	@staticmethod
	def load_ContextLM_Book_datasets(debug_dataset=False, num_context_sents=3):
		if DatasetHandler.CONTEXT_LM_BOOK_DATASETS is None:
			DatasetHandler.CONTEXT_LM_BOOK_DATASETS = DatasetHandler._load_all_type_datasets(LMContextDataset, data_path="../data/LanguageModeling/books", debug_dataset=debug_dataset, name="ContextLMBooks", params={"num_context_sents": num_context_sents})
		return DatasetHandler.CONTEXT_LM_BOOK_DATASETS[0], DatasetHandler.CONTEXT_LM_BOOK_DATASETS[1], DatasetHandler.CONTEXT_LM_BOOK_DATASETS[2]

	@staticmethod
	def set_BERT_tokenizer(tokenizer, override=False):
		# If already one Tokenizer is set, then we might not want to override it because it takes considerable amount of time to parse it.
		if DatasetHandler.BERT_TOKENIZER is not None and not override: 
			return
		# Set new BERT tokenizer
		DatasetHandler.BERT_TOKENIZER = tokenizer
		# Collect all datasets that have been loaded so far
		all_datasets = [dataset_var for key, dataset_var in DatasetHandler.__dict__.items() 
						if not key.startswith("__") and not callable(getattr(DatasetHandler, key)) and key.endswith("DATASETS")]
		for datasets in all_datasets:
			if datasets is not None: # If loaded: apply the tokenizer for every data split
				for data in datasets:
					data.set_BERT_tokenizer(DatasetHandler.BERT_TOKENIZER)


class DatasetTemplate:

	def __init__(self, data_type="train", shuffle_data=True, name=""):
		self.data_type = data_type
		self.shuffle_data = shuffle_data
		self.set_data_list(list())
		self.label_dict = dict()
		self.num_invalids = 0
		self.dataset_name = name
		self.label_count = None

	def set_data_list(self, new_data):
		self.data_list = new_data
		self.reset_index()

	def _get_next_example(self):
		exmp = self.data_list[self.perm_indices[self.example_index]]
		self.example_index += 1
		if self.example_index >= len(self.perm_indices):
			self.reset_index()
		return exmp

	def reset_index(self):
		self.example_index = 0
		self.perm_indices = list(range(len(self.data_list)))
		if self.shuffle_data:
			shuffle(self.perm_indices)

	@staticmethod
	def sents_to_Tensors(batch_stacked_sents, batch_labels=None, toTorch=False):
		lengths = []
		embeds = []
		for batch_sents in batch_stacked_sents:
			if all([x is None for x in batch_sents]):
				lengths_sents = None
				sent_embeds = None
			elif any([x is None for x in batch_sents]):
				print("[!] ERROR: Found in a list of sentences a None value while embedding. List: %s" % (str(batch_sents)))
				sys.exit(1)
			else: # No element in the sentence is None => embed to array/tensor
				lengths_sents = np.array([x.shape[0] for x in batch_sents])
				max_len = max(np.max(lengths_sents), 1) if lengths_sents.shape[0] > 0 else 1
				sent_embeds = np.zeros((len(batch_sents), max_len), dtype=np.int32) - 1
				for s_index, sent in enumerate(batch_sents):
					sent_embeds[s_index, :sent.shape[0]] = sent
				if toTorch:
					sent_embeds = torch.LongTensor(sent_embeds).to(get_device())
					lengths_sents = torch.LongTensor(lengths_sents).to(get_device())
			lengths.append(lengths_sents)
			embeds.append(sent_embeds)
		if batch_labels is not None:
			if isinstance(batch_labels[0], (list, np.ndarray)):
				lengths_labels = np.array([x.shape[0] for x in batch_labels])
				max_len = max(np.max(lengths_labels), 1) if lengths_labels.shape[0] > 0 else 1
				padded_labels = np.zeros((len(batch_labels), max_len), dtype=np.int32) - 1
				for label_index, lab in enumerate(batch_labels):
					padded_labels[label_index, :lab.shape[0]] = np.array(lab)
				batch_labels = padded_labels
			if toTorch:
				batch_labels = torch.LongTensor(batch_labels).to(get_device())
				lengths_labels = torch.LongTensor(lengths_labels).to(get_device())
		else:
			lengths_labels = None
		return embeds, lengths, batch_labels, lengths_labels

	def get_num_examples(self):
		return len(self.data_list)

	def get_word_list(self):
		all_words = dict()
		for i, data in enumerate(self.data_list):
			if debug_level() == 0:
				print("Processed %4.2f%% of the dataset %s" % (100.0 * i / len(self.data_list), self.dataset_name), end="\r")
			if isinstance(data, ParData):
				data_words = data.input_words + data.paraphrase_words + (data.context_words if data.context_words is not None else [])
			elif isinstance(data, MultiParData):
				data_words = [w for p in data.paraphrases_words for w in p]
			elif isinstance(data, DialogueContextParData):
				data_words = [w for sent in data.paraphrases_words for w in sent] + [w for cont in data.contexts_words for sent in cont for w in sent]
			elif isinstance(data, ContextSentData):
				data_words = data.sentence_words + [w for c in data.contexts_words for w in c]
			else:
				print("[!] ERROR: unknown data object " + str(data.__class__.__name__))
				sys.exit(1)
			for w in data_words:
				if w not in all_words:
					all_words[w] = 1
				else:
					all_words[w] += 1
		word_list = list(all_words.keys())
		print("Found " + str(len(word_list)) + " unique words in dataset %s (%s)" % (self.dataset_name, self.data_type) + " "*20)
		return word_list, all_words

	def set_vocabulary(self, word2vec):
		print("Setting new vocabulary...")
		missing_words = 0
		overall_words = 0
		for data_index, data in enumerate(self.data_list):
			if debug_level() == 0:
				print("Set vocabulary for %4.2f%% of the dataset %s..." % (100.0 * data_index / len(self.data_list), self.dataset_name), end="\r")
			data.translate_to_dict(word2vec)
			mw, ow = data.number_words_not_in_dict(word2vec)
			missing_words += mw 
			overall_words += ow 
		print("Amount of missing words: %4.2f%% (overall %i words)" % (100.0 * missing_words / max(overall_words, 1e-5), overall_words))

	def set_BERT_tokenizer(self, tokenizer):
		print("="*50)
		print("Translating %s - %s with Bert tokenizer" % (self.dataset_name, self.data_type))
		print("="*50)
		for data_index, data in enumerate(self.data_list):
			if debug_level() == 0:
				print("Applied BERT tokenizer for %4.2f%% of the dataset..." % (100.0 * data_index / len(self.data_list)), end="\r")
			data.translate_to_BERT(tokenizer)

	def get_batch(self, batch_size, loop_dataset=True, toTorch=False, label_lengths=False, noun_mask=False, mask_prob=0.0):
		if not loop_dataset:
			batch_size = min(batch_size, len(self.perm_indices) - self.example_index)
		batch_data = [self._get_next_example() for _ in range(batch_size)]
		return self._data_to_batch(batch_data, toTorch=toTorch, label_lengths=label_lengths, noun_mask=noun_mask, mask_prob=mask_prob)

	def _data_to_batch(self, batch_data, toTorch=False, label_lengths=False, noun_mask=False, mask_prob=0.0):
		raise NotImplementedError

	def get_random_batch(self, batch_size, toTorch=False, label_lengths=False, noun_mask=False, mask_prob=0.0):
		batch_data = [self.data_list[randint(0,len(self.data_list)-1)] for _ in range(batch_size)]
		return self._data_to_batch(batch_data, toTorch=toTorch, label_lengths=label_lengths, noun_mask=noun_mask, mask_prob=mask_prob)

	def print_statistics(self):
		print("="*50)
		print("Dataset statistics " + ((self.dataset_name + " ") if self.dataset_name is not None else "") + self.data_type)
		print("-"*50)
		print("Number of examples: " + str(len(self.data_list)))
		if len(self.data_list) != len(self.perm_indices):
			print("Number of overall instances: " + str(len(self.perm_indices)))
		print("="*50)


class ParaphraseDataset(DatasetTemplate):

	def __init__(self, data_type, data_path="../data/AH_Dialogue_Paraphrasing", add_suffix=True, shuffle_data=True, name="AH_Paraphrase", params=None):
		super(ParaphraseDataset, self).__init__(data_type, shuffle_data, name=name)
		self.params = params
		if data_path is not None:
			self.load_data(data_path, data_type)
		else:
			self.data_list = list()
		super().set_data_list(self.data_list)

	def load_data(self, data_path, data_type):
		self.data_list = list()
		dialogues = [line.rstrip() for line in open(data_path + "/dialogue." + data_type, 'r')]
		responses = [line.rstrip() for line in open(data_path + "/response." + data_type, 'r')]
		paraphrases = [line.rstrip() for line in open(data_path + "/paraphrase." + data_type, 'r')]

		assert len(dialogues) == len(responses) and len(responses) == len(paraphrases), \
			   "For the dataset %s at %s, different number of dialogues (%i), responses (%i) and paraphrases (%i) were provided." % \
			   (self.dataset_name, data_path, len(dialogues), len(responses), len(paraphrases))

		counter = 0
		for diag, resp, para in zip(dialogues, responses, paraphrases):
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * counter / len(dialogues), self.dataset_name), end="\r")
			counter += 1
			d = ParData(input_sentence=resp, paraphrase=para, context=diag)
			self.data_list.append(d)

	def _data_to_batch(self, batch_data, toTorch=False, label_lengths=False, noun_mask=False, mask_prob=0.0):
		# Check what embedding indices to use for dialogues
		batch_dialogues_Glove = [data.context_vocab for data in batch_data]
		batch_dialogues_BERT = [data.context_BERT_id for data in batch_data]
		# Check what embedding indices to use for templates
		batch_responses_Glove = [data.input_vocab for data in batch_data]
		batch_responses_BERT = [data.input_BERT_id for data in batch_data]
		# Paraphrases as labels. Note that for prediction, we stick to a word list
		batch_paraphrases = [data.paraphrase_vocab for data in batch_data]
		# Embed indices into tensor/array
		embedded_sents, lengths, _, _ = DatasetTemplate.sents_to_Tensors([batch_dialogues_Glove, batch_dialogues_BERT, batch_responses_Glove, batch_responses_BERT, batch_paraphrases], batch_labels=None, toTorch=toTorch)
		if noun_mask:
			sampled_word_masks = [data.sample_word_masks(p=mask_prob) for data in batch_data]
			masks_to_embed = [[mask[i] for mask in sampled_word_masks] for i in range(3)]
			masks_to_embed.append([mask[3][0] for mask in sampled_word_masks])
			masks_to_embed.append([mask[3][1] for mask in sampled_word_masks])
			embedded_masks, _, _, _ = DatasetTemplate.sents_to_Tensors(masks_to_embed, batch_labels=None, toTorch=toTorch)
			template_masks = (embedded_masks[0], embedded_masks[3])
			context_masks = (embedded_masks[1],)
			paraphrase_masks = (embedded_masks[2], embedded_masks[4])
		
		dialogue_sents = {DATA_GLOVE: embedded_sents[0], DATA_BERT: embedded_sents[1]}
		dialogue_lengths = {DATA_GLOVE: lengths[0], DATA_BERT: lengths[1]}
		template_sents = {DATA_GLOVE: embedded_sents[2], DATA_BERT: embedded_sents[3]}
		template_lengths = {DATA_GLOVE: lengths[2], DATA_BERT: lengths[3]}
		paraphrase_sents = embedded_sents[4]
		paraphrase_lengths = lengths[4]

		to_return = [dialogue_sents, dialogue_lengths, template_sents, template_lengths, paraphrase_sents]
		if label_lengths:
			to_return.append(paraphrase_lengths)
		if noun_mask:
			to_return += [template_masks, context_masks, paraphrase_masks]
		return to_return


class LMDataset(ParaphraseDataset):

	def __init__(self, data_type, data_path="../data/LanguageModeling/wikitext-2", add_suffix=True, shuffle_data=True, name="LanguageModeling"):
		super(LMDataset, self).__init__(data_type, data_path=data_path, add_suffix=add_suffix, shuffle_data=shuffle_data, name=name)

	# Overriding previous loading method from paraphrase
	def load_data(self, data_path, data_type):
		if data_path.endswith("/wikitext-2"):
			self.load_wikitext(data_path, data_type)
		elif data_path.endswith("/books"):
			self.load_books(data_path, data_type)
		elif data_path.endswith("/dialogues"):
			self.load_dialogues(data_path, data_type)
		else:
			print("[!] WARNING: In Language Modeling dataset, an unknown data source was used. Default loading function (wikitext-2) is applied.")
			self.load_wikitext(data_path, data_type)

		sent_lens = [len(d.paraphrase_words) for d in self.data_list]

		if len(sent_lens) > 0:
			print("Average sentence length: %i" % (mean(sent_lens)) + " "*20)
			print("Median sentence length: %i" % (median(sent_lens)))
			print("Maximum sentence length: %i" % (max(sent_lens)))
			print("Example data point: ")
			self.data_list[0].print()

	def load_wikitext(self, data_path, data_type):
		self.data_list = list()
		lines = [line.rstrip() for line in open(data_path + "/" + data_type + ".txt")]
		lines = [line for line in lines if len(line)>0]
		for token_seq in ["<unk>"]:
			lines = [line.replace(token_seq, "UNK") for line in lines]
		for c in ["-", ",", "."]:
			lines = [line.replace(" @%s@ " % (c), c) for line in lines]
		lines = [s for line in lines for s in sent_tokenize(line)]
		lines = [l for l in lines if any([stop_token == l[-1] for stop_token in [".", "?", "!"]])]

		for i in range(5):
			print("Sentence %i: %s" % (i, lines[i]))
		
		for line_index in range(1, len(lines)):
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * line_index / len(lines), self.dataset_name), end="\r")
			if "@" in lines[line_index]:
				print(lines[line_index])
				sys.exit(1)
			d = ParData(input_sentence="", paraphrase=lines[line_index], context=lines[line_index-1], max_len=80)
			self.data_list.append(d)

	def load_books(self, data_path, data_type):
		SPLIT_TOKEN = "$-#-$"*5

		self.data_list = list()
		lines = [line.rstrip() for line in open(data_path + "/" + data_type + ".txt")]
		lines = [line for line in lines if len(line)>0]

		skip_line = False
		for line_index in range(1, len(lines)):
			if skip_line:
				skip_line = False
				continue
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * line_index / len(lines), self.dataset_name), end="\r")
			if lines[line_index] == SPLIT_TOKEN:
				skip_line = True
				continue
			d = ParData(input_sentence="", paraphrase=lines[line_index], context=lines[line_index-1], max_len=80)
			self.data_list.append(d)

	def load_dialogues(self, data_path, data_type):
		self.data_list = list()
		lines = [line.rstrip() for line in open(data_path + "/" + data_type + ".txt")]
		lines = [line for line in lines if len(line)>0]
		lines = [line.split("\t") for line in lines if len(line.split("\t"))==2]

		start_time = time.time()
		if len(lines) < 10000:
			for line_index, line in enumerate(lines):
				if debug_level() == 0:
					print("Read %4.2f%% of the dataset %s" % (100.0 * line_index / len(lines), self.dataset_name), end="\r")
				d = ParData(input_sentence="", paraphrase=line[1], context=line[0], max_len=80)
				self.data_list.append(d)
		else:
			print("Reading %s dataset in parallel..." % (self.dataset_name))
			pool = Pool()
			self.data_list = pool.map(LMDataset._line_to_data, lines)
			pool.close()
		end_time = time.time()
		print("Finished reading dataset %s in %.2f seconds" % (self.dataset_name, end_time - start_time))

	@staticmethod
	def _line_to_data(line):
		return ParData(input_sentence="", paraphrase=line[1], context=line[0], max_len=80)

	def set_vocabulary(self, word2vec, unk_threshold=0.25):
		missing_words = 0
		overall_words = 0
		indices_to_remove = []
		for data_index, data in enumerate(self.data_list):
			data.translate_to_dict(word2vec)
			mw, ow = data.number_words_not_in_dict(word2vec)
			if mw*1.0/ow > unk_threshold: 
				indices_to_remove.append(data_index)
			else:
				missing_words += mw 
				overall_words += ow
		self.data_list = [d for i, d in enumerate(self.data_list) if i not in indices_to_remove]
		self.reset_index()
		print("Amount of missing words: %4.2f%% (overall %i words, removed %i sentences)" % (100.0 * missing_words / max(overall_words, 1e-5), overall_words, len(indices_to_remove)))


class PairwiseParaphraseDataset(ParaphraseDataset):

	def __init__(self, data_type, data_path="../data/Paraphrasing/MicrosoftParaphrase", add_suffix=True, shuffle_data=True, name="MicrosoftParaphrase", params=None):
		super(PairwiseParaphraseDataset, self).__init__(data_type, data_path=data_path, add_suffix=add_suffix, shuffle_data=shuffle_data, name=name, params=params)

	# Overriding previous loading method from paraphrase
	def load_data(self, data_path, data_type, flip_labels=False):
		self.data_list = list()
		lines = [line.rstrip() for line in open(data_path + "/" + data_type + ".txt")]
		lines = [line for line in lines if len(line)>0]
		lines = [line.split("\t") for line in lines]

		for i in range(min(5, len(lines))):
			print("Sentence %i: %s" % (i, str(lines[i])))
		
		start_time = time.time()
		if len(lines) < 5000:
			for line_index, line in enumerate(lines):
				if debug_level() == 0:
					print("Read %4.2f%% of the dataset %s" % (100.0 * line_index / len(lines), self.dataset_name), end="\r")
				d1 = ParData(input_sentence=line[0], paraphrase=line[1], context="", max_len=80)
				self.data_list.append(d1)
				if flip_labels:
					d2 = ParData(input_sentence=line[1], paraphrase=line[0], context="", max_len=80)			
					self.data_list.append(d2)
		else:
			print("Reading %s dataset in parallel..." % (self.dataset_name))
			pool = Pool()
			self.data_list = pool.map(PairwiseParaphraseDataset._line_to_data, [(l, flip_labels) for l in lines])
			self.data_list = [d for sublist in self.data_list for d in sublist]
			pool.close()
		end_time = time.time()
		print("Finished reading %s in %.2f seconds" % (self.dataset_name, end_time - start_time))

	@staticmethod
	def _line_to_data(_input):
		line, flip_labels = _input
		d_list = [ParData(input_sentence=line[0], paraphrase=line[1], context="", max_len=80, create_masks=False)]
		if flip_labels:
			d_list.append(ParData(input_sentence=line[1], paraphrase=line[0], context="", max_len=80, create_masks=False))
		return d_list

	def set_vocabulary(self, word2vec, unk_threshold=0.25):
		missing_words = 0
		overall_words = 0
		indices_to_remove = []
		for data_index, data in enumerate(self.data_list):
			if debug_level() == 0:
				print("Set vocabulary for %4.2f%% of the dataset %s..." % (100.0 * data_index / len(self.data_list), self.dataset_name), end="\r")
			data.translate_to_dict(word2vec)
			mw, ow = data.number_words_not_in_dict(word2vec)
			if mw*1.0/ow > unk_threshold: 
				indices_to_remove.append(data_index)
			else:
				missing_words += mw 
				overall_words += ow
		print("Removing %i items..." % (len(indices_to_remove)) + " "*75)
		self.data_list = [d for i, d in enumerate(self.data_list) if i not in indices_to_remove]
		self.reset_index()
		print("Amount of missing words: %4.2f%% (overall %i words, removed %i sentences)" % (100.0 * missing_words / max(overall_words, 1e-5), overall_words, len(indices_to_remove)))


class MultiParaphraseDataset(ParaphraseDataset):

	def __init__(self, data_type, data_path="../data/Paraphrasing/MicrosoftVideoDescription", add_suffix=True, shuffle_data=True, name="MultiParaphrase", params=None):
		super(MultiParaphraseDataset, self).__init__(data_type, data_path=data_path, add_suffix=add_suffix, shuffle_data=shuffle_data, name=name, params=params)

	# Overriding previous loading method from paraphrase
	def load_data(self, data_path, data_type):
		self.data_list = list()
		lines = [line.rstrip() for line in open(data_path + "/" + data_type + ".txt")]
		lines = [line for line in lines if len(line)>0]
		lines = [line.split("\t") for line in lines]
		
		for line_index, line in enumerate(lines):
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * line_index / len(lines), self.dataset_name), end="\r")
			d = MultiParData(paraphrases=line, max_len=80)
			self.data_list.append(d)

	def reset_index(self):
		self.example_index = 0
		if not self.shuffle_data:
			self.perm_indices = [(i,j) for i in range(len(self.data_list)) for j in range(self.data_list[i].num_pars())]
		else:
			self.perm_indices = []
			if len(self.data_list) == 0:
				return
			min_len = min([len(d.combinations) for d in self.data_list])
			per_datapoint_shuffles = [[i for i in range(len(d.combinations))] for d in self.data_list]
			[shuffle(l) for l in per_datapoint_shuffles]
			for iter_index in range(min_len):
				iter_permutation = list(range(len(self.data_list)))
				shuffle(iter_permutation)
				iter_permutation = [(i,per_datapoint_shuffles[i][iter_index]) for i in iter_permutation]
				self.perm_indices += iter_permutation

	def _get_next_example(self):
		exmp = self.data_list[self.perm_indices[self.example_index][0]]
		exmp = exmp.get_view(self.perm_indices[self.example_index][1])
		self.example_index += 1
		if self.example_index >= len(self.perm_indices):
			self.reset_index()
		return exmp

	def get_random_batch(self, batch_size, toTorch=False, label_lengths=False, noun_mask=False, mask_prob=0.0):
		batch_data = [self.data_list[randint(0,len(self.data_list)-1)] for _ in range(batch_size)]
		batch_data = [b.get_view(randint(0,len(b.combinations)-1)) for b in batch_data]
		return self._data_to_batch(batch_data, toTorch=toTorch, label_lengths=label_lengths, noun_mask=noun_mask, mask_prob=mask_prob)

	def get_num_examples(self):
		return len(self.perm_indices)

class DialogueParaphraseDataset(MultiParaphraseDataset):

	def __init__(self, data_type, data_path="../data/LanguageModeling/dialogues", add_suffix=True, shuffle_data=True, name="Dialogue Paraphrase MULTIWOZ", params=None):
		super(DialogueParaphraseDataset, self).__init__(data_type, data_path=data_path, add_suffix=add_suffix, shuffle_data=shuffle_data, name=name, params=params)

	# Overriding previous loading method from paraphrase
	def load_data(self, data_path, data_type, flip_labels=False):
		if self.params is not None:
			self.num_context_turns = self.params["num_context_turns"]
		else:
			self.num_context_turns = 2

		self.data_list = list()
		if data_path is None:
			return

		with open(os.path.join(data_path, data_type + ".json"), "r") as f:
			data_dict = json.load(f)
		with open(os.path.join(data_path, "conversations.json"), "r") as f:
			conversation_dict = json.load(f)

		counter = 0
		for key, sents in data_dict.items():
			if debug_level() == 0:
				print("Read %4.2f%% of the keys of dataset %s" % (100.0 * counter / len(data_dict.keys()), self.dataset_name), end="\r")
			paraphrases = []
			contexts = []
			for sent_key, par_sent in sents.items():
				paraphrases.append(par_sent)
				conv_key, conv_turn_ID = sent_key.split("_")[0], int(sent_key.split("_")[1][1:])
				c = conversation_dict[conv_key]
				
				prev_conv_turns = []
				for t in range(-self.num_context_turns,0):
					prev_conv_turns.append("U%i" % (conv_turn_ID+t))
					if t < -1:
						prev_conv_turns.append("B%i" % (conv_turn_ID+t))
				contexts.append([c[cturn] if cturn in c else "" for cturn in prev_conv_turns])
				
				current_response = c["B%i" % (conv_turn_ID-1)]

				slots = SLOT_REGEX.findall(par_sent)
				sent_wo_slots = par_sent.strip()
				for s in slots:
					sent_wo_slots = sent_wo_slots.replace(s,s.split("=")[-1][1:-2])
				sent_wo_slots = sent_wo_slots.replace("  "," ")
				if not current_response.startswith(sent_wo_slots): # .replace(" ","").replace(".","")
					if sent_wo_slots not in current_response:
						# if all([])
						# print("-"*75)
						# print("Key: %s\nCurrent response: %s\nSentence wo slots: %s\nOriginal sentences: %s" % (key, current_response, sent_wo_slots, par_sent))
						# print("Sentence key: %s" % (sent_key))
						# sys.exit(1)
						pass
					else:
						contexts[-1].append(current_response.split(sent_wo_slots)[0])
						del contexts[-1][0]
						# print("New context: %s" % str(contexts[-1]))


			d = DialogueContextParData(paraphrases=paraphrases, contexts=contexts, max_len=80, randomized=self.shuffle_data)
			self.data_list.append(d)
			counter += 1

	@staticmethod
	def _prepare_slots_for_batch(batch_par_slots):
		batch_par_slots_max_len = max(max([len(s) for s in batch_par_slots]), 1)
		batch_par_slots = [s + [np.array([], dtype=np.int32) for _ in range(batch_par_slots_max_len - len(s))] for s in batch_par_slots]
		batch_par_slots = [v for s in batch_par_slots for v in s]
		return batch_par_slots, batch_par_slots_max_len
	
	@staticmethod
	def _reshape_batch_data(batch_data, batch_lengths, batch_size, sub_size):
		if isinstance(batch_data, np.ndarray):
			batch_data = np.reshape(batch_data, newshape=(batch_size, sub_size, batch_data.shape[-1]))
			batch_lengths = np.reshape(batch_lengths, newshape=(batch_size, sub_size))
		else: # Torch tensor
			batch_data = batch_data.contiguous().view(batch_size, sub_size, batch_data.shape[-1])
			batch_lengths = batch_lengths.contiguous().view(batch_size, sub_size)
		return batch_data, batch_lengths

	def _data_to_batch(self, batch_data, toTorch=False, label_lengths=False, noun_mask=False, mask_prob=0.0):
		batch_size = len(batch_data)

		batch_par_1 = [data.par_1_vocab for data in batch_data]
		batch_par_2 = [data.par_2_vocab for data in batch_data]
		
		batch_par_1_slots, batch_par_1_slots_max_len = DialogueParaphraseDataset._prepare_slots_for_batch([data.slot_1_vocab for data in batch_data])
		batch_par_2_slots, batch_par_2_slots_max_len = DialogueParaphraseDataset._prepare_slots_for_batch([data.slot_2_vocab for data in batch_data])
		assert len(batch_par_1_slots) == batch_size * batch_par_1_slots_max_len, "Something went wrong when integrating the slot values for par 1"
		assert len(batch_par_2_slots) == batch_size * batch_par_2_slots_max_len, "Something went wrong when integrating the slot values for par 2"

		batch_contexts = [con_voc for data in batch_data for con_voc in data.context_1_vocab] + [con_voc for data in batch_data for con_voc in data.context_2_vocab]

		context_size = len(batch_data[0].context_1_vocab)
		assert all([len(data.context_1_vocab) == context_size for data in batch_data]), "Number of context sentences must be equal for all batch elements"
		assert all([len(data.context_2_vocab) == context_size for data in batch_data]), "Number of context sentences must be equal for all batch elements"

		embedded_sents, lengths, _, _ = DatasetTemplate.sents_to_Tensors([batch_par_1, batch_par_2, batch_par_1_slots, batch_par_2_slots, batch_contexts], batch_labels=None, toTorch=toTorch)
		
		batch_par_1 = embedded_sents[0]
		batch_par_lengths_1 = lengths[0]
		batch_par_2 = embedded_sents[1]
		batch_par_lengths_2 = lengths[1]
		batch_par_1_slots = embedded_sents[2]
		batch_par_1_slots_lengths = lengths[2]
		batch_par_2_slots = embedded_sents[3]
		batch_par_2_slots_lengths = lengths[3]
		batch_context_1 = embedded_sents[4][:int(len(batch_contexts)/2)]
		batch_context_2 = embedded_sents[4][int(len(batch_contexts)/2):]
		batch_context_lengths_1 = lengths[4][:int(len(batch_contexts)/2)]
		batch_context_lengths_2 = lengths[4][int(len(batch_contexts)/2):]
		
		batch_par_1_slots, batch_par_1_slots_lengths = DialogueParaphraseDataset._reshape_batch_data(batch_par_1_slots, batch_par_1_slots_lengths, batch_size=batch_size, sub_size=batch_par_1_slots_max_len)
		batch_par_2_slots, batch_par_2_slots_lengths = DialogueParaphraseDataset._reshape_batch_data(batch_par_2_slots, batch_par_2_slots_lengths, batch_size=batch_size, sub_size=batch_par_2_slots_max_len)
		batch_context_1, batch_context_lengths_1 = DialogueParaphraseDataset._reshape_batch_data(batch_context_1, batch_context_lengths_1, batch_size=batch_size, sub_size=context_size)
		batch_context_2, batch_context_lengths_2 = DialogueParaphraseDataset._reshape_batch_data(batch_context_2, batch_context_lengths_2, batch_size=batch_size, sub_size=context_size)

		return (batch_par_1, batch_par_lengths_1, batch_par_2, batch_par_lengths_2, batch_par_1_slots, batch_par_1_slots_lengths, batch_par_2_slots, batch_par_2_slots_lengths, batch_context_1, batch_context_lengths_1, batch_context_2, batch_context_lengths_2)

	def get_all_sentences(self):
		unique_data = []
		indices = []
		for d_index, d in enumerate(self.data_list):
			all_sents = d.get_all_sents()
			unique_data += all_sents
			indices += [d_index for _ in range(len(all_sents))]
		return unique_data, indices

	def reset_index(self):
		self.example_index = 0
		if not self.shuffle_data:
			self.perm_indices = [(i,j) for i in range(len(self.data_list)) for j in range(self.data_list[i].num_pars())]
		else:
			self.perm_indices = []
			if len(self.data_list) == 0:
				return
			for d in self.data_list:
				d.prepare_combinations(randomized=True)
			# min_len = min([len(d.combinations) for d in self.data_list])
			# per_datapoint_shuffles = [[i for i in range(len(d.combinations))] for d in self.data_list]
			# [shuffle(l) for l in per_datapoint_shuffles]
			# for iter_index in range(min_len):
			# 	iter_permutation = list(range(len(self.data_list)))
			# 	shuffle(iter_permutation)
			# 	iter_permutation = [(i,per_datapoint_shuffles[i][iter_index]) for i in iter_permutation]
			# 	self.perm_indices += iter_permutation
			# lens = [len(d.paraphrases_words) for d in self.data_list]
			# plt.hist([l if l < 100 else 100 for l in lens], bins=20)
			# # plt.yscale('log', nonposy='clip')
			# plt.show()
			min_len = min([len(d.combinations) for d in self.data_list])
			per_datapoint_shuffles = [[i for i in range(len(d.combinations))] for d in self.data_list]
			[shuffle(l) for l in per_datapoint_shuffles]
			self.perm_indices = [(i, per_datapoint_shuffles[i][j]) for i in range(len(per_datapoint_shuffles)) for j in range(max(min_len, int( (min(200,len(per_datapoint_shuffles[i])))**(0.75) ) ))]	
			shuffle(self.perm_indices)		

	def print_slot_distribution(self):
		print("="*75)
		print("Slot distribution")
		print("-"*75)
		slot_dist, slot_count = None, list()
		for d in self.data_list:
			d_slot_dist, d_slot_counts = d.get_slot_dist()
			if slot_dist is None:
				slot_dist = d_slot_dist
			else:
				slot_dist += d_slot_dist
			slot_count += d_slot_counts
		slot_count = np.array(slot_count)
		slot_tokens = get_slot_tokens()
		for word_ind, slot_name in slot_tokens.items():
			word_ind = get_slot_token_start_index() - word_ind
			print("%s: %i (%4.2f%%)" % (slot_name, slot_dist[word_ind], 100.0 * slot_dist[word_ind] / max(1e-5, np.sum(slot_dist))))
		print("-"*75)
		print("Overall: %i" % (np.sum(slot_dist)))
		print("#"*75)
		print("Number of slots per paraphrase:")
		print("-"*75)
		for i in range(max(slot_count)+1):
			print("#%i: %i" % (i, np.sum(slot_count == i)))
		print("-"*75)
		print("Number of paraphrases: %i" % (slot_count.shape[0]))
		print("Number of slots: %i" % (np.sum(slot_count)))
		print("Avg number of slots per paraphrase: %4.2f" % (np.sum(slot_count) * 1.0 / slot_count.shape[0]))
		num_words_overall = sum([sum([len(p) for p in d.paraphrases_words]) for d in self.data_list])
		print("Number of words in paraphrases: %i (%4.2f in avg)" % (num_words_overall, num_words_overall * 1.0 / slot_count.shape[0]))
		print("Proportion of words being slots: %4.2f%%" % (100.0 * np.sum(slot_count) / num_words_overall))
		print("="*75)

class LMContextDataset(ParaphraseDataset):

	def __init__(self, data_type, data_path="../data/LanguageModeling/wikitext-2", add_suffix=True, shuffle_data=True, name="ContextLanguageModeling", params=None):
		super(LMContextDataset, self).__init__(data_type, data_path=data_path, add_suffix=add_suffix, shuffle_data=shuffle_data, name=name, params=params)

	# Overriding previous loading method from paraphrase
	def load_data(self, data_path, data_type):
		if self.params is not None:
			self.num_context_sents = self.params["num_context_sents"]
		else:
			self.num_context_sents = 3

		if data_path.endswith("/wikitext-2"):
			self.load_wikitext(data_path, data_type)
		elif data_path.endswith("/books"):
			self.load_books(data_path, data_type)
		else:
			print("[!] WARNING: In %s dataset, an unknown data source was used. Default loading function (wikitext-2) is applied." % self.name)
			self.load_wikitext(data_path, data_type)

		sent_lens = [len(d.sentence_words) for d in self.data_list]

		if len(sent_lens) > 0:
			print("Average sentence length: %i" % (mean(sent_lens)) + " "*20)
			print("Median sentence length: %i" % (median(sent_lens)))
			print("Maximum sentence length: %i" % (max(sent_lens)))
			print("Example data point: ")
			self.data_list[0].print()

	def load_wikitext(self, data_path, data_type):
		raise NotImplementedError

		self.data_list = list()
		lines = [line.rstrip() for line in open(data_path + "/" + data_type + ".txt")]
		lines = [line for line in lines if len(line)>0]
		for token_seq in ["<unk>"]:
			lines = [line.replace(token_seq, "UNK") for line in lines]
		for c in ["-", ",", "."]:
			lines = [line.replace(" @%s@ " % (c), c) for line in lines]
		lines = [s for line in lines for s in sent_tokenize(line)]
		lines = [l for l in lines if any([stop_token == l[-1] for stop_token in [".", "?", "!"]])]

		for i in range(5):
			print("Sentence %i: %s" % (i, lines[i]))
		
		for line_index in range(1, len(lines)):
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * line_index / len(lines), self.dataset_name), end="\r")
			if "@" in lines[line_index]:
				print(lines[line_index])
				sys.exit(1)
			d = ParData(input_sentence="", paraphrase=lines[line_index], context=lines[line_index-1], max_len=80)
			self.data_list.append(d)

	def load_books(self, data_path, data_type):
		SPLIT_TOKEN = "$-#-$"*5

		self.data_list = list()
		lines = [line.rstrip() for line in open(data_path + "/" + data_type + ".txt")]
		lines = [line for line in lines if len(line)>0]

		skip_line = False
		for line_index in range(self.num_context_sents, len(lines)):
			if skip_line > 0:
				skip_line -= 1
				continue
			if debug_level() == 0:
				print("Read %4.2f%% of the dataset %s" % (100.0 * line_index / len(lines), self.dataset_name), end="\r")
			if lines[line_index] == SPLIT_TOKEN:
				skip_lines = self.num_context_sents
				continue
			d = ContextSentData(sent=lines[line_index], context=lines[line_index-self.num_context_sents:line_index], max_len=50)
			self.data_list.append(d)

	def set_vocabulary(self, word2vec, unk_threshold=0.25):
		missing_words = 0
		overall_words = 0
		indices_to_remove = []
		for data_index, data in enumerate(self.data_list):
			data.translate_to_dict(word2vec)
			mw, ow = data.number_words_not_in_dict(word2vec, only_sents=True)
			if mw*1.0/ow > unk_threshold: 
				indices_to_remove.append(data_index)
			else:
				missing_words += mw 
				overall_words += ow
		self.data_list = [d for i, d in enumerate(self.data_list) if i not in indices_to_remove]
		self.reset_index()
		print("Amount of missing words: %4.2f%% (overall %i words, removed %i sentences)" % (100.0 * missing_words / max(overall_words, 1e-5), overall_words, len(indices_to_remove)))

	def _data_to_batch(self, batch_data, toTorch=False, label_lengths=False, noun_mask=False, mask_prob=0.0):
		
		batch_sents = [data.sentence_vocab for data in batch_data]
		
		batch_contexts = [con_voc for data in batch_data for con_voc in data.contexts_vocab]
		context_size = len(batch_data[0].contexts_vocab)
		
		embedded_sents, lengths, _, _ = DatasetTemplate.sents_to_Tensors([batch_sents, batch_contexts], batch_labels=None, toTorch=toTorch)
		
		batch_sents = embedded_sents[0]
		batch_sents_lengths = lengths[0]
		batch_contexts = embedded_sents[1]
		batch_contexts_lengths = lengths[1]
		if isinstance(batch_contexts, np.ndarray):
			batch_contexts = np.reshape(batch_contexts, newshape=(len(batch_data), context_size, batch_contexts.shape[-1]))
			batch_contexts_lengths = np.reshape(batch_contexts_lengths, newshape=(len(batch_data), context_size))
		else: # Torch tensor
			batch_contexts = batch_contexts.contiguous().view(len(batch_data), context_size, batch_contexts.shape[-1])
			batch_contexts_lengths = batch_contexts_lengths.contiguous().view(len(batch_data), context_size)

		return (batch_sents, batch_sents_lengths, batch_contexts, batch_contexts_lengths)


class ParData:

	def __init__(self, input_sentence, paraphrase, context=None, max_len=-1, create_masks=True):
		self.input_words = ParData._preprocess_sentence(input_sentence, max_len=max_len)
		self.paraphrase_words = ParData._preprocess_sentence(paraphrase, max_len=max_len)
		self.context_words = ParData._preprocess_sentence(context, max_len=max_len) if context is not None else None
		self.input_vocab = None
		self.paraphrase_vocab = None
		self.context_vocab = None

		self.input_BERT = None
		self.context_BERT = None
		self.input_BERT_id = None
		self.context_BERT_id = None

		self.input_maskable_words = None
		self.paraphrase_maskable_words = None
		self.context_maskable_words = None
		self.aligned_shared_words = None

		if create_masks:
			self._align_nouns()

	def translate_to_dict(self, word_dict):
		self.input_vocab = ParData._sentence_to_dict(word_dict, self.input_words)
		self.paraphrase_vocab = ParData._sentence_to_dict(word_dict, self.paraphrase_words)
		if self.context_words is not None:
			self.context_vocab = ParData._sentence_to_dict(word_dict, self.context_words)

		if self.input_maskable_words is not None:
			self.input_maskable_words = ParData._update_maskable_words(self.input_maskable_words, self.input_vocab, word_dict)
			self.paraphrase_maskable_words = ParData._update_maskable_words(self.paraphrase_maskable_words, self.paraphrase_vocab, word_dict)
			self.aligned_shared_words = ParData._determine_aligned_words(self.input_maskable_words, self.paraphrase_maskable_words)
			if self.context_maskable_words is not None:
				self.context_maskable_words = ParData._update_maskable_words(self.context_maskable_words, self.context_vocab, word_dict)

	def translate_to_BERT(self, tokenizer):
		self.input_BERT, self.input_BERT_id = ParData._sentence_to_BERT(tokenizer, self.input_words)
		if self.context_words is not None:
			self.context_BERT, self.context_BERT_id = ParData._sentence_to_BERT(tokenizer, self.context_words)

	def number_words_not_in_dict(self, word_dict):
		missing_words = 0
		all_words = self.input_words + self.paraphrase_words + (self.context_words if self.context_words is not None else [])
		for w in all_words:
			if w not in word_dict:
				missing_words += 1
		return missing_words, len(all_words)

	def _align_nouns(self):
		input_pos_tags = nltk.pos_tag(self.input_words)
		paraphrase_pos_tags = nltk.pos_tag(self.paraphrase_words)

		self.input_maskable_words = ParData._extract_maskable_words(self.input_words, input_pos_tags)
		self.paraphrase_maskable_words = ParData._extract_maskable_words(self.paraphrase_words, paraphrase_pos_tags)

		if self.context_words is not None:
			context_pos_tags = nltk.pos_tag(self.context_words)
			self.context_maskable_words = ParData._extract_maskable_words(self.context_words, context_pos_tags)

		self.aligned_shared_words = ParData._determine_aligned_words(self.input_maskable_words, self.paraphrase_maskable_words)

	def sample_word_masks(self, p=0.0):
		"""
		Samples a mask which can be used to replace nouns by UNK tokens
		Inputs:
			`p`: Probability with which regular, known nouns are replaced by UNK
		"""
		input_mask = ParData._create_mask(self.input_maskable_words, p=p)
		par_mask = ParData._create_mask(self.paraphrase_maskable_words, p=p)
		context_mask = ParData._create_mask(self.context_maskable_words, p=p) if self.context_maskable_words is not None else None

		input_par_mask = ParData._create_aligned_mask(self.aligned_shared_words, s1_len=len(self.input_words), s2_len=len(self.paraphrase_words), p=p)

		return input_mask, context_mask, par_mask, input_par_mask

	def print(self):
		print("+"*50)
		print("Input sentence: \"%s\"" % (" ".join(self.input_words)))
		print("Paraphrase: \"%s\"" % (" ".join(self.paraphrase_words)))
		if self.context_words is not None:
			print("Context sentence: \"%s\"" % (" ".join(self.context_words)))
		else:
			print("No context provided")
		id2word = get_id2word_dict()
		print("-"*50)
		for vocab_name, vocab_array in zip(["input", "paraphrase", "context"], [self.input_vocab, self.paraphrase_vocab, self.context_vocab]):
			if vocab_array is not None:
				print("Reconstructed %s: \"%s\"" % (vocab_name, " ".join([id2word[w] for w in vocab_array])))
			else:
				print("No vocabulary was set yet for %s." % (vocab_name))
		print("-"*50)
		print("Maskable input nouns: \"%s\"" % (" ".join([str(w) if w is not None else "_" for w in self.input_maskable_words])))
		print("Maskable paraphrase nouns: \"%s\"" % (" ".join([str(w) if w is not None else "_" for w in self.paraphrase_maskable_words])))
		print("Aligned words: \"%s\"" % (str(self.aligned_shared_words)))
		print("-"*50)
		input_mask, par_mask, context_mask, input_par_mask = self.sample_word_masks(p=0.5)
		print("Mask input: %s" % str(input_mask))
		print("Mask paraphrase: %s" % str(par_mask))
		print("Mask context: %s" % str(context_mask))
		print("Aligned mask: (input) %s, (par) %s" % (str(input_par_mask[0]), str(input_par_mask[1])))
		print("+"*50)

	@staticmethod
	def _preprocess_sentence(sent, max_len=-1, full_preprocessing=False):
		"""
		Preprocessing a sentence for Glove word embeddings. Therefore, we apply:
		(1) Lower-casing
		(2) Splitting words by spaces
		(3) Correcting punctuation

		Inputs:
			`sent`: A string which summarizes the sentence.
			`max_len`: Maximum length of a sentence we allow. If a sentence is longer than this defined length, the sentence will be cut.
			`full_preprocessing`: If selected, we apply nltk's `word_tokenize` function on the sentences.

		Outputs:
			`sent_words`: List of words which can be used to look up in the vocabulary
		"""
		sent = sent.lower().strip()

		sent_splits = SLOT_REGEX.split(sent)
		slots = SLOT_REGEX.findall(sent)
		all_sent_words = []

		for s_index in range(len(sent_splits)):
			if s_index > 0:
				all_sent_words.append(slots[s_index-1])
			if full_preprocessing:
				sent_words = word_tokenize(sent_splits[s_index])
			else:
				sent_words = sent_splits[s_index].split(" ")
				if "." in sent_words[-1] and len(sent_words[-1]) > 1:
					sent_words[-1] = sent_words[-1].replace(".","")
					sent_words.append(".")
				sent_words = [w for w in sent_words if len(w) > 0]
			all_sent_words += sent_words

		all_sent_words = [w for w in all_sent_words if len(w) > 0]
		if max_len > 0 and len(all_sent_words) > max_len:
			all_sent_words = all_sent_words[:max_len]
		return all_sent_words

	@staticmethod
	def _preprocess_slots(sent, max_len=-1):
		"""
		Extracting all slots and the corresponding values from a sentence
		"""
		sent = sent.lower().strip()

		slots = SLOT_REGEX.findall(sent)
		slots = [s.replace("<","").replace(">","").replace("\"","").split("=") for s in slots]
		slots = [(s[0], s[1].replace(":", " : ")) for s in slots]
		slots = [(s[0], [w for w in s[1].split(" ") if len(w) > 0]) for s in slots]
		if max_len > 0:
			slots = [(s[0], s[1][:min(max_len, len(s[1]))]) for s in slots]

		return slots

	@staticmethod
	def _sentence_to_dict(word_dict, sent, add_SOS_EOS=True):
		"""
		Translates a single sentence to word ids of the vocabulary. Add start-of-sentence and end-of-sentence symbol

		Inputs:
			`word_dict`: Vocabulary dictionary of words (string) to ids (int)
			`sent`: List of word tokens in the sentence. Can be generated by applying  `_preprocess_sentence` beforehand

		Outputs: 
			`vocab_words`: Numpy array of word ids corresponding to the input sentence
		"""
		vocab_words = list()
		if len(sent) > 0:
			if add_SOS_EOS:
				vocab_words += [word_dict[SOS_TOKEN]]
			vocab_words += ParData._word_seq_to_dict(sent, word_dict)
			if add_SOS_EOS:
				vocab_words += [word_dict[EOS_TOKEN]]
		vocab_words = np.array(vocab_words, dtype=np.int32)
		return vocab_words

	@staticmethod
	def _sentence_to_BERT(tokenizer, sent):
		"""
		Prepares a sentence for input to a BERT model by applying its tokenizer.
		Also adds classifier token [CLS] and separater [SEP]

		Inputs:
			`tokenizer`: A tokenizer of the type `BertTokenizer` that is created for the used model in the Encoder part.
			`sent`: A list of word tokens or string that represents the input sentence to the BERT model.

		Outputs:
			`BERT_sent`: The sentence/word token list that is the output of the tokenizer
			`BERT_id`: Word token IDs for input to BERT
		"""
		if isinstance(sent, list):
			sent = " ".join(sent)
		sent = sent.replace(" unk", " [UNK]")
		BERT_sent = tokenizer.tokenize(sent)
		if len(BERT_sent) > 0:
			BERT_sent = ["[CLS]"] + BERT_sent + ["[SEP]"]
		BERT_id = tokenizer.convert_tokens_to_ids(BERT_sent)
		BERT_id = np.array(BERT_id)
		return BERT_sent, BERT_id

	@staticmethod
	def _word_seq_to_dict(word_seq, word_dict):
		"""
		Finds for a word sequence the most suitable fits in a vocabulary. Considers perfect fits, and also artifacts like "-" and "/" between words.

		Inputs: 
			`word_seq`: List of word tokens in the sentence. Can be generated by applying  `_preprocess_sentence` beforehand
			`word_dict`: Vocabulary dictionary of words (string) to ids (int)

		Output: 
			`vocab_words`: List of word ids
		"""
		global SLOT_REGEX
		vocab_words = list()
		for w in word_seq:
			if len(w) <= 0:
				continue
			if w in word_dict:
				vocab_words.append(word_dict[w])
			elif SLOT_REGEX.match(w):
				slot_ind = "<%s>" % (w.split("=")[0][1:])
				if slot_ind not in word_dict:
					print("[#] WARNING: Slot index not found in word dict! Slot %s" % w) 
				vocab_words.append(word_dict[slot_ind])
			elif "-" in w:
				sw = [c for c in w.split("-") if len(c) > 0]
				vocab_words += ParData._word_seq_to_dict([sw[0] if len(sw) > 0 else UNK_TOKEN], word_dict) # Only the first element is added to make sure that the sentence length is not changed 
			elif "/" in w:
				sw = [c for c in w.split("/") if len(c) > 0]
				vocab_words += ParData._word_seq_to_dict([sw[0] if len(sw) > 0 else UNK_TOKEN], word_dict)
			else:
				subword = re.sub('\W+','', w)
				if subword in word_dict:
					vocab_words.append(word_dict[subword])
				else:
					vocab_words.append(word_dict[UNK_TOKEN])
		return vocab_words

	@staticmethod
	def _extract_maskable_words(sent, pos_tags=None):
		if pos_tags is None:
			pos_tags = nltk.pos_tag(sent)
		return [(w, 'noun' if pos_tags[w_ind][1]!='CD' else 'number') if pos_tags[w_ind][1] in ['NN','NNS','NNP','NNPS','NP','NP-tl','NN-tl','CD'] else None for w_ind, w in enumerate(sent)]

	@staticmethod
	def _determine_aligned_words(masked_sent_1, masked_sent_2):
		aligned_words = list()
		for w_ind_1, w in enumerate(masked_sent_1):
			if w is None:
				continue
			if w in masked_sent_2:
				num_prev_occurrences = sum([w[0] == al_w[3] for al_w in aligned_words])
				if num_prev_occurrences > 0: # Check if this word was already aligned
					w_ind_2 = -1
					for w_ind_2_local, w_2 in enumerate(masked_sent_2):
						if w_2 == w:
							if num_prev_occurrences == 0:
								w_ind_2 = w_ind_2_local
								break
							else:
								num_prev_occurrences -= 1
					if w_ind_2 == -1:
						continue # Skip this word because there are no alignments anymore
				else:
					w_ind_2 = masked_sent_2.index(w)
				aligned_words.append((w_ind_1, w_ind_2, w[1], w[0]))
		return aligned_words

	@staticmethod
	def _update_maskable_words(masked_sent, sent_vocab, word_dict):
		# Set all unknown tokens to required masks
		# w_ind+1 for sent_vocab because we have there the start token as well
		if sent_vocab.shape[0] != 0 and len(masked_sent)+2 != sent_vocab.shape[0]:
			print("Sizes of sentence (%i) and vocab (%i) do not align!" % (len(masked_sent), sent_vocab.shape[0]))
			print("Masked sentence: %s" % str(masked_sent))
			print("Sentence vocab: %s" % str(sent_vocab))
			print("Unknown word index: %s" % str(word_dict[UNK_TOKEN]))
		masked_sent = [w if w is None else (w[0], 'unknown' if sent_vocab[w_ind+1] == word_dict[UNK_TOKEN] else w[1]) for w_ind, w in enumerate(masked_sent)] 
		return masked_sent

	@staticmethod
	def _create_mask(maskable_words, p=0.0):
		mask = [0] # First (SOS) and last token (EOS) cannot be masked
		counter = 0
		for w in maskable_words:
			if w is not None and (w[1] != 'noun' or random.random() < p):
				counter += 1 if w is not None else 0
				mask.append(counter if w is not None else 0)
			else:
				mask.append(0)
		mask.append(0) # First (SOS) and last token (EOS) cannot be masked
		return np.array(mask, dtype=np.int32)

	@staticmethod
	def _create_aligned_mask(aligned_maskable_words, s1_len, s2_len, p=0.0):
		s1_mask = np.zeros(shape=(s1_len+2,), dtype=np.int32)
		s2_mask = np.zeros(shape=(s2_len+2,), dtype=np.int32)

		counter = 0
		for mask_word in aligned_maskable_words:
			if mask_word[2] != 'noun' or random.random() < p:
				counter += 1
				s1_mask[mask_word[0]+1] = counter
				s2_mask[mask_word[1]+1] = counter

		return s1_mask, s2_mask


class MultiParData:

	def __init__(self, paraphrases, max_len=-1):
		self.paraphrases_words = [ParData._preprocess_sentence(p, max_len=max_len) for p in paraphrases]
		self.paraphrases_vocab = None
		self.paraphrases_BERT = None
		self.paraphrases_BERT_id = None
		self.combinations = [(i, j) for i in range(len(self.paraphrases_words)) for j in range(len(self.paraphrases_words))]
		self.combinations = [(i, j) for (i,j) in self.combinations if i != j]

		self.paraphrases_masks = [ParData._extract_maskable_words(p) for p in self.paraphrases_words]

	def _create_aligned_maskings(self):
		self.aligned_maskings = list()
		for c in self.combinations:
			self.aligned_maskings.append(ParData._determine_aligned_words(self.paraphrases_masks[c[0]], self.paraphrases_masks[c[1]]))

	def translate_to_dict(self, word_dict):
		self.word_dict = word_dict
		self.paraphrases_vocab = [ParData._sentence_to_dict(word_dict, p) for p in self.paraphrases_words]
		self.paraphrases_masks = [ParData._update_maskable_words(pm, pv, word_dict) for pm, pv in zip(self.paraphrases_masks, self.paraphrases_vocab)]
		self._create_aligned_maskings()

	def translate_to_BERT(self, tokenizer):
		self.paraphrases_BERT, self.paraphrases_BERT_id = [], []
		for sent in self.paraphrases_words:
			BERT_sent, BERT_id = ParData._sentence_to_BERT(tokenizer, sent)
			self.paraphrases_BERT.append(BERT_sent)
			self.paraphrases_BERT_id.append(BERT_id)

	def number_words_not_in_dict(self, word_dict):
		missing_words = 0
		all_words = [w for sent in self.paraphrases_words for w in sent]
		for w in all_words:
			if w not in word_dict:
				missing_words += 1
		return missing_words, len(all_words)

	def print(self):
		print("+"*50)
		print("Paraphrases: ")
		for i in range(len(self.paraphrases_words)):
			print("(%i) \"%s\"" % (i, " ".join(self.paraphrases_words[i])))
		print("+"*50)

	def num_pars(self):
		return len(self.paraphrases_words)

	def get_view(self, comb_index):
		new_data_view = ParData("", "", context="") 
		new_data_view.input_words = self.paraphrases_words[self.combinations[comb_index][0]] 
		new_data_view.paraphrase_words = self.paraphrases_words[self.combinations[comb_index][1]] 
		new_data_view.context_words = []
		new_data_view.input_vocab = self.paraphrases_vocab[self.combinations[comb_index][0]] 
		new_data_view.paraphrase_vocab = self.paraphrases_vocab[self.combinations[comb_index][1]] 
		new_data_view.context_vocab = np.array([], dtype=np.int32)
		if self.paraphrases_BERT_id is not None:
			new_data_view.input_BERT_id = self.paraphrases_BERT_id[self.combinations[comb_index][0]]
		new_data_view.context_BERT_id = np.array([], dtype=np.int32)
		new_data_view.input_maskable_words = self.paraphrases_masks[self.combinations[comb_index][0]]
		new_data_view.context_maskable_words = []
		new_data_view.paraphrase_maskable_words = self.paraphrases_masks[self.combinations[comb_index][1]]
		new_data_view.aligned_shared_words = self.aligned_maskings[comb_index]
		return new_data_view


class ContextParData:

	def __init__(self, sent_1, context_1, sent_2, context_2, max_len=-1):
		self.par_1_words = ParData._preprocess_sentence(sent_1, max_len=max_len)
		self.par_2_words = ParData._preprocess_sentence(sent_2, max_len=max_len)
		self.slot_1_words = ParData._preprocess_slots(sent_1, max_len=max_len)
		self.slot_2_words = ParData._preprocess_slots(sent_2, max_len=max_len)
		self.context_1_words = [ParData._preprocess_sentence(c, max_len=max_len) for c in context_1]
		self.context_2_words = [ParData._preprocess_sentence(c, max_len=max_len) for c in context_2]

		self.par_1_vocab = None
		self.par_2_vocab = None
		self.slot_1_vocab = None
		self.slot_2_vocab = None
		self.context_1_vocab = None
		self.context_2_vocab = None

	def translate_to_dict(self, word_dict):
		self.par_1_vocab = ParData._sentence_to_dict(word_dict, self.par_1_words)
		self.par_2_vocab = ParData._sentence_to_dict(word_dict, self.par_2_words)
		self.slot_1_vocab = [ParData._sentence_to_dict(word_dict, s) for s in self.slot_1_words]
		self.slot_2_vocab = [ParData._sentence_to_dict(word_dict, s) for s in self.slot_2_words]
		self.context_1_vocab = [ParData._sentence_to_dict(word_dict, c) for c in self.context_1_words]
		self.context_2_vocab = [ParData._sentence_to_dict(word_dict, c) for c in self.context_2_words]

	def number_words_not_in_dict(self, word_dict):
		missing_words = 0
		all_words = self.par_1_words + self.par_2_words + [w for c in self.slot_1_words for w in c] + [w for c in self.slot_2_words for w in c] + [w for c in self.context_1_words for w in c] + [w for c in self.context_2_words for w in c]
		for w in all_words:
			if w not in word_dict:
				missing_words += 1
		return missing_words, len(all_words)

	def print(self):
		print("+"*100)
		print("Par 1: %s" % (" # ".join(self.par_1_words)))
		print("Par 2: %s" % (" # ".join(self.par_2_words)))
		print("-"*100)
		print("Context 1:\n%s" % ("\n".join(["\t" + " # ".join(c) for c in self.context_1_words])))
		print("Context 2:\n%s" % ("\n".join(["\t" + " # ".join(c) for c in self.context_2_words])))
		print("+"*100)



class DialogueContextParData:

	def __init__(self, paraphrases, contexts, max_len=-1, randomized=True):
		self.paraphrases_words = [ParData._preprocess_sentence(p, max_len=max_len) for p in paraphrases]
		if any([len(p) == 0 for p in self.paraphrases_words]):
			print("[#] WARNING: Found empty paraphrase!")
		self.paraphrases_vocab = None
		self.slot_words = [ParData._preprocess_slots(p, max_len=max_len) for p in paraphrases]
		self.slot_vocab = None
		self.contexts_words = [[ParData._preprocess_sentence(c, max_len=max_len) for c in sub_context] for sub_context in contexts]
		self.contexts_vocab = None
		self.prepare_combinations(randomized=randomized)

	def prepare_combinations(self, randomized=True):
		if not randomized and len(self.paraphrases_words) < 7:
			self.combinations = [(i, j) for i in range(len(self.paraphrases_words)) for j in range(len(self.paraphrases_words))]
			self.combinations = [(i, j) for (i,j) in self.combinations if i != j]
		else:
			self.combinations = []
			for i in range(len(self.paraphrases_words)):
				if randomized:
					new_rand_comb = randint(0,len(self.paraphrases_words)-2)
					if new_rand_comb >= i:
						new_rand_comb += 1 # Preventing that i=j
				else:
					new_rand_comb = (i+1) % len(self.paraphrases_words)
				self.combinations.append((i,new_rand_comb))

	def translate_to_dict(self, word_dict):
		self.word_dict = word_dict
		self.paraphrases_vocab = [ParData._sentence_to_dict(word_dict, p) for p in self.paraphrases_words]
		self.slot_vocab = [[ParData._sentence_to_dict(word_dict, slot_val[1], add_SOS_EOS=False) for slot_val in slot_list] for slot_list in self.slot_words]
		self.contexts_vocab = [[ParData._sentence_to_dict(word_dict, c) for c in sub_context] for sub_context in self.contexts_words]

	def translate_to_BERT(self, tokenizer):
		raise NotImplementedError

	def number_words_not_in_dict(self, word_dict):
		missing_words = 0
		all_words = [w for sent in self.paraphrases_words for w in sent] + [w for cont in self.contexts_words for sent in cont for w in sent]
		for w in all_words:
			if w not in word_dict:
				missing_words += 1
		return missing_words, len(all_words)

	def get_slot_dist(self):
		num_slot_tokens = get_num_slot_tokens()
		start_index = get_slot_token_start_index()
		end_index = start_index - num_slot_tokens

		slot_dist = np.zeros(shape=(num_slot_tokens,), dtype=np.int32)
		slot_count = list()

		if self.paraphrases_vocab is not None:
			for p in self.paraphrases_vocab:
				p_num_slots = 0
				for word_ind in range(min(start_index, end_index+1), max(start_index+1, end_index)):
					p_count_word_ind = np.sum(p == word_ind)
					slot_dist[start_index - word_ind] += p_count_word_ind
					p_num_slots += p_count_word_ind
				slot_count.append(p_num_slots)
		return slot_dist, slot_count

	def print(self):
		print("+"*100)
		print("Paraphrases: ")
		for i in range(min(5, len(self.paraphrases_words))):
			if i > 0:
				print("-"*100)
			print("(%i) \"%s\"" % (i, " ".join(self.paraphrases_words[i])))
			print("Slots:")
			for j in range(len(self.slot_words[i])):
				print("\t[%i] %s: \"%s\"" % (j, self.slot_words[i][j][0], " ".join(self.slot_words[i][j][1])))
			print("Context:")
			for j in range(len(self.contexts_words[i])):
				print("\t/%i/ \"%s\"" % (j, " ".join(self.contexts_words[i][j])))
		print("="*100)
		print("Example: ")
		self.get_view(1).print()
		print("+"*100)

	def num_pars(self):
		return len(self.combinations) # len(self.paraphrases_words)

	def get_view(self, comb_index):
		new_data_view = ContextParData(sent_1 = "", context_1 = [""], sent_2 = "", context_2 = [""]) 
		
		new_data_view.par_1_words = self.paraphrases_words[self.combinations[comb_index][0]]
		new_data_view.par_2_words = self.paraphrases_words[self.combinations[comb_index][1]]
		new_data_view.par_1_vocab = self.paraphrases_vocab[self.combinations[comb_index][0]]
		new_data_view.par_2_vocab = self.paraphrases_vocab[self.combinations[comb_index][1]]

		new_data_view.slot_1_words = self.slot_words[self.combinations[comb_index][0]]
		new_data_view.slot_2_words = self.slot_words[self.combinations[comb_index][1]]
		new_data_view.slot_1_vocab = self.slot_vocab[self.combinations[comb_index][0]]
		new_data_view.slot_2_vocab = self.slot_vocab[self.combinations[comb_index][1]]

		new_data_view.context_1_words = self.contexts_words[self.combinations[comb_index][0]]
		new_data_view.context_2_words = self.contexts_words[self.combinations[comb_index][1]]
		new_data_view.context_1_vocab = self.contexts_vocab[self.combinations[comb_index][0]]
		new_data_view.context_2_vocab = self.contexts_vocab[self.combinations[comb_index][1]]
		
		return new_data_view

	def get_all_sents(self):
		old_combs = self.combinations
		self.combinations = [(i, i) for i in range(len(self.paraphrases_words))]
		data = [self.get_view(i) for i in range(len(self.combinations))]
		self.combinations = old_combs
		return data


class ContextSentData:

	def __init__(self, sent, context, max_len=-1):
		self.sentence_words = ParData._preprocess_sentence(sent, max_len=max_len)
		self.sentence_vocab = None

		self.slot_words = ParData._preprocess_slots(sent, max_len=max_len)
		self.slot_vocab = None

		self.contexts_words = [ParData._preprocess_sentence(c, max_len=max_len) for c in context]
		self.contexts_vocab = None

	def translate_to_dict(self, word_dict):
		self.word_dict = word_dict
		self.sentence_vocab = ParData._sentence_to_dict(word_dict, self.sentence_words)
		self.slot_vocab = [ParData._sentence_to_dict(word_dict, s[1]) for s in self.slot_words]
		self.contexts_vocab = [ParData._sentence_to_dict(word_dict, c) for c in self.contexts_words]

	def translate_to_BERT(self, tokenizer):
		raise NotImplementedError

	def number_words_not_in_dict(self, word_dict, only_sents=False):
		missing_words = 0
		if only_sents:
			all_words = self.sentence_words
		else:
			all_words = self.sentence_words + [w for s in self.slot_words for w in s[1]] + [w for c in self.context_words for w in c]
		for w in all_words:
			if w not in word_dict:
				missing_words += 1
		return missing_words, len(all_words)

	def print(self):
		print("+"*100)
		print("Sentence: %s" % (" ".join(self.sentence_words)))
		print("Slots:")
		for i in range(len(self.slot_words)):
			print("\t[%i] %s: \"%s\"" % (i, self.slot_words[i][0], " ".join(self.slot_words[i][1])))
		print("Context:")
		for i in range(len(self.contexts_words)):
			print("\t/%i/ \"%s\"" % (i, " ".join(self.contexts_words[i])))
		print("+"*100)	



def reconstruct_sentences(embeds, lengths, slot_vals=None, slot_lengths=None, slot_preds=None, list_to_add=None, add_sents_up=True, make_pretty=True):
	id2word = get_id2word_dict()
	sentences = list() if list_to_add is None else list_to_add
	
	if slot_preds is not None:
		assert slot_preds.shape[0] == embeds.shape[0], "[!] ERROR: Batch size does not match for slot predictions and embeddings: %i and %i" % (slot_preds.shape[0], embeds.shape[0])
		assert slot_preds.shape[1] == embeds.shape[1], "[!] ERROR: Sequence length does not match for slot predictions and embeddings: %i and %i" % (slot_preds.shape[1], embeds.shape[1])

	for batch_index in range(embeds.shape[0]):
		p_words = list()
		num_slots = 0
		if len(lengths.shape) == 1:
			for word_index in range(lengths[batch_index]):
				p_words.append(id2word[embeds[batch_index, word_index]])
				if slot_vals is not None and embeds[batch_index, word_index] <= get_slot_token_start_index():
					if slot_preds is None:
						slot_index = num_slots
					else:
						slot_index = np.argmax(slot_preds[batch_index, word_index])
					p_words[-1] = "<%s=\"%s\">" % (p_words[-1][1:-1], " ".join([id2word[slot_vals[batch_index, slot_index, i]] for i in range(slot_lengths[batch_index, slot_index])]))
					num_slots += 1
			if add_sents_up:
				sents = (("[%i] " % (lengths[batch_index])) if make_pretty else "") + " ".join(p_words)
			else:
				sents = p_words
		else:
			lengths = np.reshape(lengths, [lengths.shape[0], -1])
			for sent_index in range(lengths.shape[1]):
				s_words = [(("[%i] " % (lengths[batch_index, sent_index])) if make_pretty else "")]
				for word_index in range(lengths[batch_index, sent_index]):
					s_words.append(id2word[embeds[batch_index, sent_index, word_index]])
				p_words.append((("(Sentence %i) " % (sent_index+1)) if (add_sents_up and make_pretty) else "") + " ".join(s_words))
			if add_sents_up:
				sents = "\n".join(p_words)
			else:
				sents = p_words

		# print("Batch index %i: %s" % (batch_index, sents))
		sentences.append(sents)

	return sentences


if __name__ == "__main__":
	np.random.seed(42)
	random.seed(42)
	# train_data, _, _ = DatasetHandler.load_LM_Wikitext_datasets()
	# train_data, _, _ = DatasetHandler.load_LM_Book_datasets()
	# train_data, _, _ = DatasetHandler.load_LM_Wikitext_datasets(debug_dataset=True)
	# train_data, _, _ = DatasetHandler.load_LM_Dialogue_datasets(debug_dataset=False)
	# train_data, _, _ = DatasetHandler.load_Microsoft_Paraphrase_datasets(debug_dataset=True)
	# train_data, _, _ = DatasetHandler.load_Microsoft_Video_Description_datasets(debug_dataset=True)
	# train_data, _, _ = DatasetHandler.load_Quora_Paraphrase_datasets(debug_dataset=True)
	train_data, val_data, _ = DatasetHandler.load_Dialogue_Paraphrase_datasets(debug_dataset=False)
	# train_data, _, _ = DatasetHandler.load_ContextLM_Book_datasets(debug_dataset=False)
	# DatasetHandler.set_BERT_tokenizer(BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True))
	# print(BERT_TOKENIZER)
	# train_data.set_BERT_tokenizer(BERT_TOKENIZER)
	# train_data, _, _ = DatasetHandler.load_Microsoft_Paraphrase_datasets()
	# train_data, _, _ = DatasetHandler.load_Microsoft_Video_Description_datasets()
	train_data.print_statistics()
	val_data.print_statistics()
	# train_data.print_slot_distribution()
	for _ in range(4):
		batch = train_data.get_batch(4, toTorch=False)

		for e in batch:
			print(e)
	par_1_words, par_1_lengths, par_2_words, par_2_lengths, par_1_slots, par_1_slot_lengths, par_2_slots, par_2_slot_lengths, contexts_1_words, contexts_1_lengths, contexts_2_words, contexts_2_lengths = batch
	print("Reconstructed par 1 words: ")
	print("\n".join(reconstruct_sentences(par_1_words, par_1_lengths, slot_vals=par_1_slots, slot_lengths=par_1_slot_lengths)))
	# print("Reconstructed par 2 words: ")
	# reconstruct_sentences(par_2_words, par_2_lengths)
	# print("Reconstructed contexts 1 words: ")
	# reconstruct_sentences(contexts_1_words, contexts_1_lengths)
	# print("Reconstructed contexts 2 words: ")
	# reconstruct_sentences(contexts_2_words, contexts_2_lengths)

	# for i in [randint(0, len(train_data.data_list)) for _ in range(8)]:
	# 	v_data = train_data.data_list[i].get_view(0)
	# 	print("Reconstructed par 1 words: ")
	# 	reconstruct_sentences(v_data.par_1_vocab, np.array([v_data.par_1_vocab.shape[0]]))
	# 	print("Reconstructed par 2 words: ")
	# 	reconstruct_sentences(v_data.par_2_vocab, np.array([v_data.par_2_vocab.shape[0]]))
	# 	print("Reconstructed contexts 1 words: ")
	# 	reconstruct_sentences(v_data.context_1_vocab, np.array([v_data.context_1_vocab.shape[0]]))
	# 	print("Reconstructed contexts 2 words: ")
	# 	reconstruct_sentences(v_data.context_2_vocab, np.array([v_data.context_2_vocab.shape[0]]))
	for i in [randint(0, len(train_data.data_list)) for _ in range(4)]:
		train_data.data_list[i].print()

