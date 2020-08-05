import os
import numpy as np
import torch
import json
import re
import sys
from data import DatasetHandler
from vocab import *
import matplotlib.pyplot as plt

SMALL_GLOVE_WORDS = "../data/small_glove_words.txt"
SMALL_GLOVE_NUMPY = "../data/small_glove_embed.npy"

def create_default_vocab():
	global SMALL_GLOVE_WORDS, SMALL_GLOVE_NUMPY

	if os.path.isfile(SMALL_GLOVE_WORDS) and os.path.isfile(SMALL_GLOVE_NUMPY):
		return

	with open(SMALL_GLOVE_WORDS, "w") as f:
		f.write("\n".join(['<s>', '</s>', '<p>', 'UNK']))

	np_word_array = np.zeros(shape=(4,300), dtype=np.float32)
	np.save(SMALL_GLOVE_NUMPY, np_word_array)



def create_word2vec_vocab():
	global SMALL_GLOVE_WORDS, SMALL_GLOVE_NUMPY
	
	dataset_methods = [getattr(DatasetHandler, method) for method in dir(DatasetHandler) 
					   if method.startswith("load") and callable(getattr(DatasetHandler, method))]
	dataset_methods = [DatasetHandler.load_Dialogue_Paraphrase_datasets, DatasetHandler.load_ContextLM_Book_datasets, DatasetHandler.load_Quora_Paraphrase_datasets]
	print("Found the following dataset loading function: " + str(dataset_methods))
	dataset_list = list()
	for m in dataset_methods:
		dataset_list += list(m())
	print("Loaded " + str(len(dataset_list)) + " datasets")

	datasets_word_list = list()
	word_freq_dict = dict()
	for dataset_num, d in enumerate(dataset_list):
		d_word_list, d_word_freq = d.get_word_list()
		datasets_word_list += d_word_list
		for key, val in d_word_freq.items():
			if key not in word_freq_dict:
				word_freq_dict[key] = val
			else:
				word_freq_dict[key] += val

	x_indices = list(range(20))
	thresholds = [2**(x) for x in x_indices]
	print("Number of words: " + str(len(word_freq_dict.keys())))
	for t_index, t in enumerate(thresholds):
		s = "Number of words with at least %i occurrences: %i." % (t, len([v for key, v in word_freq_dict.items() if v >= t]))
		if t_index > 0:
			loosing_words = ["\"%s\"" % (key) for key, v in word_freq_dict.items() if (v < t and v >= thresholds[t_index-1])]
			if len(loosing_words) > 0:
				s += " Loosing words such as %s,..." % (",".join(loosing_words[:min(5, len(loosing_words))]))
		print(s)
	
	# WARNING: CRASHES MAC-OS!
	# plt.bar(x_indices, [len([v for key, v in word_freq_dict.items() if v >= t]) for t in thresholds])
	# plt.xticks(ticks=x_indices, labels=[str(t) for t in thresholds])
	# plt.show()

	datasets_word_list = [key for key, v in word_freq_dict.items() if v >= 10]

	if os.path.isfile(SMALL_GLOVE_WORDS):
		old_glove = [l.strip() for l in open(SMALL_GLOVE_WORDS)]
		print("Found " + str(len(old_glove)) + " words in old GloVe embeddings")
	else:
		old_glove = []

	word_list = list(set(datasets_word_list + old_glove + ['<s>', '</s>', '<p>', 'UNK']))
	if 'unk' in word_list: # Remove small version of unknown token. 
		word_list.remove('unk')
	# Allow both with "-" and without "-" words to cover all possible preprocessing steps
	print("Created word list with " + str(len(word_list)) + " words. Checking for \"-\" confusion...")
	for word in word_list:
		if "-" in word:
			for w in word.split("-"):
				if len(w) >= 1 and w not in word_list:
					word_list.append(w)
	print("Number of unique words in all datasets: " + str(len(word_list)))

	voc = build_vocab(word_list)
	np_word_list = []
	with open(SMALL_GLOVE_WORDS, 'w') as f:
		# json.dump(voc, f)
		for key, val in voc.items():
			f.write(key + "\n")
			np_word_list.append(val)
	np_word_array = np.stack(np_word_list, axis=0)
	np.save(SMALL_GLOVE_NUMPY, np_word_array)


if __name__ == '__main__':
	create_default_vocab()
	create_word2vec_vocab()
	save_word2vec_as_GloVe()