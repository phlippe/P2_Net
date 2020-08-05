import os
import numpy as np
import torch
import json
import re
import sys
from random import shuffle


WORD2VEC_DICT = None
WORD2ID_DICT = None
ID2WORD_DICT = None
WORDVEC_TENSOR = None

SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "UNK"

SLOT_TOKENS = {
	-100: "<area>",
	-101: "<type>",
	-102: "<choice>",
	-103: "<price>",
	-104: "<name>",
	-105: "<phone>",
	-106: "<addr>",
	-107: "<post>",
	-108: "<fee>",
	-109: "<car>",
	-110: "<dest>",
	-111: "<depart>",
	-112: "<arrive>",
	-113: "<leave>",
	-114: "<stay>",
	-115: "<time>",
	-116: "<ref>",
	-117: "<day>",
	-118: "<people>",
	-119: "<stars>",
	-120: "<food>",
	-121: "<id>",
	-122: "<ticket>",
	-123: "<open>",
	-124: "<meal>",
	-125: "<movie>"
}

def get_num_slot_tokens():
	return len(SLOT_TOKENS.keys())

def get_slot_token_start_index():
	return max(SLOT_TOKENS.keys())

def get_slot_tokens():
	global SLOT_TOKENS
	return SLOT_TOKENS



def get_EOS_index():
	_, word2id, _ = load_word2vec_from_file()
	return word2id[EOS_TOKEN]

def get_SOS_index():
	_, word2id, _ = load_word2vec_from_file()
	return word2id[SOS_TOKEN]

def get_UNK_index():
	_, word2id, _ = load_word2vec_from_file()
	return word2id[UNK_TOKEN]

def load_word2vec_from_file(word_file="../data/small_glove_words.txt", numpy_file="../data/small_glove_embed.npy"):
	global WORD2VEC_DICT, WORD2ID_DICT, WORDVEC_TENSOR, SOS_TOKEN, SOS_INDEX, EOS_TOKEN, EOS_INDEX, SLOT_TOKENS
	
	if WORD2VEC_DICT is None or WORD2ID_DICT is None or WORDVEC_TENSOR is None:
		
		word2vec = dict()
		word2id = dict()
		word_vecs = np.load(numpy_file)
		with open(word_file, "r") as f:
			for i, l in enumerate(f):
				word2vec[l.replace("\n","")] = word_vecs[i,:]
		index = 0
		for key, _ in word2vec.items():
			word2id[key] = index
			index += 1

		for slot_index, slot_word in SLOT_TOKENS.items():
			word2id[slot_word] = slot_index

		SOS_INDEX = word2id[SOS_TOKEN]
		EOS_INDEX = word2id[EOS_TOKEN]
		UNK_INDEX = word2id[UNK_TOKEN]

		print("Loaded vocabulary of size " + str(word_vecs.shape[0]))
		WORD2VEC_DICT, WORD2ID_DICT, WORDVEC_TENSOR = word2vec, word2id, word_vecs

	return WORD2VEC_DICT, WORD2ID_DICT, WORDVEC_TENSOR

def get_id2word_dict():
	global ID2WORD_DICT, WORD2ID_DICT
	if ID2WORD_DICT is None:
		if WORD2ID_DICT is None:
			load_word2vec_from_file()
		ID2WORD_DICT = {v:k for k,v in WORD2ID_DICT.items()}
	return ID2WORD_DICT

def save_word2vec_as_GloVe(output_file="small_glove_torchnlp.txt"):
	word2vec, word2id, word_vecs = load_word2vec_from_file()
	s = ""
	for key, val in word2vec.items():
		s += key + " " + " ".join([("%g" % (x)) for x in val]) + "\n"
	with open(output_file, "w") as f:
		f.write(s)


def build_vocab(word_list, glove_path='../data/glove.840B.300d.txt'):
	word2vec = {}
	num_ignored_words = 0
	num_missed_words = 0
	num_found_words = 0
	word_list = set(word_list)
	overall_num_words = len(word_list)

	with open(glove_path, "r") as f:
		lines = f.readlines()
		number_lines = len(lines)
		for i, line in enumerate(lines):
			# if debug_level() == 0:
			print("Processed %4.2f%% of the glove (found %4.2f%% of words yet)" % (100.0 * i / number_lines, 100.0 * num_found_words / overall_num_words), end="\r")
			if num_found_words == overall_num_words:
				break
			word, vec = line.split(' ', 1)
			if word in word_list:
				glove_vec = [float(x) for x in vec.split()]
				word2vec[word] = np.array(glove_vec)
				num_found_words += 1
			else:
				num_ignored_words += 1

	example_missed_words = list()
	for word in word_list:
		if word not in word2vec:
			num_missed_words += 1
			if num_missed_words < 30:
				example_missed_words.append(word)

	print("Created vocabulary with %i words. %i words were ignored from Glove, %i words were not found in embeddings." % (len(word2vec.keys()), num_ignored_words, num_missed_words))
	if num_missed_words > 0:
		print("Example missed words: " + " +++ ".join(example_missed_words))

	return word2vec

load_word2vec_from_file()
