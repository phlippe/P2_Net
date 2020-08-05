import torch 
import torch.nn as nn
import argparse
import random
import numpy as np
import math
import datetime
import os
import sys
import json
import pickle
import copy
from decimal import Decimal
import math
import scipy
from glob import glob
from shutil import copyfile

from supervised_models.model import ModelSupervisedParaphrasingTemplate
from data import DatasetHandler, debug_level, set_debug_level, DatasetTemplate
from vocab import load_word2vec_from_file

PARAM_CONFIG_FILE = "param_config.pik"


###################
## MODEL LOADING ##
###################

def load_model(checkpoint_path, model=None, optimizer=None, lr_scheduler=None, load_best_model=False):
	if os.path.isdir(checkpoint_path):
		checkpoint_files = sorted(glob(os.path.join(checkpoint_path, "*.tar")))
		if len(checkpoint_files) == 0:
			return dict()
		checkpoint_file = checkpoint_files[-1]
	else:
		checkpoint_file = checkpoint_path
	print("Loading checkpoint \"" + str(checkpoint_file) + "\"")
	if torch.cuda.is_available():
		checkpoint = torch.load(checkpoint_file)
	else:
		checkpoint = torch.load(checkpoint_file, map_location='cpu')
	
	# If best model should be loaded, look for it if checkpoint_path is a directory
	if os.path.isdir(checkpoint_path) and load_best_model:
		return load_model(os.path.join(checkpoint_path.rsplit("/",1)[0],checkpoint["best_save_dict"]["file"].split("/")[-1]), model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, load_best_model=False)

	if model is not None:
		pretrained_model_dict = {key: val for key, val in checkpoint['model_state_dict'].items()} #  if not key.startswith("embeddings")
		# print("Load model weights: " + str(pretrained_model_dict.keys()))
		model_dict = model.state_dict()
		unchanged_keys = [key for key in model_dict.keys() if key not in pretrained_model_dict.keys()]
		# print("Unchanged weights: " + str(unchanged_keys))
		model_dict.update(pretrained_model_dict)
		model.load_state_dict(model_dict)
		# model.load_state_dict(checkpoint['model_state_dict'])
	if optimizer is not None and 'optimizer_state_dict' in checkpoint:
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	if lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
		lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	
	add_param_dict = dict()
	for key, val in checkpoint.items():
		if "state_dict" not in key:
			add_param_dict[key] = val
	return add_param_dict


def load_model_from_args(args, checkpoint_path=None, load_best_model=False):
	model_params, optimizer_params = args_to_params(args)
	_, _, wordvec_tensor = load_word2vec_from_file()
	model = ModelSupervisedParaphrasingTemplate(model_params, wordvec_tensor).to(get_device())
	if checkpoint_path is not None:
		load_model(checkpoint_path, model=model, load_best_model=load_best_model)
	return model


def load_args(checkpoint_path):
	if os.path.isfile(checkpoint_path):
		checkpoint_path = checkpoint_path.rsplit("/",1)[0]
	param_file_path = os.path.join(checkpoint_path, PARAM_CONFIG_FILE)
	if not os.path.exists(param_file_path):
		print("[!] ERROR: Could not find parameter config file: " + str(param_file_path))
	with open(param_file_path, "rb") as f:
		print("Loading parameter configuration from \"" + str(param_file_path) + "\"")
		args = pickle.load(f)
	return args


def general_args_to_params(args, model_params=None):

	optimizer_params = {
		"optimizer": args.optimizer,
		"lr": args.learning_rate,
		"weight_decay": args.weight_decay,
		"lr_decay_factor": args.lr_decay,
		"lr_decay_step": args.lr_decay_step,
		"momentum": args.momentum if hasattr(args, "momentum") else 0.0
	}

	# Set seed
	np.random.seed(args.seed)
	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available: 
		torch.cuda.manual_seed_all(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	return model_params, optimizer_params


def unsupervised_args_to_params(args):
	_, _, wordvec_tensor = load_word2vec_from_file()

	# Define model parameters
	model_params = {
		"embed_word_dim": 300,
		"embed_dropout": args.embed_dropout,
		"finetune_embeds": args.finetune_embeds,
		"switch_rate": args.switch_rate,
		"teacher_forcing_ratio": args.teacher_forcing_ratio,
		"teacher_forcing_annealing": args.teacher_forcing_annealing,
		"VAE_loss_scaling": args.VAE_loss_scaling,
		"VAE_annealing_iters": args.VAE_annealing_iters,
		"VAE_annealing_func": args.VAE_annealing_func,
		"VAE_scheduler": args.VAE_scheduler,
		"cosine_loss_scaling": args.cosine_loss_scaling,
		"cosine_counter_loss": args.cosine_counter_loss,
		"style_loss_scaling": args.style_loss_scaling,
		"style_loss_module": args.style_loss_module,
		"style_loss_stop_grads": args.style_loss_stop_grads,
		"style_loss_annealing_iters": args.style_loss_annealing_iters,
		"semantics_dropout": args.semantics_dropout,
		"semantic_full_dropout": args.semantic_full_dropout,
		"semantic_size": args.semantic_size,
		"style_size": args.style_size,
		"response_style_size": args.response_style_size if args.response_style_size > 0 else args.style_size,
		"num_context_turns": args.num_context_turns,
		"pure_style_loss": args.pure_style_loss,
		"positional_embedding_factor": args.positional_embedding_factor,
		"pretraining_iterations": args.pretraining_iterations,
		"pretraining_second_task": args.pretraining_second_task,
		"only_paraphrasing": args.only_paraphrasing,
		"slot_value_embeddings": not args.no_slot_value_embeddings,
		"use_semantic_specific_attn": args.use_semantic_specific_attn,
		"style_exponential_dropout": args.style_exponential_dropout,
		"style_full_dropout": args.style_full_dropout
	}

	model_params["slot_encoder_module"] = {
		"use_CBOW": args.slots_CBOW,
		"hidden_size": model_params["embed_word_dim"]
	}

	model_params["encoder_module"] = {
		"type": args.encoder_model,
		"input_size": model_params["embed_word_dim"],
		"hidden_size": args.encoder_hidden_size,
		"num_layers": args.encoder_num_layers,
		"dropout": args.encoder_dropout,
		"semantic_size": model_params["semantic_size"],
		"style_size": model_params["style_size"],
		"response_style_size": model_params["response_style_size"],
		"share_attention": not args.encoder_separate_attentions,
		"use_prototype_styles": args.use_prototype_styles,
		"num_prototypes": args.num_prototypes,
		"use_semantic_for_context_proto": args.use_semantic_for_context_proto,
		"no_prototypes_for_context": args.no_prototypes_for_context
	}

	model_params["decoder_module"] = {
		"type": args.decoder_model,
		"num_classes": wordvec_tensor.shape[0],
		"hidden_size": args.decoder_hidden_size,
		"num_layers": args.decoder_num_layers,
		"input_dropout": args.decoder_input_dropout,
		"lstm_dropout": args.decoder_lstm_dropout,
		"output_dropout": args.decoder_output_dropout,
		"concat_features": args.decoder_concat_features,
		"lstm_additional_input": args.decoder_lstm_additional_input,
		"semantic_size": model_params["semantic_size"],
		"style_size": (model_params["style_size"] if True or not model_params["encoder_module"]["use_prototype_styles"] else 0) + model_params["response_style_size"]
	}

	model_params, optimizer_params = general_args_to_params(args, model_params)	

	return model_params, optimizer_params


def discriminator_args_to_params(args):
	_, _, wordvec_tensor = load_word2vec_from_file()

	# Define model parameters
	model_params = {
		"embed_word_dim": 300,
		"embed_dropout": args.embed_dropout,
		"finetune_embeds": args.finetune_embeds,
		"semantic_size": args.semantic_size,
		"style_size": args.style_size,
		"use_VAE": args.use_VAE,
		"use_semantic_specific_attn": args.use_semantic_specific_attn,
		"num_context_turns": args.num_context_turns,
		"slot_value_embeddings": args.slot_value_embeddings,
		"use_small_dataset": args.use_small_dataset
	}

	model_params["slot_encoder_module"] = {
		"use_CBOW": args.slots_CBOW,
		"hidden_size": model_params["embed_word_dim"]
	}

	model_params["encoder_module"] = {
		"type": args.encoder_model,
		"input_size": model_params["embed_word_dim"],
		"hidden_size": args.encoder_hidden_size,
		"num_layers": args.encoder_num_layers,
		"dropout": args.encoder_dropout,
		"semantic_size": model_params["semantic_size"],
		"style_size": model_params["style_size"],
		"share_attention": not args.encoder_separate_attentions
	}

	model_params["discriminator_module"] = {
		"type": args.discriminator_model,
		"hidden_size": args.discriminator_hidden_size,
		"num_hidden_layers": args.discriminator_num_layers,
		"input_dropout": args.discriminator_dropout,
		"semantic_size": model_params["semantic_size"],
		"style_size": model_params["style_size"]
	}

	model_params, optimizer_params = general_args_to_params(args, model_params)	

	return model_params, optimizer_params

def supervised_args_to_params(args):
	_, _, wordvec_tensor = load_word2vec_from_file()

	# Define model parameters
	model_params = {
		"embed_word_dim": 300,
		"embed_dropout": args.embed_dropout,
		"finetune_embeds": args.finetune_embeds,
		"share_encoder": args.share_encoder,
		"teacher_forcing_ratio": args.teacher_forcing_ratio,
		"teacher_forcing_annealing": args.teacher_forcing_annealing
	}

	model_params["dialogue_module"] = {
		"type": args.dialogue_model,
		"input_size": model_params["embed_word_dim"],
		"hidden_size": args.dialogue_hidden_size,
		"num_layers": args.dialogue_num_layers,
		"dropout": args.dialogue_dropout,
		"bert_model": args.dialogue_bert_model,
		"bert_finetune_layers": args.dialogue_bert_finetune_layers
	}

	model_params["template_module"] = {
		"type": args.template_model,
		"input_size": model_params["embed_word_dim"],
		"hidden_size": args.template_hidden_size,
		"num_layers": args.template_num_layers,
		"dropout": args.template_dropout,
		"bert_model": args.template_bert_model,
		"bert_finetune_layers": args.template_bert_finetune_layers
	}

	model_params["paraphrase_module"] = {
		"type": args.paraphrase_model,
		"num_classes": wordvec_tensor.shape[0],
		"hidden_size": args.paraphrase_hidden_size,
		"num_layers": args.paraphrase_num_layers,
		"input_dropout": args.paraphrase_input_dropout,
		"lstm_dropout": args.paraphrase_lstm_dropout,
		"output_dropout": args.paraphrase_output_dropout
	}

	model_params, optimizer_params = general_args_to_params(args, model_params=model_params)

	return model_params, optimizer_params


def get_dict_val(checkpoint_dict, key, default_val):
	if key in checkpoint_dict:
		return checkpoint_dict[key]
	else:
		return default_val

def add_if_not_none(new_val, old_val):
	if old_val is None or ((isinstance(old_val, tuple) or isinstance(old_val, list)) and all([v is None for v in old_val])):
		return new_val
	elif isinstance(old_val, tuple) and isinstance(new_val, tuple):
		return tuple([old_val[i] + new_val[i] for i in range(len(old_val))])
	else:
		return old_val + new_val


####################################
## VISUALIZATION WITH TENSORBOARD ##
####################################

def write_dict_to_tensorboard(writer, val_dict, base_name, iteration):
	for name, val in val_dict.items():
		if isinstance(val, dict):
			write_dict_to_tensorboard(writer, val, base_name=base_name+"/"+name, iteration=iteration)
		elif isinstance(val, (list, np.ndarray)):
			continue
		elif isinstance(val, (int, float)):
			writer.add_scalar(base_name + "/" + name, val, iteration)
		else:
			if debug_level() == 0:
				print("Skipping output \""+str(name) + "\" of value " + str(val) + "(%s)" % (val.__class__.__name__))

