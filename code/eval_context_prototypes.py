import torch
import torch.nn as nn

from pytorch_transformers import *

from sklearn.neighbors import NearestNeighbors
import numpy as np

import argparse
import os 
import sys
import math

from vocab import load_word2vec_from_file, get_UNK_index, get_SOS_index
from mutils import load_model, load_args, get_dict_val, PARAM_CONFIG_FILE, write_dict_to_tensorboard, unsupervised_args_to_params, add_if_not_none
from model_utils import get_device
from unsupervised_task import DialogueModelingTask, LanguageModelingTask, ParaphraseTask, PretrainingTask, ContextAwareDialogueTask, ContextAwareLanguageModelingTask, ContextAwarePretrainingTask
from unsupervised_models.model import ModelUnsupervisedParaphrasingTemplate, ModelUnsupervisedContextParaphrasingTemplate
from data import DialogueParaphraseDataset, DialogueContextParData, set_debug_level
from task import TaskTemplate
from metrics import get_BLEU_batch_stats, get_BLEU_score, get_ROUGE_score, euclidean_distance


def load_data(checkpoint_path):
	train_dir = os.path.join(checkpoint_path, "train_ContextAwareDialogueParaphrase_export")
	val_dir = os.path.join(checkpoint_path, "val_ContextAwareDialogueParaphrase_export")
	train_vals, val_vals = dict(), dict()

	for val_dict, data_dir in zip([train_vals, val_vals], [train_dir, val_dir]):
		val_dict["context_style_attn"] = np.load(os.path.join(data_dir, "context_style_attn_vecs.npz"))["arr_0"]
		val_dict["context_style"] = np.load(os.path.join(data_dir, "context_style_vecs.npz"))["arr_0"]
		val_dict["par_semantics"] = np.load(os.path.join(data_dir, "par_semantic_vecs.npz"))["arr_0"]
		val_dict["par_style"] = np.load(os.path.join(data_dir, "par_style_vecs.npz"))["arr_0"]

		with open(os.path.join(data_dir, "responses.txt"), "r") as f:
			val_dict["responses"] = [l.strip() for l in f.readlines()]
		with open(os.path.join(data_dir, "contexts.txt"), "r") as f:
			val_dict["contexts"] = [l.strip().split("\t") for l in f.readlines()]

	return train_vals, val_vals


def perform_KNN(train_vecs, val_vecs, n_neighbors=1):
	print("-> Preparing KNN...")
	nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(train_vecs)
	print("-> Finding closest neighbors...")
	distances, indices = nbrs.kneighbors(val_vecs)
	return indices


def eval_distance(pred_vecs, gt_vecs):
	dist = np.linalg.norm(pred_vecs - gt_vecs[:,None,:], axis=-1)
	avg_dist_gt = np.linalg.norm(gt_vecs[:,None,:] - gt_vecs[None,:,:], axis=-1).mean()
	
	cosine_similarity = (pred_vecs * gt_vecs[:,None,:]).sum(axis=-1) / (np.linalg.norm(pred_vecs, axis=-1) * np.linalg.norm(gt_vecs[:,None,:], axis=-1))
	avg_cosine_gt = (gt_vecs[None,:,:] * gt_vecs[:,None,:]).sum(axis=-1) / (np.linalg.norm(gt_vecs[:,None,:], axis=-1)**2)
	avg_cosine_gt = avg_cosine_gt.mean()
	print(cosine_similarity.shape)
	
	print("="*100)
	print("Evaluation results")
	for metric_name, metric_avg, metric_vals, clos_fun in zip(["distance", "cosine"], [avg_dist_gt, avg_cosine_gt], [dist, cosine_similarity], [np.min, np.max]):
		print("="*100)
		print("Mean %s among validation vecs: %f" % (metric_name, metric_avg))
		print("Mean %s over all n neighbors: %f" % (metric_name, metric_vals.mean()))
		print("-"*100)
		for i in range(pred_vecs.shape[1]):
			print("Mean %s over %ith neighbor: %f" % (metric_name, i, metric_vals[:,i].mean()))
		print("-"*100)
		print("Mean %s over closest neighbor: %f" % (metric_name, clos_fun(metric_vals, axis=-1).mean()))
	print("="*100)

	best_indices = np.argmax(metric_vals, axis=-1)
	return best_indices


OUR_MODEL = None
def load_our_model(checkpoint_path):
	global OUR_MODEL
	if OUR_MODEL is None:
		args = load_args(checkpoint_path)

		print("-> Loading model...")
		model_params, _ = unsupervised_args_to_params(args)

		_, _, wordvec_tensor = load_word2vec_from_file()
		model = ModelUnsupervisedContextParaphrasingTemplate(model_params, wordvec_tensor)

		print(checkpoint_path)
		_ = load_model(checkpoint_path, model=model, load_best_model=True)
		model = model.to(get_device())

		model.eval()

		OUR_MODEL = model
	return OUR_MODEL


def generate_responses(style_vecs, input_templates, checkpoint_path):
	if len(style_vecs.shape) == 3:
		style_vecs = style_vecs[:,0,:]

	model = load_our_model(checkpoint_path)

	print("-> Loading dataset...")
	dataset = create_dataset(input_templates)
	
	# Prepare metrics
	batch_size = 64
	number_batches = int(math.ceil(len(dataset.data_list) * 1.0 / batch_size))
	hypotheses, references = None, None

	# Evaluation loop
	for batch_ind in range(number_batches):
		# print("Evaluation process: %4.2f%% (batch %i of %i)" % (100.0 * batch_ind / number_batches, batch_ind+1, number_batches), end="\r")

		batch = dataset._data_to_batch([d.get_view(0) for d in dataset.data_list[batch_ind*batch_size:min(len(dataset.data_list), (batch_ind+1) * batch_size)]], toTorch=True)
		par_1_words, par_1_lengths, par_2_words, _, par_1_slots, par_1_slot_lengths, _, _, _, _, _, _ = batch
		batch_style_vecs = torch.from_numpy(style_vecs[batch_ind*batch_size:min(len(dataset.data_list), (batch_ind+1) * batch_size),:]).to(get_device())
		# Evaluate single batch
		with torch.no_grad():
			# TODO: 
			#	3) Run model on batch
			resp_results = model.generate_new_style((par_1_words, par_1_lengths, par_1_slots, par_1_slot_lengths), style_vecs=batch_style_vecs)
			_, _, generated_words, generated_lengths = resp_results

		batch_labels = par_2_words
		if (batch_labels[:,0] == get_SOS_index()).byte().all():
			batch_labels = batch_labels[:,1:]
		unknown_label = (batch_labels == get_UNK_index()).long()
		batch_labels = batch_labels * (1 - unknown_label) + (-1) * unknown_label

		batch_hyp, batch_ref = TaskTemplate._preds_to_sents(batch_labels, generated_words, generated_lengths)
		hypotheses, references = add_if_not_none((batch_hyp, batch_ref), (hypotheses, references))
		# BLEU_score, _ = get_BLEU_score(hypotheses, references)
		# print("BLEU at batch %i: %4.2f%%" % (batch_ind, BLEU_score*100.0))
	
	BLEU_score, prec_per_ngram = get_BLEU_score(hypotheses, references)
	print("="*50)
	print("Achieved BLEU score of: %4.2f%%" % (BLEU_score * 100.0))
	print(prec_per_ngram)
	print("="*50)

	return hypotheses, references, BLEU_score


def create_dataset(templates):
	dataset = DialogueParaphraseDataset(data_path=None, data_type='train', shuffle_data=False)

	par_indices = [t.split("\t",1)[0] for t in templates]
	templates = [t.split("\t",1)[1] for t in templates]
	for sent_index, sent in enumerate(templates):
		if (sent_index+1) < len(templates) and par_indices[sent_index+1] == par_indices[sent_index]:
			partner_sent = templates[sent_index+1] 
		elif sent_index > 0 and par_indices[sent_index] == par_indices[sent_index-1]:
			partner_sent = templates[sent_index-1]
		else:
			print("[!] ERROR: Neither the sentence before or after had the same index...")
			print(par_indices[sent_index])
			sys.exit(1) 
		dataset.data_list.append(DialogueContextParData(paraphrases=[partner_sent, sent], contexts=[["Test"]]*2, max_len=80, randomized=False))

	_, word2id_dict, _ = load_word2vec_from_file()
	dataset.set_vocabulary(word2id_dict)
	dataset.reset_index()
	return dataset



if __name__ == '__main__':
	np.random.seed(42) 

	parser = argparse.ArgumentParser()

	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints is stored", type=str)
	
	args = parser.parse_args()
	set_debug_level(2)

	print("Loading data...")
	train_vals, val_vals = load_data(args.checkpoint_path)

	all_experiment_results = {}

	print("Generating Baselines...")
	print("\n== Reconstructing GT ==")
	gt_vecs = np.concatenate([val_vals["par_style"], val_vals["context_style"]], axis=1)
	_, _, gt_reconst_BLEU = generate_responses(style_vecs=gt_vecs, input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)

	print("\n== GT style dropout ==")
	dropout_factor = (val_vals["par_style"].shape[-1] + val_vals["context_style"].shape[-1]) * 1.0 / val_vals["context_style"].shape[-1]
	dropouted_style_vecs = np.concatenate([val_vals["par_style"] * 0.0, val_vals["context_style"] * dropout_factor], axis=1)
	pred_responses, gt_responses, dropout_BLEU = generate_responses(style_vecs=dropouted_style_vecs, input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)
	
	print("\n== Random paraphrase style selection ==")
	par_style_random_selection = train_vals["par_style"][np.random.randint(0, train_vals["par_style"].shape[0], size=(gt_vecs.shape[0])),:]
	combined_style_vecs = np.concatenate([par_style_random_selection, val_vals["context_style"]], axis=1)
	_, _, random_sel_BLEU = generate_responses(style_vecs=combined_style_vecs, input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)
	
	print("\n== Random context style selection ==")
	context_style_random_selection = train_vals["context_style"][np.random.randint(0, train_vals["context_style"].shape[0], size=(gt_vecs.shape[0])),:]
	combined_style_vecs = np.concatenate([par_style_random_selection, context_style_random_selection], axis=1)
	_, _, random_context_BLEU = generate_responses(style_vecs=combined_style_vecs, input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)
	
	print("\n== Random style from normal distribution ==")
	_, _, random_normal_BLEU = generate_responses(style_vecs=np.random.normal(size=gt_vecs.shape).astype(np.float32), input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)

	print("\n== Closest kNN context style ==")
	indices = perform_KNN(train_vecs=train_vals["context_style"], val_vecs=val_vals["context_style"], n_neighbors=4)
	pred_vecs = train_vals["par_style"][indices, :]
	_ = eval_distance(pred_vecs=pred_vecs, gt_vecs=val_vals["par_style"])
	kNN_style_vecs = np.concatenate([pred_vecs[:,0,:], val_vals["context_style"]], axis=1)
	_, _, context_kNN_BLEU = generate_responses(style_vecs=kNN_style_vecs, input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)


	all_experiment_results["gt reconstruction"] = gt_reconst_BLEU
	all_experiment_results["template dropout"] = dropout_BLEU
	all_experiment_results["random selection"] = random_sel_BLEU
	all_experiment_results["random context"] = random_context_BLEU
	all_experiment_results["random normal sampling"] = random_normal_BLEU

	s = ("="*100) + "\n"
	s += ("Experiment results") + "\n"
	s += ("-"*100) + "\n"
	exp_keys = list(all_experiment_results.keys())
	exp_keys = sorted(exp_keys, key=lambda x: all_experiment_results[x])
	for key in exp_keys:
		s += ("%s: %4.2f%%" % (key, all_experiment_results[key]*100.0)) + "\n"
	s += ("="*100) + "\n"
	print(s)

	print("Exporting results...")
	pred_dir = os.path.join(args.checkpoint_path, "predictions_ContextAwareDialogueParaphrase")
	os.makedirs(pred_dir, exist_ok=True)

	with open(os.path.join(pred_dir, "results.txt"), "w") as f:
		f.write(s)

	np.savez_compressed(os.path.join(pred_dir, "predicted_style_vecs.npz"), pred_vecs)
	with open(os.path.join(pred_dir, "predicted_responses.txt"), "w") as f:
		for context, pred, gt in zip(val_vals["contexts"], pred_responses, gt_responses):
			f.write("%s\nContext: %s\n%s\nPrediction: %s\nGround Truth: %s\n%s\n\n" % ("="*150, "\n".join(context), "-"*150, " ".join(pred), " ".join(gt), "="*150))

	