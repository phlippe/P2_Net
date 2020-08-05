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


def semantic_from_neighbor(semantic_vectors, templates):
	par_indices = [t.split("\t",1)[0] for t in templates]
	partner_sent_indices = np.zeros(shape=(semantic_vectors.shape[0],), dtype=np.int32)

	for sent_index in range(len(templates)):
		if (sent_index+1) < len(templates) and par_indices[sent_index+1] == par_indices[sent_index]:
			partner_sent_indices[sent_index] = sent_index+1
		elif sent_index > 0 and par_indices[sent_index] == par_indices[sent_index-1]:
			partner_sent_indices[sent_index] = sent_index-1

	partner_semantic = semantic_vectors[partner_sent_indices]
	return partner_semantic


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

# PyTorch-Transformers has a unified API
# for 7 transformer architectures and 30 pretrained weights.
#                        Model          | Tokenizer          | Pretrained weights shortcut
TRANSFORMER_MODELS =   [(BertModel,       BertTokenizer,      'bert-base-uncased'),
						(OpenAIGPTModel,  OpenAIGPTTokenizer, 'openai-gpt'),
						(GPT2Model,       GPT2Tokenizer,      'gpt2'),
						(TransfoXLModel,  TransfoXLTokenizer, 'transfo-xl-wt103'),
						(XLNetModel,      XLNetTokenizer,     'xlnet-base-cased'),
						(XLMModel,        XLMTokenizer,       'xlm-mlm-enfr-1024'),
						(RobertaModel,    RobertaTokenizer,   'roberta-base')]
def encode_by_transformer(input_sentences, transformer_model=TRANSFORMER_MODELS[0], export_checkpoint=None, postfix="", overwrite=False):
	model_class, tokenizer_class, pretrained_weights = transformer_model
	
	if export_checkpoint is not None:
		export_file = os.path.join(export_checkpoint, "transformer_vecs_" + pretrained_weights + postfix + ".npz")
		if os.path.isfile(export_file) and not overwrite:
			print("-> Found stored exports at %s, trying to load them..." % (export_file))
			return np.load(export_file)

	# Load pretrained model/tokenizer
	tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
	model = model_class.from_pretrained(pretrained_weights).to(get_device())

	batch_size = 64
	number_batches = int(math.ceil(len(input_sentences) * 1.0 / batch_size))
	output_vecs = {"max": list(), "min": list(), "avg": list(), "orig": list()}
	for batch_index in range(number_batches):
		# print("Transforming done by %4.2f%%" % (100.0 * batch_index / number_batches), end="\r")
		batch = input_sentences[batch_index*batch_size:min((batch_index+1)*batch_size, len(input_sentences))]
		batch = ["[CLS] " + s.replace(" unk ", " [UNK] ") + " [SEP]" for s in batch]
		input_ids = [tokenizer.encode(s) for s in batch]
		max_input_len = max([len(ids) for ids in input_ids])
		input_ids = [ids + [0]*(max_input_len - len(ids)) for ids in input_ids]
		attention_mask = [[float(x > 0) for x in ids] for ids in input_ids]
		input_ids = torch.tensor(input_ids).to(get_device())
		attention_mask = torch.tensor(attention_mask).to(get_device())
		# Encode text
		# print(batch)
		# input_ids = torch.tensor(tokenizer.encode(batch))
		with torch.no_grad():
			last_hidden_states, pooler_output = model(input_ids, attention_mask=attention_mask)  # Models outputs are now tuples

			output_vecs["orig"].append(pooler_output.cpu().numpy())
			output_vecs["max"].append(last_hidden_states.max(dim=1)[0].cpu().numpy())
			output_vecs["min"].append(last_hidden_states.min(dim=1)[0].cpu().numpy())
			output_vecs["avg"].append(last_hidden_states.mean(dim=1).cpu().numpy())

	output_vecs = {key: np.concatenate(val, axis=0) for key, val in output_vecs.items()}

	if export_checkpoint is not None:
		print("-> Exporting results to %s..." % export_file)
		np.savez(export_file, **output_vecs)

	return output_vecs




if __name__ == '__main__':
	np.random.seed(42) 

	parser = argparse.ArgumentParser()

	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints is stored", type=str)
	parser.add_argument("--num_neighbors", help="Number of neighbors used in KNN", type=int, default=1)
	parser.add_argument("--use_BERT", help="Whether to also try variants of BERT or not.", action="store_true")

	args = parser.parse_args()
	set_debug_level(2)

	print("Loading data...")
	train_vals, val_vals = load_data(args.checkpoint_path)

	if args.use_BERT:
		print("Running transformers...")
		transformer_checkpoint = args.checkpoint_path[:-1] if args.checkpoint_path[-1] == "/" else ""
		transformer_checkpoint = transformer_checkpoint.rsplit("/",1)[0] if transformer_checkpoint.rsplit("/",1)[-1].startswith("experiment_") else transformer_checkpoint
		transformer_train_vecs = encode_by_transformer([c[0] for c in train_vals["contexts"]], export_checkpoint=transformer_checkpoint, postfix="_train", overwrite=False)
		transformer_val_vecs = encode_by_transformer([c[0] for c in val_vals["contexts"]], export_checkpoint=transformer_checkpoint, postfix="_val", overwrite=False)

	all_experiment_results = {}

	print("Generating Baselines...")
	gt_vecs = val_vals["par_style"]
	_, _, random_sel_BLEU = generate_responses(style_vecs=train_vals["par_style"][np.random.randint(0, train_vals["par_style"].shape[0], size=(gt_vecs.shape[0],1)),:], input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)
	_, _, gt_reconst_BLEU = generate_responses(style_vecs=gt_vecs[:,None,:], input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)
	_, _, random_normal_BLEU = generate_responses(style_vecs=np.random.normal(size=gt_vecs[:,None,:].shape).astype(np.float32), input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)

	all_experiment_results["random selection"] = random_sel_BLEU
	all_experiment_results["gt reconstruction"] = gt_reconst_BLEU
	all_experiment_results["random normal sampling"] = random_normal_BLEU

	train_vec_model_context = train_vals["context_style_attn"].squeeze()
	val_vec_model_context = val_vals["context_style_attn"].squeeze()
	train_vec_semantic = np.concatenate([train_vals["context_style_attn"].squeeze(), train_vals["par_semantics"]], axis=-1)
	val_vec_semantic = np.concatenate([val_vals["context_style_attn"].squeeze(), val_vals["par_semantics"]], axis=-1)
	train_vec_pure_semantic = train_vals["par_semantics"]
	val_vec_pure_semantic = val_vals["par_semantics"]
	train_vec_semantic_from_neighbor = semantic_from_neighbor(train_vals["par_semantics"], train_vals["responses"])
	val_vec_semantic_from_neighbor = semantic_from_neighbor(val_vals["par_semantics"], val_vals["responses"])
	train_vec_semantic_from_neighbor_context = np.concatenate([train_vals["context_style_attn"].squeeze(), train_vec_semantic_from_neighbor], axis=-1)
	val_vec_semantic_from_neighbor_context = np.concatenate([val_vals["context_style_attn"].squeeze(), val_vec_semantic_from_neighbor], axis=-1)

	if args.use_BERT:
		BERT_train_vecs = [transformer_train_vecs["orig"], transformer_train_vecs["max"], transformer_train_vecs["min"], transformer_train_vecs["avg"]]
		BERT_val_vecs = [transformer_val_vecs["orig"], transformer_val_vecs["max"], transformer_val_vecs["min"], transformer_val_vecs["avg"]]
		BERT_exp_names = ["BERT orig", "BERT max", "BERT min", "BERT avg"]
	else:
		BERT_train_vecs, BERT_val_vecs, BERT_exp_names = [], [], []
	for train_vecs, val_vecs, exp_name in zip([train_vec_model_context, train_vec_semantic, train_vec_pure_semantic, train_vec_semantic_from_neighbor, train_vec_semantic_from_neighbor_context] + BERT_train_vecs,
											  [val_vec_model_context, val_vec_semantic, val_vec_pure_semantic, val_vec_semantic_from_neighbor, val_vec_semantic_from_neighbor_context] + BERT_val_vecs,
											  ["Model attention on context", "With semantic", "Pure semantic", "Semantic from neighbor", "With neighbor's semantic"] + BERT_exp_names):

		print("\n\n===== Starting experiment %s =====" % exp_name)
		print("Performing KNN...")
		indices = perform_KNN(train_vecs = train_vecs, 
							  val_vecs = val_vecs, 
							  n_neighbors = args.num_neighbors)

		print("Evaluating...")
		print(indices[:32, 0])
		pred_vecs = train_vals["par_style"][indices, :]
		# pred_vecs = gt_vecs[:,None,:]
		best_indices = eval_distance(pred_vecs=pred_vecs, gt_vecs=gt_vecs)
		best_pred_vecs = pred_vecs[np.arange(pred_vecs.shape[0]), best_indices]

		print("Generating responses...")
		pred_responses, gt_responses, exp_BLEU = generate_responses(style_vecs=pred_vecs, input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)
		_, _, exp_best_BLEU = generate_responses(style_vecs=best_pred_vecs, input_templates=val_vals["responses"], checkpoint_path=args.checkpoint_path)

		all_experiment_results[exp_name] = exp_BLEU
		all_experiment_results[exp_name + " (best out of %i)"%args.num_neighbors] = exp_best_BLEU

	print("="*100)
	print("Experiment results")
	print("-"*100)
	exp_keys = list(all_experiment_results.keys())
	exp_keys = sorted(exp_keys, key=lambda x: all_experiment_results[x])
	for key in exp_keys:
		print("%s: %4.2f%%" % (key, all_experiment_results[key]*100.0))
	print("="*100)

	print("Exporting results...")
	pred_dir = os.path.join(args.checkpoint_path, "predictions_ContextAwareDialogueParaphrase")
	os.makedirs(pred_dir, exist_ok=True)

	np.savez_compressed(os.path.join(pred_dir, "predicted_style_vecs.npz"), pred_vecs)
	with open(os.path.join(pred_dir, "predicted_responses.txt"), "w") as f:
		for pred, gt in zip(pred_responses, gt_responses):
			f.write("Prediction: %s\nGround Truth: %s\n\n" % (pred, gt))

	