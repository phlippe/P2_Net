import torch 
import torch.nn as nn
import argparse
import random
import numpy as np
import math
import datetime
import os, shutil
import sys
import json
import pickle
import time
from glob import glob

from tensorboardX import SummaryWriter

from model_utils import ModelTemplate, get_device, ModelTypes
from data import debug_level, set_debug_level
from vocab import load_word2vec_from_file
from mutils import load_model, load_args, get_dict_val, PARAM_CONFIG_FILE, write_dict_to_tensorboard


class TrainTemplate:


	OPTIMIZER_SGD = 0
	OPTIMIZER_ADAM = 1


	def __init__(self, model_params, optimizer_params, batch_size, checkpoint_path, debug=False):
		## Load vocabulary
		_, self.word2id, wordvec_tensor = load_word2vec_from_file()
		self.batch_size = batch_size
		## Load model
		self.model = self._create_model(model_params, wordvec_tensor).to(get_device())
		## Load task
		self.task = self._create_task(model_params, debug=debug)
		## Load optimizer and checkpoints
		self._create_optimizer(optimizer_params)
		self._prepare_checkpoint(checkpoint_path)


	def _create_model(self, model_params, wordvec_tensor):
		raise NotImplementedError


	def _create_task(self, model_params, debug=False):
		raise NotImplementedError


	def _get_all_parameters(self):
		parameters_to_optimize = list(self.model.parameters())
		parameters_to_optimize = [p for p in parameters_to_optimize if p.requires_grad]
		return parameters_to_optimize


	def _create_optimizer(self, optimizer_params):
		parameters_to_optimize = self._get_all_parameters()
		if optimizer_params["optimizer"] == TrainTemplate.OPTIMIZER_SGD:
			self.optimizer = torch.optim.SGD(parameters_to_optimize, 
											 lr=optimizer_params["lr"], 
											 weight_decay=optimizer_params["weight_decay"],
											 momentum=optimizer_params["momentum"])
		elif optimizer_params["optimizer"] == TrainTemplate.OPTIMIZER_ADAM:
			self.optimizer = torch.optim.Adam(parameters_to_optimize, 
											  lr=optimizer_params["lr"],
											  weight_decay=optimizer_params["weight_decay"])
		else:
			print("[!] ERROR: Unknown optimizer: " + str(optimizer_params["optimizer"]))
			sys.exit(1)
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, optimizer_params["lr_decay_step"], gamma=optimizer_params["lr_decay_factor"])


	def _prepare_checkpoint(self, checkpoint_path):
		if checkpoint_path is None:
			current_date = datetime.datetime.now()
			checkpoint_path = "checkpoints/%02d_%02d_%02d__%02d_%02d_%02d/" % (current_date.day, current_date.month, current_date.year, current_date.hour, current_date.minute, current_date.second)
		if not os.path.exists(checkpoint_path):
			os.makedirs(checkpoint_path)
		self.checkpoint_path = checkpoint_path


	def train_model(self, max_iterations=1e6, loss_freq=50, eval_freq=2000, save_freq=1e5, max_gradient_norm=10.0, no_model_checkpoints=False):

		# Setup training parameters
		parameters_to_optimize = self._get_all_parameters()
		print("Trainable model parameters: " + str([name for name, p in self.model.named_parameters() if p.requires_grad]))
		checkpoint_dict = self.load_recent_model()
		start_iter = get_dict_val(checkpoint_dict, "iteration", 0)
		evaluation_dict = get_dict_val(checkpoint_dict, "evaluation_dict", dict())
		best_save_dict = get_dict_val(checkpoint_dict, "best_save_dict", {"file": None, "metric": -1, "detailed_metrics": None})
		best_save_iter = best_save_dict["file"]
		last_save = None if start_iter == 0 else self.get_checkpoint_filename(start_iter)
		if last_save is not None and not os.path.isfile(last_save):
			print("[!] WARNING: Could not find last checkpoint file specified as " + last_save)
			last_save = None

		writer = SummaryWriter(self.checkpoint_path)

		# Function for saving model. Add here in the dictionary necessary parameters that should be saved
		def save_train_model(iteration, only_weights=True):
			if no_model_checkpoints:
				return
			checkpoint_dict = {
				"iteration": iteration,
				"best_save_dict": best_save_dict
			}
			if only_weights:
				self.save_model(iteration, checkpoint_dict, save_optimizer=False)
			else:
				self.save_model(iteration, checkpoint_dict, save_optimizer=True)

		def export_weight_parameters(iteration):
			# Export weight distributions
			for name, param in self.model.named_parameters():
				if not param.requires_grad:
					continue
				writer.add_histogram(name, param.data.view(-1), global_step=iteration)
			
		time_per_step = np.zeros((2,), dtype=np.float32)
		train_losses, train_accs = [], []

		if start_iter == 0 and writer is not None:
			export_weight_parameters(0)
		# Try-catch if user terminates
		try:
			print("="*50 + "\nStarting training...\n"+"="*50)
			self.model.train()
			
			for index_iter in range(start_iter, int(max_iterations)):
				
				# Training step
				start_time = time.time()
				loss, acc = self.task.train_step(self.batch_size, iteration=index_iter)
				self.optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(parameters_to_optimize, max_gradient_norm)
				self.optimizer.step()
				self.lr_scheduler.step()
				end_time = time.time()
				time_per_step[0] += end_time - start_time
				time_per_step[1] += 1
				train_losses.append(loss.item())
				train_accs.append(acc.item())

				# Debug loss printing
				if (index_iter + 1) % loss_freq == 0:
					loss_avg, acc_avg = sum(train_losses)/len(train_losses), sum(train_accs)/len(train_accs)
					print("Training iteration %i|%i. Loss: %6.5f" % (index_iter+1, max_iterations, loss_avg))
					writer.add_scalar("train/loss", loss_avg, index_iter + 1)
					writer.add_scalar("train/acc", acc_avg, index_iter + 1)
					writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], index_iter+1)
					writer.add_scalar("train/training_time", time_per_step[0] / max(1e-5, time_per_step[1]), index_iter+1)
					self.task.add_summary(writer, index_iter + 1)
					time_per_step[:] = 0
					train_losses, train_accs = [], []

				# Evaluation
				if (index_iter + 1) % eval_freq == 0:
					
					self.model.eval()
					eval_BLEU, detailed_scores = self.task.eval(batch_size=self.batch_size)
					self.model.train()

					write_dict_to_tensorboard(writer, detailed_scores, base_name="eval", iteration=index_iter+1)
					if (index_iter + 1) % (eval_freq * 5) == 0:
						export_weight_parameters(index_iter+1)

					if best_save_dict["metric"] < 0 or eval_BLEU > best_save_dict["metric"]: # TODO: Test whether this is new best score or not
						best_save_iter = self.get_checkpoint_filename(index_iter+1)
						if not os.path.isfile(best_save_iter):
							print("Saving model at iteration " + str(index_iter+1))
							save_train_model(index_iter+1)
							if best_save_dict["file"] is not None and os.path.isfile(best_save_dict["file"]):
								os.remove(best_save_dict["file"])
							if last_save is not None and os.path.isfile(last_save):
								os.remove(last_save)
							best_save_dict["file"] = best_save_iter
							last_save = best_save_iter
						best_save_dict["metric"] = eval_BLEU
						best_save_dict["detailed_metrics"] = detailed_scores
						self.task.export_best_results(self.checkpoint_path, index_iter + 1)
					evaluation_dict[index_iter + 1] = best_save_dict["metric"]

				# Saving
				if (index_iter + 1) % save_freq == 0 and not os.path.isfile(self.get_checkpoint_filename(index_iter+1)):
					save_train_model(index_iter + 1)
					if last_save is not None and os.path.isfile(last_save) and last_save != best_save_iter:
						os.remove(last_save)
					last_save = self.get_checkpoint_filename(index_iter+1)

			eval_BLEU, detailed_scores = self.task.eval(batch_size=self.batch_size)
			print("Before reloading, the model achieved a score of %f" % eval_BLEU)
			if not no_model_checkpoints and best_save_iter is not None:
				load_model(best_save_iter, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
				eval_BLEU, detailed_scores = self.task.eval(batch_size=self.batch_size)
				print("Best model achieved %s" % str(eval_BLEU))
				if eval_BLEU != best_save_dict["metric"]:
					print("[!] WARNING: new evaluation differs from saved one (%s vs %s)!" % (str(eval_BLEU), str(best_save_dict["metric"])))
				self.task.finalize_summary(writer, max_iterations, self.checkpoint_path)
			else:
				print("Skipping finalizing the summary because %s..." % ("no model checkpoints were saved" if no_model_checkpoints else "best_save_iter was None"))


		except KeyboardInterrupt:
			print("User keyboard interrupt detected. Saving model at step %i..." % (index_iter))
			save_train_model(index_iter + 1)
			if last_save is not None and os.path.isfile(last_save) and not any([val == last_save for _, val in best_save_dict.items()]):
				os.remove(last_save)

		with open(os.path.join(self.checkpoint_path, "results.txt"), "w") as f:
			for eval_iter, eval_dict in evaluation_dict.items():
				f.write("Iteration %i: " % (eval_iter))
				f.write("BLEU: %s" % str(best_save_dict["metric"]))
				f.write("\n")

		writer.close()


	def get_checkpoint_filename(self, iteration):
		checkpoint_file = os.path.join(self.checkpoint_path, 'checkpoint_' + str(iteration).zfill(7) + ".tar")
		return checkpoint_file


	def save_model(self, iteration, add_param_dict, save_embeddings=False, save_optimizer=True):
		checkpoint_file = self.get_checkpoint_filename(iteration)
		model_dict = self.model.state_dict()
		# model_dict = {k:v for k,v in model_dict.items() if v.requires_grad}
		# print(model_dict)
		# sys.exit(1)
		
		checkpoint_dict = {
			'model_state_dict': model_dict
		}
		if save_optimizer:
			checkpoint_dict['optimizer_state_dict'] = self.optimizer.state_dict()
			checkpoint_dict['scheduler_state_dict'] = self.lr_scheduler.state_dict()
		checkpoint_dict.update(add_param_dict)
		torch.save(checkpoint_dict, checkpoint_file)


	def load_recent_model(self):
		checkpoint_dict = load_model(self.checkpoint_path, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler)
		return checkpoint_dict



def get_default_train_arguments():
	parser = argparse.ArgumentParser()
	# Training extras
	parser.add_argument("--max_iterations", help="Maximum number of epochs to train. Default: dynamic with learning rate threshold", type=int, default=1e6)
	parser.add_argument("--batch_size", help="Batch size used during training", type=int, default=64)
	parser.add_argument("--eval_freq", help="In which frequency the model should be evaluated (in number of iterations). Default: 2000", type=int, default=2000)
	parser.add_argument("--save_freq", help="In which frequency the model should be saved (in number of iterations). Default: 10,000", type=int, default=1e4)
	# Loading experiment
	parser.add_argument("--restart", help="Does not load old checkpoints, and deletes those if checkpoint path is specified (including tensorboard file etc.)", action="store_true")
	parser.add_argument("--checkpoint_path", help="Folder(name) where checkpoints should be saved", type=str, default=None)
	parser.add_argument("--load_config", help="Tries to find parameter file in checkpoint path, and loads all given parameters from there", action="store_true")
	parser.add_argument("--no_model_checkpoints", help="If selected, no model checkpoints will be saved", action="store_true")
	# Output control
	parser.add_argument("--seed", help="Seed to make experiments reproducable", type=int, default=42)
	parser.add_argument("--cluster", help="Enable option if code is executed on cluster. Reduces output size", action="store_true")
	parser.add_argument("-d", "--debug", help="Whether debug output should be activated or not", action="store_true")
	parser.add_argument("--clean_up", help="Whether to remove all files after finishing or not", action="store_true")
	# Optimizer parameters
	parser.add_argument("--learning_rate", help="Learning rate of the optimizer", type=float, default=1e-4)
	parser.add_argument("--lr_decay", help="Decay of learning rate of the optimizer. Always applied if eval accuracy droped compared to mean of last two epochs", type=float, default=0.2)
	parser.add_argument("--lr_decay_step", help="Number of steps after which learning rate should be decreased", type=float, default=1e6)
	parser.add_argument("--weight_decay", help="Weight decay of the optimizer", type=float, default=0.0)
	parser.add_argument("--optimizer", help="Which optimizer to use. 0: SGD, 1: Adam", type=int, default=1)
	parser.add_argument("--momentum", help="Apply momentum to SGD optimizer", type=float, default=0.0)

	return parser

def start_training(args, parse_args_to_params_fun, TrainClass):

	if args.cluster:
		set_debug_level(2)
		loss_freq = 250
	else:
		set_debug_level(0)
		loss_freq = 2

	if args.load_config:
		if args.checkpoint_path is None:
			print("[!] ERROR: Please specify the checkpoint path to load the config from.")
			sys.exit(1)
		args = load_args(args.checkpoint_path)
		args.clean_up = False

	# Setup training
	model_params, optimizer_params = parse_args_to_params_fun(args)
	trainModule = TrainClass(model_params=model_params,
							 optimizer_params=optimizer_params, 
							 batch_size=args.batch_size,
							 checkpoint_path=args.checkpoint_path, 
							 debug=args.debug
							 )

	def clean_up_dir():
		print("Cleaning up directory " + str(trainModule.checkpoint_path) + "...")
		for file_in_dir in sorted(glob(os.path.join(trainModule.checkpoint_path, "*"))):
			print("Removing file " + file_in_dir)
			try:
				if os.path.isfile(file_in_dir):
					os.remove(file_in_dir)
				elif os.path.isdir(file_in_dir): 
					shutil.rmtree(file_in_dir)
			except Exception as e:
				print(e)

	if args.restart and args.checkpoint_path is not None and os.path.isdir(args.checkpoint_path):
		clean_up_dir()

	args_filename = os.path.join(trainModule.checkpoint_path, PARAM_CONFIG_FILE)
	with open(args_filename, "wb") as f:
		pickle.dump(args, f)

	trainModule.train_model(args.max_iterations, loss_freq=loss_freq, eval_freq=args.eval_freq, save_freq=args.save_freq, no_model_checkpoints=args.no_model_checkpoints)

	if args.clean_up:
		clean_up_dir()
		os.rmdir(trainModule.checkpoint_path)

