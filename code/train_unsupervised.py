import argparse

from unsupervised_task import DialogueModelingTask, LanguageModelingTask, ParaphraseTask, PretrainingTask, ContextAwareDialogueTask, ContextAwareLanguageModelingTask, ContextAwarePretrainingTask
from unsupervised_models.model import ModelUnsupervisedParaphrasingTemplate, ModelUnsupervisedContextParaphrasingTemplate
from mutils import unsupervised_args_to_params
from train import TrainTemplate, get_default_train_arguments, start_training
from model_utils import ModelTypes
from scheduler_annealing_KL import KLSchedulerTypes

class UnsupervisedParaphraseTrain(TrainTemplate):


	OPTIMIZER_SGD = 0
	OPTIMIZER_ADAM = 1


	def __init__(self, model_params, optimizer_params, batch_size, checkpoint_path, debug=False):
		super(UnsupervisedParaphraseTrain, self).__init__(model_params, optimizer_params, batch_size, checkpoint_path, debug=debug)
		

	def _create_model(self, model_params, wordvec_tensor):
		# model = ModelUnsupervisedParaphrasingTemplate(model_params, wordvec_tensor) # FIXME
		model = ModelUnsupervisedContextParaphrasingTemplate(model_params, wordvec_tensor)
		return model


	def _create_task(self, model_params, debug=False):
		# (Possible other tasks)
		# task = PretrainingTask(self.model, model_params, debug=debug) # FIXME
		# task = LanguageModelingTask(self.model, model_params, debug=debug)
		# task = ParaphraseTask(self.model, model_params, debug=debug)
		# task = ContextAwareDialogueTask(self.model, model_params, debug=debug)
		task = ContextAwarePretrainingTask(self.model, model_params, debug=debug)
		return task


if __name__ == '__main__':
	parser = get_default_train_arguments()
	# General model parameters
	parser.add_argument("--embed_dropout", help="Dropout applied on the input embeddings", type=float, default=0.0)
	parser.add_argument("--finetune_embeds", help="Whether to finetune the embeddings or not", action="store_true")
	parser.add_argument("--switch_rate", help="The ratio with which semantics are switched. Default: 80%", type=float, default=0.8)
	parser.add_argument("--teacher_forcing_ratio", help="The ratio with which teacher forcing is applied. Specify it as number between 0.0 and 1.0", type=float, default=0.95)
	parser.add_argument("--teacher_forcing_annealing", help="Whether to anneal the teacher forcing ratio (going to 0) over time.", type=float, default=-1)
	parser.add_argument("--semantic_size", help="Size of semantic embedding", type=int, default=256)
	parser.add_argument("--style_size", help="Size of style embedding", type=int, default=128)
	parser.add_argument("--response_style_size", help="The size of the style embedding for the response can be adjusted with this parameter. Only valid if the context style is used as well.", type=int, default=-1)
	parser.add_argument("--VAE_loss_scaling", help="Factor with which the VAE regularization loss should be scaled.", type=float, default=0.0)
	parser.add_argument("--VAE_annealing_iters", help="Number of iterations for the annealing process used in VAE scheduler.", type=int, default=10000)
	parser.add_argument("--VAE_annealing_func", help="Annealing function to use with the scheduler. Implemented functions are: %s" % (KLSchedulerTypes.anneal_func_string()), type=int, default=0)
	parser.add_argument("--VAE_scheduler", help="Scheduler to use for the VAE annealing. Implemented schedulers are: %s" % (KLSchedulerTypes.scheduler_string()), type=int, default=1)
	parser.add_argument("--cosine_loss_scaling", help="Factor with which the cosine regularization loss should be scaled.", type=float, default=0.0)
	parser.add_argument("--cosine_counter_loss", help="Whether the cosine loss should include a term to push other styles/semantics away from each other or not.", action="store_true")
	parser.add_argument("--style_loss_scaling", help="Factor with which the style regularization loss should be scaled.", type=float, default=0.0)
	parser.add_argument("--style_loss_module", help="Which loss module to use.", type=int, default=0)
	parser.add_argument("--style_loss_stop_grads", help="Whether to stop gradients from the ground truth side in the style loss or not.", action="store_true")
	parser.add_argument("--style_loss_annealing_iters", help="Number of iterations for the annealing process used for style loss scaling.", type=int, default=-1)
	parser.add_argument("--semantics_dropout", help="Dropout applied on the semantic embedding before decoding.", type=float, default=0.2)
	parser.add_argument("--semantic_full_dropout", help="Rate with which the whole semantic vector is dropped.", type=float, default=0.1)
	parser.add_argument("--num_context_turns", help="Number of turns that are taken into account for generating the context style.", type=int, default=2)
	parser.add_argument("--pure_style_loss", help="Whether to use only style loss or not. WARNING: This option should only be used for debugging purposes.", action="store_true")
	parser.add_argument("--pretraining_iterations", help="Number of iterations used for pretraining the dialogue model.", type=int, default=15000)
	parser.add_argument("--pretraining_second_task", help="Influence of the second task.", type=float, default=0.15)
	parser.add_argument("--only_paraphrasing", help="If selected, only the Quora Paraphrasing dataset is used for training.", action="store_true")
	parser.add_argument("--use_semantic_specific_attn", help="Whether to use the semantic encoding as context vector for the style attention in the context style extraction.", action="store_true")
	parser.add_argument("--use_prototype_styles", help="Whether to use prototypes for paraphrase style or not.", action="store_true")
	parser.add_argument("--num_prototypes", help="Number of prototype vectors to use if \"--use_prototype_styles\" option is activated.", type=int, default=5)
	parser.add_argument("--use_semantic_for_context_proto", help="Whether to use the semantic vector as additional input for generating prototype distribution or not.", action="store_true")
	parser.add_argument("--no_prototypes_for_context", help="If selected, no prototypes for context will be used.", action="store_true")
	parser.add_argument("--style_exponential_dropout", help="Dropout applied on the GT style, exponentially decaying the number of steps when GT is valid again.", type=float, default=0.4)
	parser.add_argument("--style_full_dropout", help="Dropout applied on the GT style valid over all time steps", type=float, default=0.0)
	# Slot encoder models 
	parser.add_argument("--positional_embedding_factor", help="Factor with which positional encodings are added to slot embeddings.", type=float, default=0.25)
	parser.add_argument("--no_slot_value_embeddings", help="Removes the slot value embedding.", action="store_true")
	parser.add_argument("--slots_CBOW", help="In the slot embedding values, whether to use a Continuous BoW approach or a simple BoW", action="store_true")
	# Encoder models 
	parser.add_argument("--encoder_model", help="The kind of model that should be used as paraphrase model. Implemented models are: %s" % (ModelTypes.encoder_string()), type=int, default=2)
	parser.add_argument("--encoder_hidden_size", help="Hidden size for the paraphrase module", type=int, default=256)
	parser.add_argument("--encoder_num_layers", help="Number of layers (e.g. LSTMs) for the paraphrase module", type=int, default=1)
	parser.add_argument("--encoder_dropout", help="Dropout applied on the input encodings to the paraphrase module", type=float, default=0.0)
	parser.add_argument("--encoder_separate_attentions", help="Dropout applied on the input encodings to the paraphrase module", action="store_true")
	# Paraphrase models
	parser.add_argument("--decoder_model", help="The kind of model that should be used as paraphrase model. Implemented models are: %s" % (ModelTypes.decoder_string()), type=int, default=0)
	parser.add_argument("--decoder_hidden_size", help="Hidden size for the paraphrase module", type=int, default=512)
	parser.add_argument("--decoder_num_layers", help="Number of layers (e.g. LSTMs) for the paraphrase module", type=int, default=1)
	parser.add_argument("--decoder_input_dropout", help="Dropout applied on the input encodings to the paraphrase module", type=float, default=0.2)
	parser.add_argument("--decoder_lstm_dropout", help="Dropout applied on the outputs within the LSTM stacks for paraphrase module", type=float, default=0.2)
	parser.add_argument("--decoder_output_dropout", help="Dropout applied on the encodings going to the classifier head", type=float, default=0.2)
	parser.add_argument("--decoder_concat_features", help="Whether to concatenate semantic, style and lstm output features in the output layer or to apply a reduction layer.", action="store_true")
	parser.add_argument("--decoder_lstm_additional_input", help="Whether to add a combination of semantics, style and last slot embeddings to the input of the decoder LSTM or not.", action="store_true")

	args = parser.parse_args()
	print(args)

	start_training(args, unsupervised_args_to_params, UnsupervisedParaphraseTrain)
	