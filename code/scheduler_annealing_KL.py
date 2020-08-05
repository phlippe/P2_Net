import sys
import os
import numpy as np 
import matplotlib.pyplot as plt

##############################
## HELPER FUNCTIONS/CLASSES ##
##############################

class KLSchedulerTypes:

	CONSTANT = 0
	MONOTONIC = 1
	CYCLIC = 2

	LINEAR = 0
	SIGMOID = 1
	EXPONENTIAL = 2

	def scheduler_string():
		return "Constant: %i, Monotonic: %i, Cyclic: %i" % (KLSchedulerTypes.CONSTANT, KLSchedulerTypes.MONOTONIC, KLSchedulerTypes.CYCLIC)

	def anneal_func_string():
		return "Linear: %i, Sigmoid: %i, Exponential: %i" % (KLSchedulerTypes.LINEAR, KLSchedulerTypes.SIGMOID, KLSchedulerTypes.EXPONENTIAL)


def create_KLScheduler(scheduler_type, loss_scaling=1, num_iters=10000, annealing_func_type=KLSchedulerTypes.LINEAR):
	annealing_func = create_KLAnnealingFunc(annealing_func_type)
	if scheduler_type == KLSchedulerTypes.CONSTANT:
		return KLSchedulerTemplate(annealing_func=annealing_func, loss_scaling=loss_scaling)
	elif scheduler_type == KLSchedulerTypes.MONOTONIC:
		return KLMonotonicScheduler(annealing_func=annealing_func, loss_scaling=loss_scaling, num_iters=num_iters)
	elif scheduler_type == KLSchedulerTypes.CYCLIC:
		return KLCyclicScheduler(annealing_func=annealing_func, loss_scaling=loss_scaling, num_iters=num_iters)
	else:
		print("[!] ERROR: Unknown scheduler type %s." % (str(scheduler_type)))
		sys.exit(1)


def create_KLAnnealingFunc(annealing_func_type):
	if annealing_func_type == KLSchedulerTypes.LINEAR:
		return KLAnnealingFunctionLinear()
	elif annealing_func_type == KLSchedulerTypes.SIGMOID:
		return KLAnnealingFunctionSigmoid()
	elif annealing_func_type == KLSchedulerTypes.EXPONENTIAL:
		return KLAnnealingFunctionExponential()
	else:
		print("[!] ERROR: Unknown annealing function type %s." % (str(annealing_func_type)))
		sys.exit(1)

##################
## KL SCHEDULER ##
##################


class KLSchedulerTemplate:

	def __init__(self, annealing_func=None, loss_scaling=1):
		self.loss_scaling = loss_scaling
		self.annealing_func = annealing_func

	def get(self, iteration):
		return self.loss_scaling


class KLMonotonicScheduler(KLSchedulerTemplate):

	def __init__(self, annealing_func, loss_scaling=1, num_iters=10000):
		super(KLMonotonicScheduler, self).__init__(annealing_func, loss_scaling)
		self.num_iters = num_iters

	def get(self, iteration):
		if iteration < self.num_iters:
			return self.loss_scaling * self.annealing_func.get(iteration / self.num_iters)
		else:
			return self.loss_scaling


class KLCyclicScheduler(KLSchedulerTemplate):

	def __init__(self, annealing_func, loss_scaling=1, num_iters=10000):
		super(KLCyclicScheduler, self).__init__(annealing_func, loss_scaling)
		self.num_iters = num_iters

	def get(self, iteration):
		iteration = iteration % self.num_iters
		if iteration < self.num_iters / 2:
			return self.loss_scaling * self.annealing_func.get(iteration / (self.num_iters / 2.0))
		else:
			return self.loss_scaling


############################
## KL ANNEALING FUNCTIONS ##
############################

class KLAnnealingFunctionTemplate:

	def __init__(self):
		# assert self.get(0) == 0, "[!] Implementation ERROR: KL Annealing Functions have to start at 0."
		# assert self.get(1) == 1, "[!] Implementation ERROR: KL Annealing Functions have to end at 1."
		self.min, self.max = 0, 1

	def prep(self):
		self.min = self.annealing(0)
		self.max = self.annealing(1)
		assert self.max > self.min, "[!] Implementation ERROR: KL Annealing Functions must be constantly increasing!"

	def get(self, v):
		assert (v >= 0 and v <= 1), "[!] ERROR: KL Annealing Functions only take input of the range 0 to 1!"
		return (self.annealing(v) - self.min) / max(1e-5, self.max - self.min)

	def annealing(self, v):
		raise NotImplementedError


class KLAnnealingFunctionLinear(KLAnnealingFunctionTemplate):

	def __init__(self):
		super(KLAnnealingFunctionLinear, self).__init__()
		self.prep()

	def annealing(self, v):
		return v


class KLAnnealingFunctionSigmoid(KLAnnealingFunctionTemplate):

	def __init__(self, comp_fac=10):
		super(KLAnnealingFunctionSigmoid, self).__init__()
		self.comp_fac = comp_fac
		self.prep()

	def annealing(self, v):
		return 1.0 / (1.0 + np.exp(- self.comp_fac * (v - 0.5)))


class KLAnnealingFunctionExponential(KLAnnealingFunctionTemplate):

	def __init__(self, exp_fac=0.05):
		super(KLAnnealingFunctionExponential, self).__init__()
		self.exp_fac = exp_fac
		self.prep()

	def annealing(self, v):
		return self.exp_fac ** (1-v)


if __name__ == '__main__':
	funcs = [(name, create_KLAnnealingFunc(func_type)) for name, func_type in zip(["Linear", "Sigmoid", "Exponential"],
																				  [KLSchedulerTypes.LINEAR, KLSchedulerTypes.SIGMOID, KLSchedulerTypes.EXPONENTIAL])]
	x_vals = np.arange(start=0, stop=1, step=0.001).tolist()
	plt.grid(b=True, which='both', axis='both', linestyle='--', linewidth=2, alpha=0.5)
	for func_name, func_obj in funcs:
		plt.plot(x_vals, [func_obj.get(x) for x in x_vals], label=func_name)	
	plt.xlim([-0.1, 1.1])
	plt.ylim([-0.1, 1.1])
	plt.title("KL Annealing Functions")
	plt.legend()
	plt.show()
