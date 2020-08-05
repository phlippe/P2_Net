import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def str_to_array(s):
	s = s.split("[",1)[-1].rsplit("]",1)[0]
	s = s.replace("\n","")
	s = [c for c in s.split(" ") if len(c)>0]
	arr = [float(num) for num in s]
	arr = np.array(arr)
	if abs(np.sum(arr)-1) > 0.01:
		print("Sum is not 1: ", arr)
	return arr

def text_to_tokens(s):
	tokens = [c for c in s.split(" ") if len(c)>0]
	comb_tokens = []
	is_slot = False
	for t in tokens:
		if t.startswith("<") and not t.endswith(">"):
			comb_tokens.append(t)
			is_slot = True
		elif t.endswith(">") and not t.startswith("<"):
			comb_tokens[-1] += " "+t 
			is_slot = False
		elif is_slot:
			comb_tokens[-1] += " "+t
		else:
			comb_tokens.append(t)
	return comb_tokens


def load_attn_mask(filename):
	with open(filename, "r") as f:
		lines = [line.strip() for line in f.readlines()]
	attn_masks = []
	for i in range(len(lines)):
		line = lines[i]
		if line.startswith("Input:"):
			attn_masks.append({"sent": text_to_tokens(line.split("]",1)[-1])})
		elif line.startswith("(0):"):
			array_str = line.split("Semantic attention: ")[-1]
			for j in range(i+1, i+10):
				if not lines[j].startswith("("):
					array_str += " " + lines[j]
				else:
					break
			attn_masks[-1]["semantic"] = str_to_array(array_str)
			if len(attn_masks[-1]["sent"]) > attn_masks[-1]["semantic"].shape[0]:
				print("Length sentence", len(attn_masks[-1]["sent"]), "Length semantic", attn_masks[-1]["semantic"].shape[0])
				print(attn_masks)
				sys.exit(1)
		elif line.startswith("(1):"):
			array_str = line.split("Style attention: ")[-1]
			for j in range(i+1, i+10):
				if not lines[j].startswith("("):
					array_str += " " + lines[j]
				else:
					break
			attn_masks[-1]["style"] = str_to_array(array_str)
			if len(attn_masks[-1]["sent"]) > attn_masks[-1]["style"].shape[0]:
				print("Length sentence", len(attn_masks[-1]["sent"]), "Length style", attn_masks[-1]["style"].shape[0])
				print(attn_masks[-1])
				sys.exit(1)
		elif line.startswith("(2):"):
			array_str = line.split("Prototype distribution: ")[-1]
			attn_masks[-1]["proto"] = str_to_array(array_str)

	duplicates = []
	for i in range(len(attn_masks)):
		if i == 0:
			continue
		if attn_masks[i]["sent"] == attn_masks[i-1]["sent"]:
			duplicates.append(i)

	attn_masks = [mask for i, mask in enumerate(attn_masks) if i not in duplicates]
	return attn_masks


def visualize_attn(sent, mask, filename):
	fig = plt.figure()
	ax = fig.add_subplot(111)
			
	sent_attention_map = mask[None,:len(sent)]
	print(sent_attention_map.shape)
	print(sent_attention_map)

	cax = ax.matshow(sent_attention_map, cmap=plt.cm.gray, vmin=0.0)
	ax.set_xticks(np.arange(len(sent)))
	ax.set_xticklabels(sent, rotation=90)
	ax.set_yticklabels([""])
	ax.set_yticks([0])
	ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

	plt.tight_layout()
	plt.savefig(filename)
	plt.close()


def visualize_combined_attn(sent, mask_semantic, mask_style, filename):
	fig = plt.figure(figsize=(8,5))
	ax = fig.add_subplot(111)
	
	mask = np.stack([mask_semantic, mask_style], axis=0)
	sent_attention_map = mask[:,:len(sent)]
	sent_attention_map = sent_attention_map / np.max(sent_attention_map, axis=1, keepdims=True)
	colors = np.zeros((2, 1, 3))
	colors[0,:,:] = np.array([1.0, 0.9, 0.9])
	colors[1,:,:] = np.array([0.9, 1.0, 1.0])

	# cax = ax.matshow(sent_attention_map, cmap=plt.cm.gray, vmin=0.0, vmax=1.0) # 
	cax = ax.imshow(sent_attention_map[:,:,None] * colors, vmin=0.0, vmax=1.0, aspect="equal", origin="lower")
	ax.set_xticks(np.arange(len(sent)))
	ax.set_xticklabels(sent, rotation=90)
	ax.set_yticklabels(["Semantic", "Style"])
	ax.set_yticks([0,1])
	ax.set_ylim(-0.5,1.5)
	ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

	plt.tight_layout()
	plt.savefig(filename)
	plt.close()


def print_proto_dists(attn_masks):
	for mask in attn_masks:
		print("Sentence", " ".join(mask["sent"]))
		print("Prototype dist", mask["proto"])
		print("--------------")



if __name__ == "__main__":
	attn_masks = load_attn_mask("checkpoints/best_model_238441/contextawaredialogueparaphrase_gt_attention_maps.txt")

	for i, mask in enumerate(attn_masks[:100:-1]):
		print("Mask %i" % i)
		# visualize_attn(sent=mask["sent"], mask=mask["style"], filename="checkpoints/best_model_238441/attn_style_"+str(i).zfill(3)+".png")
		# visualize_attn(sent=mask["sent"], mask=mask["semantic"], filename="checkpoints/best_model_238441/attn_semantic_"+str(i).zfill(3)+".png")
		# visualize_attn(sent=mask["sent"], mask=mask["style"], filename="checkpoints/best_model_238441/attn_style_"+str(i).zfill(3)+".pdf")
		# visualize_attn(sent=mask["sent"], mask=mask["semantic"], filename="checkpoints/best_model_238441/attn_semantic_"+str(i).zfill(3)+".pdf")
		visualize_combined_attn(sent=mask["sent"], mask_semantic=mask["semantic"], mask_style=mask["style"], filename="checkpoints/best_model_238441/attn_combined_"+str(i).zfill(3)+".pdf")
	print_proto_dists(attn_masks)