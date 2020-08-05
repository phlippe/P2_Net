import torch
import torch.nn as nn
import numpy as np 
from collections import Counter
import math
import rouge

def get_BLEU_batch_stats(hypotheses, references, max_N_gram=5):
	stats = np.zeros(shape=(max_N_gram, 2), dtype=np.float32)
	for hypothesis, reference in zip(hypotheses, references):

		if isinstance(reference, str) or isinstance(hypothesis, str):
			print("[#] WARNING: Reference or hypothesis was a string, not a list of words")
		elif isinstance(reference, list) and len(reference) > 0 and isinstance(reference[0], list):
			print("[#] WARNING: Reference is a list of tokens!")
		elif isinstance(hypothesis, list) and len(hypothesis) > 0 and isinstance(hypothesis[0], list):
			print("[#] WARNING: Hypothesis is a list of tokens!")

		stats[0, 0] += len(reference)
		stats[0, 1] += len(hypothesis)
		for n in range(1, max_N_gram):
			s_ngrams = Counter(
				[tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
			)
			r_ngrams = Counter(
				[tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
			)
			stats[n, 0] += max([sum((s_ngrams & r_ngrams).values()), 0])
			stats[n, 1] += max([len(hypothesis) + 1 - n, 0])
	return stats

def get_BLEU_score(hypotheses, references, max_N_gram=5):
	stats = get_BLEU_batch_stats(hypotheses, references, max_N_gram = max_N_gram)
	prec_per_ngram = stats[1:,0] / np.maximum(stats[1:,1], 1e-5)
	if (stats == 0).sum() > 0:
		bleu_score = 0
	else:
		log_bleu_prec = np.log(prec_per_ngram).sum() / (stats.shape[0] - 1)
		bleu_score = np.exp(min(0, 1 - stats[0,0] / stats[0,1]) + log_bleu_prec)
	return bleu_score, prec_per_ngram

ROUGE_EVALUATOR = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=True,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)
def get_ROUGE_score(hypotheses, references):
	if len(hypotheses) > 0 and not isinstance(hypotheses[0], str):
		hypotheses = [" ".join(h) for h in hypotheses]
		references = [" ".join(r) for r in references]
	return ROUGE_EVALUATOR.get_scores(hypotheses, references)

def euclidean_distance(vecs_1, vecs_2):
	dist = torch.sqrt(((vecs_1 - vecs_2)**2).sum(dim=-1))
	return dist

def get_diversity_measure(sentences, n_gram=1):
	if isinstance(sentences[0][0], str):
		div = _get_diversity_measure(sentences, n_gram=n_gram)
		return (div[0]/1.0*div[2], div[0])
	elif isinstance(sentences[0][0], list):
		diversity_per_set = [_get_diversity_measure(sentence_set, n_gram=n_gram) for sentence_set in sentences]
		diversity_per_sentence = [[_get_diversity_measure([s], n_gram=n_gram) for s in sentence_set] for sentence_set in sentences]
		set_size = [len(sentence_set) for sentence_set in sentences]
		avg_diversity_per_set = [tuple([sum([d[i] for d in dset])/(1.0*len(dset)) for i in range(2)]) for dset in diversity_per_sentence]
		diversity = tuple([sum([(set_div[i] - avg_div[i])/max(size-1,1) for avg_div, set_div, size in zip(avg_diversity_per_set, diversity_per_set, set_size)])/len(diversity_per_set) for i in range(2)])
		
		distinct_ngram = sum([div_per_set[0]/(1.0*div_per_set[-1]+1e-5) for div_per_set in diversity_per_set])/len(diversity_per_set)
		diversity = (distinct_ngram, diversity[0])

		return diversity

def _get_diversity_measure(sentences, n_gram=1):
	sent_counters = Counter()
	assert isinstance(sentences[0], list) and isinstance(sentences[0][0], str), "[!] ERROR: Sentences are not provided in the right form: %s" % str(sentences)
	for s in sentences:
		ngram_tuples = [tuple(s[i:i+n_gram]) for i in range(len(s) + 1 - n_gram)]
		sent_counters.update(ngram_tuples)

	token_dict = dict(sent_counters)
	num_counts = sum(token_dict.values())
	num_different_ngrams = len(token_dict.keys())
	prob_dist = [val * 1.0 / num_counts for val in token_dict.values()]
	entropy_val = -sum([p * math.log(max(1e-10, p)) for p in prob_dist])

	return num_different_ngrams, entropy_val, num_counts



if __name__ == '__main__':
	sentences = [["A B C D", "D C B A", "A C B D", "B C A D"]]
	sentences = [[sent.split(" ") for sent in sent_set] for sent_set in sentences]
	out = get_diversity_measure(sentences, n_gram=3)
	print(out)

	