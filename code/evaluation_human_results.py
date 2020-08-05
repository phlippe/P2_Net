import os
import csv
from statistics import mean, stdev


METRICS = ["Context", "Grammar", "Natural", "Semantic"]


def read_csv(filename):

	with open(filename, "r") as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
		line_count = 0
		lines = []
		for row in csv_reader:
			if line_count == 0:
				print(f'Column names are {", ".join(row)}')
				# print("Number of columns", len(row))
				# for i, r in enumerate(row):
				# 	print("%i: %s" % (i, r))
				print(row[35:65])
			else:
				lines.append((row[35:41], [float(e) for e in row[41:65]]))
			line_count += 1
		print(f'Processed {line_count} lines.')
	return lines

def process_results(lines):
	result_dict = {}
	for model_names, results in lines:
		unique_models = []
		for i, name in enumerate(model_names):
			if name not in unique_models:
				unique_models.append(name)
			if name not in result_dict:
				result_dict[name] = {m: [] for m in METRICS}
			result_dict[name]["Context"].append(results[i])
			result_dict[name]["Grammar"].append(results[i+6])
			result_dict[name]["Natural"].append(results[i+12])
			result_dict[name]["Semantic"].append(results[i+18])

	return result_dict

def print_statistics(result_dict):
	for name, results in result_dict.items():
		print("Model %s" % name)
		for metric, res_list in results.items():
			res_mean = mean(res_list)
			res_stdev = stdev(res_list)
			print("-> %s: %3.2f (+-%3.2f)" % (metric, res_mean, res_stdev))

def determine_win_losses(metric_dict1, metric_dict2):
	scores = {m: {"win_1": 0, "win_2": 0, "ties": 0} for m in METRICS}
	for metric in METRICS:
		for i in range(0, len(metric_dict1[metric]), 6):
			mean_score_1 = mean(metric_dict1[metric][i:i+6])
			mean_score_2 = mean(metric_dict2[metric][i:i+6])
			if mean_score_1 > mean_score_2:
				scores[metric]["win_1"] += 1
			elif mean_score_1 < mean_score_2:
				scores[metric]["win_2"] += 1
			elif mean_score_1 == mean_score_2:
				scores[metric]["ties"] += 1
	for metric in METRICS:
		print("%s: Wins_1=%i , Wins_2=%i , Ties=%i" % (metric, scores[metric]["win_1"], scores[metric]["win_2"], scores[metric]["ties"]))
	return scores


if __name__ == '__main__':
	lines = read_csv("MTurk_results.csv")
	result_dict = process_results(lines)
	print_statistics(result_dict)
	print("Proto vs Beam")
	determine_win_losses(result_dict["proto"], result_dict["beam"])
	print("Proto vs Human responses")
	determine_win_losses(result_dict["proto"], result_dict["human_resp"])
	print("Beam vs Human responses")
	determine_win_losses(result_dict["beam"], result_dict["human_resp"])