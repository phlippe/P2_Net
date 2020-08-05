import json
from nltk import word_tokenize

with open("data.json", "r") as f:
	data = json.load(f)

print("Num conversations: " + str(len(data.keys())))

conversations = dict()

for conv_key, conv_data in data.items():
	conversations[conv_key] = list()
	conversations[conv_key] = [(i%2, c["text"]) for i, c in enumerate(conv_data["log"])]

with open("conversations.json", "w") as f:
	json.dump(conversations, f, indent=4) 


lines = []
for conv_key, conv_list in conversations.items():
	for i in range(len(conv_list)):
		if i % 2 == 0:
			continue
		if len(conv_list[i-1][1]) == 0 or len(conv_list[i][1]) == 0:
			continue
		sentences = [" ".join(word_tokenize(s.replace("\n"," "))) for s in [conv_list[i-1][1], conv_list[i][1]]]
		l = "\t".join(sentences)
		lines.append(l)

with open("conversations.txt", "w") as f:
	f.write("\n".join(lines))

with open("val.txt", "w") as f:
	f.write("\n".join(lines[:2000]))
with open("train.txt", "w") as f:
	f.write("\n".join(lines[2000:]))