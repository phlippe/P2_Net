import json
from nltk import word_tokenize, sent_tokenize
import sys
from copy import copy
import re
from spellchecker import SpellChecker
import random
from random import shuffle
import argparse
from statistics import median, mean, stdev
random.seed(42)

spell = SpellChecker()
missspellings = [("addres sis", "adress is"),
				 ("restuarant", "restaurant"),
				 ("anythign", "anything"),
				 ("musuem", "museum"),
				 ("adress", "address"),
				 ("thatres", "theaters"),
				 ("parsk", "parks"),
				 ("a.m.", "am"),
				 ("A.M.", "am")]

SLOT_REGEX = re.compile(r"<[a-zA-Z_]*=[a-zA-Z_0-9.:,\\\"'&\-; ]*>")
def find_slots(temp):
	return SLOT_REGEX.findall(temp)

def create_regex_for_slot_val(val):
	return re.compile(r"<[a-zA-Z_]*=[a-zA-Z_0-9.:,\\\"'&\-; ]*"+val+r"[a-zA-Z_0-9.:,\\\"'&\-; ]*>")

def is_question(s):
	return any([a in s for a in ["Booking-Inform", "Request", "general-reqmore", "OfferBook"]])

def check_unknown_slots(temp, turn_vals):
	slots_found = find_slots(temp)
	unknown_slots = False
	for s in slots_found:
		slot_ind = s.split("=")[0][1:]
		slot_val = s.split("=")[1][:-1].replace("\"","")
		if not any([False if not isinstance(l, list) else (slot_ind.lower() == l[0].lower() and slot_val.lower() == l[1].lower()) for l in turn_vals]):
			unknown_slots = True
			break
	if not unknown_slots:
		for t in turn_vals:
			if isinstance(t, list) and t[0] != "none" and t[1] != "?": # not in ["none", "?"]:
				if not any([s == "<%s=\"%s\">" % (t[0].strip(), t[1].strip()) for s in slots_found]):
					unknown_slots = True
	# if unknown_slots:
	# 	print("-"*50 + "\nTemplate: \"%s\"\nFound slots: %s\nTurn vals: %s\nFinal decision: %s" % (temp, str(slots_found), str(turn_vals), str(unknown_slots)), file=sys.stderr)
	return unknown_slots

def correct_spelling(s):
	global spell, missspellings
	for wrong, correct in missspellings:
		s = s.replace(wrong, correct)
	# missspelled = spell.unknown(word_tokenize(s))
	# if len(missspelled) != 0:
	# 	print("Missspellings: %s" % (",".join(missspelled)), file=sys.stderr)
	return s

def correct_comma_separation(temp, turn_vals):
	if not "," in temp or len(turn_vals) == 0:
		return temp, turn_vals
	
	dupl_vals = [t for t in turn_vals if isinstance(t, list) and any([(t[0] == ty[0] and t[1] != ty[1]) for ty in turn_vals]) and t[0] in ["Addr", "Choice", "Post"]]
	if len(dupl_vals) == 0:
		return temp, turn_vals

	combined_new_slots = True
	counter = -1
	while combined_new_slots:
		combined_new_slots = False
		counter += 1
		slots_found = find_slots(temp)
		# if counter > 0:
		# 	print("New template: %s\nNew slots found: %s" % (temp, slots_found), file=sys.stderr)
		for s_index in range(1,len(slots_found)):
			s_ind_1 = slots_found[s_index-1].split("=")[0][1:]
			s_ind_2 = slots_found[s_index].split("=")[0][1:]
			s_val_1 = slots_found[s_index-1].split("=")[1][:-1].replace("\"","")
			s_val_2 = slots_found[s_index].split("=")[1][:-1].replace("\"","")
			if s_ind_1 == s_ind_2:
				for c in ([",",", "," , "] if s_ind_1 == "Addr" else [","]):
					comb = slots_found[s_index-1]+c+slots_found[s_index]
					if comb in temp:
						# if s_ind_1 != "Addr":
						# 	print("-"*150 + "\nTemplate: %s\nTurn vals: %s\nDuplicate vals: %s\nSlots found: %s" % (temp, str(turn_vals), str(dupl_vals), str(slots_found)), file=sys.stderr)
						new_slots = "<%s=\"%s\">" % (s_ind_1, s_val_1 + c + s_val_2)
						temp = temp.replace(comb, new_slots)
						turn_val_1 = [t for t in turn_vals if (isinstance(t, list) and t[0] == s_ind_1 and t[1] == s_val_1)][0]
						turn_val_2 = [t for t in turn_vals if (isinstance(t, list) and t[0] == s_ind_2 and t[1] == s_val_2)][0]
						turn_vals.remove(turn_val_1)
						if turn_val_2 not in turn_vals:
							print(turn_vals)
							print(turn_val_1)
							print(turn_val_2)
						turn_vals.remove(turn_val_2)
						turn_vals.append([turn_val_1[0], turn_val_1[1] + c + turn_val_2[1]])
						combined_new_slots = True
						break
			if combined_new_slots:
				break

	# if counter > 0:
	# 	print("="*150 + "\nFinal template: %s\nFinal turn vals: %s" % (temp, str(turn_vals)), file=sys.stderr)
	# 	# sys.exit(1)

	return temp, turn_vals

PHONE_REGEX = re.compile(r"[0-9]{10,11}")
TRAIN_ID_REGEX = re.compile(r"[0-9]{4}")
def correct_turn_acts(turn_acts):
	global PHONE_REGEX, TRAIN_ID_REGEX
	for key, turn_vals in turn_acts.items():
		for t in turn_vals:
			changed_slot = True
			while changed_slot:
				prev_t0 = t[0]
				changed_slot = False
				t[1] = t[1].strip()
				if t[0] != "Post" and t[1] != "?" and len(t[1]) > 1 and t[1][-1] in [".", "?", "!"]:
					print("Changing %s to %s" % (t[1], t[1][:-1]))
					t[1] = t[1][:-1]
				t1_wo_spaces = t[1].replace(" ","")
				if len(t1_wo_spaces) == 0:
					print("[!] ERROR: Turn val is empty: (%s, %s)" % (t[0], t[1]))
				if t[0] != "Id" and t[1].startswith("tr") and len(t1_wo_spaces) == 6 and TRAIN_ID_REGEX.match(t1_wo_spaces[2:]):
					t[0] = "Id"
					changed_slot = True
				elif t[0] != "Phone" and PHONE_REGEX.match(t1_wo_spaces):
					t[0] = "Phone"
					changed_slot = True
				elif t[0] not in ["Addr", "Name", "Dest", "Depart"] and "street" in t[1]:
					t[0] = "Addr"
					changed_slot = True
				elif t[0] not in ["Time", "Arrive", "Leave"] and "minutes" in t[1].lower():
					print("Looks like a time: <%s=\"%s\">" % (t[0], t[1]))
					t[0] = "Time"
					changed_slot = True
				elif t[0] not in ["Ticket", "Fee", "Price"] and "pound" in t[1].lower():
					t[0] = "Ticket"
					changed_slot = True
				elif t[0] == "Area":
					if t[1].startswith("in"):
						t[1] = t[1][2:].strip()
					if t[1].startswith("the"):
						t[1] = t[1][3:].strip()
					if t[1].lower() in ["thai"]:
						t[0] = "Food"
						changed_slot = True
					if t[1].lower() == "that":
						t[1] = "that area"
				elif t[0] == "Id":
					if not any([t[1].startswith(s) for s in ["tr", "train"]]): # IDs are always for train, and start either by "tr" or "train"
						if len(t[1]) == 8:
							t[0] = "Ref"
							changed_slot = True
				elif t[0] == "Ref":
					if len(t[1]) != 8:
						# Reference numbers have exactly 8 letters/numbers
						if any([t[1].startswith(s) for s in ["tr", "train"]]):
							t[0] = "ID"
							changed_slot = True
						else:
							for pos_val, correct_slot in zip(["friday", "i will get that reference number", "dojo noodle bar"], ["Day", "none", "Name"]):
								if t[1].lower() == pos_val:
									t[0] = correct_slot
									if t[0] == "none":
										t[1] = "none"
									changed_slot = True
									break
				elif t[0] == "Stay":
					if t[1].lower() == "nna3oc5w":
						t[0] = "Ref"
						changed_slot = True
				elif t[0] == "Ticket":
					pass
				elif t[0] == "Addr":
					for pos_val, correct_slot in zip(["cb11ly", "only address"],["Post", "none"]):
						if t[1] == pos_val:
							t[0] = correct_slot
							if t[0] == "none":
								t[1] = "none"
							changed_slot = True
							break
					if not changed_slot:
						for wrong_starts in ["located at ", "it is in ", "is ", "in ", "it 's on ", "it's on ", "in the ", "on "]:
							if t[1].startswith(wrong_starts):
								t[1] = t[1][len(wrong_starts):]
				elif t[0] == "Day":
					if t[1].lower() == "that":
						t[1] = "that day"
				elif t[0] == "Time":
					if t[1].isdigit():
						if len(t[1]) == 4:
							t[1] = t[1][:2] + ":" + t[1][2:]
						else:
							t[1] = t[1] + " minutes"
				if changed_slot:
					print("Changed slot from == %s == to == %s == for value %s" % (prev_t0, t[0], t[1]))
	return turn_acts

REPLACEMENT_LIST = [
	("<Area=\"all over town\"> , <Area=\"except in the north\">", "<Area=\"all over town , except in the north\">"),
	("<Choice=\"many\"> , <Choice=\"many\">", "<Choice=\"many , many\">")
]
POSSIBLE_MISSING_SLOT_WORDS = ["west", "east", "north", "south", "centre", "center", "museum", "theater", "saint", "1", "2"]
POSSIBLE_MISSING_SLOT_WORD_REGEXS = [create_regex_for_slot_val(s) for s in POSSIBLE_MISSING_SLOT_WORDS]
NOT_EXPECTED_FEES = [
	"\"the entrance fee is unknown\"",
    "\"i do n't know\"",
    "\"do not have\"",
    "\"need to call\"",
    "\"do not have information\"",
    "\"do not have the entrance fee\"",
    "\"i 'm sorry ; i do n't have that information\"",
    "\"i do n't have any information about fees\"",
    "\"i do n't have the entrance fees\"",
    "\"i 'm afraid that i do n't know their entrance fee\"",
    "\"do not have entrance fee info\"",
    "\"do n't have an entrance fee\"",
    "\"none have entrance fees listed\"",
    "\"entrance fee\"",
    "\"i do n't have their entrance fee\"",
    "\"no listing\"",
    "\"i do not have entrance fee information\"",
    "\"a cost\"",
    "\"do n't have price info\"",
    "\"i do not have any information on an entrance fee\"",
    "\"do not know\"",
    "\"does n't have a fee\"",
    "\"do n't have the information to provide\"",
    "\"i do not have information\"",
    "\"not listed\"",
    "\"can not disclose any admission fee 's\"",
    "\"no entrance fee information listed\"",
    "\"do n't have that information available\"",
    "\"pay admission\"",
    "\"i do not have access to that info\"",
    "\"i do n't have information\"",
    "\"do n't have admission information\"",
    "\"i do n't have any information on the entrance fee\"",
    "\"there is n't any entrance fee\"",
    "\"i do n't have any info\"",
    "\"no entrance fee information is available\"",
    "\"no information on the admission price\"",
    "\"they do not charge an entrance fee\"",
    "\"no entrance fee listed\"",
    "\"not listed .\"",
    "\"i 'm sorry but you will have to go there in person to see the fee\"",
    "\"no entrance fee information available\"",
    "\"i do n't have that information\"",
    "\"information is not available\"",
    "\"do n't have the entrance fee available\"",
    "\"i do not show an entrance fee amount\"",
    "\"unsure of their entrance fee\"",
    "\"i do n't see any entrance fee\"",
    "\"fees vary by boat type and length of time rented\"",
    "\"does not list an entrance fee\"",
    "\"not in our system\"",
    "\"no information about their entrance fee\"",
    "\"do not have that information\"",
    "\"not sure if they charge an entrance fee\"",
    "\"does not list\"",
    "\"i 'm not sure\"",
    "\"do not have the information on the entrance fee\"",
    "\"it 's not showing up\"",
    "\"i do n't have access to their entrance fee\"",
    "\"admission charge\"",
    "\"unsure of the entrance fee\"",
    "\"the information i have does n't include a entrance fee\"",
    "\"the entrance fee isnt available on line\"",
    "\"can not look up the entry fee\"",
    "\"do not have any information\"",
    "\"not sure about the entrance fee\"",
    "\"not an issue\"",
    "\"do not have the information\"",
    "\"unavailable\"",
    "\"do not have it\"",
    "\"not available\"",
    "\"i do not see an entrance fee listed\"",
    "\"uncertain\"",
    "\"i ca n't find the entrance fee information\"",
    "\"entrance fee information is not listed\"",
    "\"some are free some are n't\"",
    "\"they do not have an entrance fee listed\"",
    "\"i am sorry i have no information about their entrance fee\"",
    "\"i 'm unsure\"",
    "\"not present in our database\"",
    "\"do not see any information\"",
    "\"there is n't an entrance fee listed\"",
    "\"do not have the entrance fee information\"",
    "\"do n't have an entrance fee listed\"",
    "\"our system does not have information\"",
    "\"do n't have\"",
    "\"no entrance information is provided\"",
    "\"the entrance fee isnt stated\"",
    "\"not have it 's entrance fee available\"",
    "\"venue has n't provided information on their entrance fee\"",
    "\"not in the system\"",
    "\"i do n't have the admission listed in my database\"",
    "\"the entrance fee is n't listed\"",
    "\"ca n't find the information in the database unfortunately\"",
    "\"does not have it 's entrance fee posted to the public\"",
    "\"unfortunately my system is not showing an entrance fee but i can give you their phone number so you can call and inquire\"",
    "\"no information\"",
    "\"system does not provide the entrance fee\"",
    "\"entrance fee is not listed\"",
    "\"unable to find the entrance fee\"",
    "\"no info\"",
    "\"does n't tell me what the entrance fee\"",
    "\"system does not show fees\"",
    "\"no information on the entrance fee listed\"",
    "\"i not have the information about the entry fee\"",
    "\"database does not have that information currently available\"",
    "\"not known\"",
    "\"unable to view\"",
    "\"do n't have a listing\"",
    "\"you 'll have to call them directly for entrance fee information\"",
    "\"almost all of them are free\"",
    "\"there is n't an admission fee\"",
    "\"have n't listed their entrance fees\"",
    "\"but some have a small entrance fee\"",
    "\"arent listed\"",
    "\"we have no entrance fee information for any of the theatres at this time\"",
    "\"i do n't have any information\"",
    "\"do not have the entrance fee listed\"",
    "\"the system is not saying whether there is an entrance fee\"",
    "\"ca n't find\"",
    "\"not available online\"",
    "\"they often change so we do not have that information\"",
    "\"does not tell me if there is an entrance fee\"",
    "\"the entrance fee is not listed\"",
    "\"does not show\"",
    "\"entrance fee is not listed in our system at this time\"",
    "\"the entrance fee is not currently listed\"",
    "\"do not have it listed\"",
    "\"no entrance fee in our data base\"",
    "\"not sure\"",
    "\"unfortunately i do n't have any information on the entrance fee\"",
    "\"they do n't list an entrance fee\"",
    "\"do n't have access\"",
    "\"do n't have an entrance fee available\"",
    "\"i do not have information on the entrance fee\"",
    "\"i do not have any information\"",
    "\"does not have\"",
    "\"i 'm not showing any information on the entrance fee\"",
    "\"is n't listed\"",
    "\"i do n't have any information about their entrance fee\"",
    "\"can only be seen once at the location\"",
    "\"there is no admission fee currently , so you may want to call\"",
    "\"do n't have fee information\"",
    "\"some have entrance fees\"",
    "\"do n't know the entrance fee\"",
    "\"i do not unfortunately have information\"",
    "\"do n't have any information listed\"",
    "\"do n't have any information\"",
    "\"paid\"",
    "\"i do not know\"",
    "\"not sure if it 's free or if they charge anything\"",
    "\"you have to call to inquire about the entrance fee\"",
    "\"west\"",
    "\"no entrance fee information availble\"",
    "\"centre of town\"",
    "\"unable to see\"",
    "\"cost money\"",
    "\"absent from my database\"",
    "\"my records do not show\"",
    "\"do n't have the information\"",
    "\"unsure\"",
    "\"that information is not available to me\"",
    "\"none of them have a listed entrance fee\"",
    "\"is n't shown\"",
    "\"we do not have any information\"",
    "\"i am not sure\"",
    "\"do n't have that information\"",
    "\"we do n't have the entrance fee listed\"",
    "\"do not have access\"",
    "\"no entrance fees are listed\"",
    "\"not provided\"",
    "\"unlisted\"",
    "\"do n't have information\"",
    "\"does not look like there is an entrance fee\"",
    "\"do n't have any information on the entrance fee\"",
    "\"i 'm not sure if there is an entrance fee\"",
    "\"i do not have any information on what their fee is\"",
    "\"do not have an entrance fee listed\"",
    "\"are n't listed\"",
    "\"do n't have entrance fee info\"",
    "\"i do n't have the entrance fee\"",
    "\"not sure if there are any fees\"",
    "\"does n't have an entrance fee\"",
    "\"no info on their entrance fee\"",
    "\"only available at the location\"",
    "\"the entrance fee is not available\"",
    "\"i do n't have the entrance fee information\"",
    "\"i do n't have any information about the fee\"",
    "\"i am not able to see the price of the entrance fee\"",
    "\"do not show the entrance fee in my database\"",
    "\"i 'm sorry\"",
    "\"not showing anything\"",
    "\"has not provided any information\"",
    "\"so you may want to call their box office\"",
    "\"no information on entrance fee\"",
    "\"i do n't have the entrance fee listed here\"",
    "\"not stated\"",
    "\"not an entrance fee listed\"",
    "\"i do not have any information abotu the entrance fee\"",
    "\"they do not note if an entrance fee is required\"",
    "\"i do n't have the fee information\"",
    "\"they do not list their entrance fee\"",
    "\"there is no information\"",
    "\"fluctuate based on time of year so we do not have them\"",
    "\"do n't show the entrance fee\"",
    "\"not listed online\"",
    "\"not able to see their entrance fee\"",
    "\"the admission fee is not currently available\"",
    "\"do not have inform\"",
    "\"no entrance fee 's\"",
    "\"neither venue provides their entrance fee to us\"",
    "\"do not know the entrance fee\"",
    "\"a mystery\"",
    "\"no info fee information\"",
    "\"do n't know however if they charge a fee\"",
    "\"church\"",
    "\"charge\"",
    "\"not in my database\"",
    "\"we do n't have that information\"",
    "\"unfortunately\"",
    "\"free of charge\"",
    "\"do not\"",
    "\"i do n't have any entrance fee information\"",
    "\"do not have information on their entrance fee\"",
    "\"does n't seem to have an entrance fee\"",
    "\"fee\"",
    "\"do not have the entrance fee available\"",
    "\"entrance fees\"",
    "\"i do not have that information\"",
    "\"suspect it varies\"",
    "\"no currently listed\"",
    "\"i do n't have info\"",
    "\"they can inform you\"",
    "\"have to call\"",
    "\"not information about the entrance fee\"",
    "\"no entry fee\"",
    "\"currently unavailable\"",
    "\"not sure of\"",
    "\"not showing\"",
    "\"does not have their entrance fee available\""
]
def perform_final_check(sent, key):
	global REPLACEMENT_LIST
	for parts_to_replace, replace_value in REPLACEMENT_LIST:
		if parts_to_replace in sent:
			sent = sent.replace(parts_to_replace, replace_value)
	
	if "'' >" in sent or "< ''" in sent:
		# print("#"*50)
		# print("[#] WARNING: Found (and will remove) akward sentence %s" % sent)
		# print("#"*50)
		return None


	if "-Inform:" in key and key.split(":")[-1] != "none":
		sub_sents = sent_tokenize(sent)
		while len(sub_sents) > 1:
			for s_index in range(len(sub_sents)):
				if sub_sents[s_index].count("<") == sub_sents[s_index].count(">"):
					continue
				elif sub_sents[s_index].count("<") >= sub_sents[s_index].count(">"):
					if s_index < len(sub_sents)-1 and sub_sents[s_index+1].strip().startswith("\">"):
						sub_sents[s_index] += "\">"
						sub_sents[s_index+1] = sub_sents[s_index+1][2:]
				else:
					sub_sents[s_index] = sub_sents[s_index].strip()
					if sub_sents[s_index].startswith("\">"):
						sub_sents[s_index] = sub_sents[s_index][2:]
			last_sent = sub_sents[-1].strip()
			if len(last_sent) > 0 and last_sent[-1] == "?":
				if "<" not in last_sent:
					# print("*"*50)
					# print("[#] WARNING: Found additional question in inform sentence (key %s) %s" % (key, sent))
					# print("*"*50)
					sent = " ".join(sub_sents[:-1])
				elif "," in last_sent and ">" not in last_sent.rsplit(",",1)[-1]:
					# print("+"*50)
					# print("[#] WARNING: Found additional question (separated by comma) in inform sentence (key %s) %s" % (key, sent))
					# print("+"*50)
					sent = " ".join(sub_sents[:-1] + last_sent.rsplit(",", 1)[:-1] + ["."])
				else:
					break
				sub_sents = sent_tokenize(sent)
			else:
				break

	for slot_word, slot_word_regex in zip(POSSIBLE_MISSING_SLOT_WORDS, POSSIBLE_MISSING_SLOT_WORD_REGEXS):
		if slot_word in sent and (slot_word_regex.search(sent) is None):
			# print("-"*50)
			# print("[#] WARNING: Possible missing slot (word %s) in sentence %s" % (slot_word, sent))
			# print("-"*50)
			return None

	if any([("<fee=%s>" % f.lower()) in sent.lower() for f in NOT_EXPECTED_FEES]):
		# print("<>"*25)
		# print("[#] WARNING: Unwanted values for fee's included: %s" % sent)
		# print("<>"*25)
		return None

	if " fee " in sent.lower() and "<Fee=" not in sent and "<Ticket=" not in sent:
		# print("%"*50)
		# print("[#] WARNING: Talking about entrance fee without giving any value: %s" % sent)
		# print("%"*50)
		return None

	words = [w.strip() for w in sent.split(" ") if len(w) > 0]
	while words[-1] == "." and words[-2] in ["?", "!", "."]:
		del words[-1]
	sent = " ".join(words).strip()

	if len(sent) < 2:
		print("[#] WARNING: Remaining sentence is: %s" % sent)
		sent = None
	elif sent[1] == "." and sent[2] == " ":
		sent = sent[3:]
	elif sent[0] == "." and sent[1] == " ":
		sent = sent[2:]

	return sent

def get_len_of_sent(sent):
	sent_splits = SLOT_REGEX.split(sent)
	num_words = sum([len([w for w in s.split(" ") if len(w) > 0]) for s in sent_splits])
	num_slots = len(sent_splits) - 1
	return num_words + num_slots

def preprocess_dataset(debug=False):
	global SLOT_REGEX

	print("Loading data...")
	with open("data.json", "r") as f:
		data = json.load(f)

	if debug and len(data.keys()) > 1000:
		keys_to_remove = list(data.keys())[1000:]
		for k in keys_to_remove:
			data.pop(k)

	conversations = dict()

	print("Tokenizing conversations...")
	for conv_key, conv_data in data.items():
		conv_key = conv_key.split(".")[0]
		conversations[conv_key] = {("B" if i%2 else "U") + str(int(i//2)): correct_spelling(" ".join(word_tokenize(c["text"]))) for i, c in enumerate(conv_data["log"])}

	with open("conversations.json", "w") as f:
		json.dump(conversations, f, indent=4) 

	print("Loading dialogue acts...")
	with open("dialogue_acts.json", "r") as f:
		dialogue_acts = json.load(f)

	conv_file = open("conversations_templates.txt", "w")
	counter = 0
	answ_by_acts = dict()
	for conv_enum_index, conv_key in enumerate(data.keys()):
		print("Processing conversation %i of %i (%4.2f%%)" % (conv_enum_index, len(data.keys()), conv_enum_index * 100.0 / len(data.keys())), end="\r")
		act_key = conv_key.split(".")[0]
		print("="*100, file=conv_file)
		print("Conversation \"%s\"" % (conv_key), file=conv_file)
		# print(conv_key, file=sys.stderr)
		known_slots = list()
		for dialogue_turn_id in sorted(list(dialogue_acts[act_key].keys())):
			turn_int = int(dialogue_turn_id) * 2 - 1
			# print(turn_int, file=sys.stderr)
			if turn_int >= len(data[conv_key]["log"]):
				print("Maximum turn id not sufficient: %i | %i (dialogue %s)" % (turn_int, len(data[conv_key]["log"]), conv_key), file=sys.stderr)
				continue
			turn = data[conv_key]["log"][turn_int]
			sys_resp = turn["text"]
			for c1, c2 in [(".\n.", ". "), ("?\n.", "? "), ("?\n", "? "), ("!\n.", "! "), ("!\n", "! "), ("\n.",". "), (".\n",". "), ("\n",". "), ("\t"," ")]:
				sys_resp = sys_resp.replace(c1,c2)
			sys_resp = sys_resp.rstrip()
			needed_corrections = re.findall(r"([.][A-Z])|([a-zA-Z]:[a-zA-Z0-9])", sys_resp)
			for ndc in needed_corrections:
				sys_resp = sys_resp.replace(ndc[0], ndc[0].replace("."," . ").replace(":"," : "))
			turn_acts = dialogue_acts[act_key][dialogue_turn_id]

			print("-"*100, file=conv_file)
			print("System response: \"%s\"" % (sys_resp), file=conv_file)
			print("Dialogue actions: ", file=conv_file)

			if isinstance(turn_acts, dict):
				turn_acts = correct_turn_acts(turn_acts)
				for turn_identification, turn_vals in turn_acts.items():
					s = "\t\"%s\": " % turn_identification
					s_list = list()
					for l in turn_vals:
						if isinstance(l, list):
							s_list.append("%s - %s" % (l[0], l[1]))
						elif isinstance(l, str):
							s_list.append("%s" % l)
						else:
							print("[!] ERROR: Unknown type of turn val: \"%s\"" % (str(l)))
					print(s + ", ".join(s_list), file=conv_file)

				sys_temp = sys_resp
				known_slots = known_slots
				turn_vals = [d for tvals in turn_acts.values() for d in tvals]
				# for turn_identification, turn_vals in turn_acts.items():
				prev_repl = []
				turn_vals = sorted(turn_vals, key=lambda x: len(x[1]) if isinstance(x, list) else len(str(x)), reverse=True)
				for l in turn_vals:
					if isinstance(l, list): 
						l[1] = l[1].strip()
						l[0] = l[0].strip()
						if (l[0] + " - " + l[1]) in prev_repl or (l[0] == "none" and l[1] == "none"):
							continue
						if l[1] == "none" and l[0] != "none":
							test_c = "".join(["[%s]"%s for s in l[0]])
							found_replacements = re.findall(r"(( |^|,)"+ test_c +r"($|[ ,?!]|([.]$)|([.][ ])))", sys_temp)
							for frpl in found_replacements:
								sys_temp = sys_temp.replace(frpl[0], frpl[0].replace(l[0], "<%s=\"%s\">" % ("entity", l[0])))
						elif l[1] != "?":
							test_c = "".join(["[%s]"%s for s in l[1]])
							found_replacements = re.findall(r"(( |^|,)"+ test_c +r"($|[ ,?!]|([.]$)|([.][ ])))", sys_temp)
							for frpl in found_replacements:
								sys_temp = sys_temp.replace(frpl[0], frpl[0].replace(l[1], "<%s=\"%s\">" % (l[0], l[1])))
						prev_repl.append(l[0] + " - " + l[1])

				print("System template: \"%s\"" % (sys_temp), file=conv_file)

				# sys_temp, turn_acts = correct_comma_separation(sys_temp, turn_acts)

				act_vals = [(turn_ids, turn_vs) for turn_ids, turn_vs in turn_acts.items()]
				act_vals = sorted(act_vals, key=lambda x: is_question(x[0]), reverse=True)
				# if conv_key == "MUL2179.json": # "MUL0592.json"
				# 	print("Act vals: %s\nTurn acts: %s" % (str(act_vals), str(turn_acts)), file=sys.stderr)
				# 	sys.exit(1)
				for turn_identification, turn_vals in act_vals:
					if len(sys_temp) == 0:
						continue
					if is_question(turn_identification) and len(turn_acts.keys()) > 1:
						# if "?" not in sys_temp:
						# 	print("[*] WARNING: Could not find \"?\" in response although turn identification was %s." % str(turn_identification))
						sents_sys_temp = sent_tokenize(sys_temp)
						turn_sys_temp = None
						if sys_temp.count("?") > 1:
							# print("[#] WARNING: Found more than one \"?\" in question. (%s | %s)" % (sys_temp, str(turn_acts)))
							continue # If we have more than one "?" in our response, it is hard to say what part of the sentence belongs to which label. Hence, we skip those
						for s in sents_sys_temp[::-1]:
							if s[-1] == "?":
								turn_sys_temp = s
								ind = sents_sys_temp.index(s)
								if ind > 0 and ind < len(sents_sys_temp) - 1:
									prev_sentences = " ".join(sents_sys_temp[ind+1:])
									if "<" not in prev_sentences:
										turn_sys_temp += " " + prev_sentences
										del sents_sys_temp[ind+1:]
								sents_sys_temp.remove(s)
								sys_temp = " ".join(sents_sys_temp)
								break
						if turn_sys_temp is None:
							# print(sents_sys_temp, file=sys.stderr)
							# print(sys_temp, file=sys.stderr)
							turn_sys_temp = sents_sys_temp[-1]

						if "<" in turn_sys_temp and ">" in turn_sys_temp and not any([False] + [(v[1] != "?" and v[0] != "none") for v in turn_vals]):
							sp = turn_sys_temp.split(",")
							sents_sys_temp += sp[:-1]
							turn_sys_temp = sp[-1]
							if "<" in turn_sys_temp and ">" in turn_sys_temp:
								continue
						if turn_sys_temp.startswith("\">"):
							sents_sys_temp += "\">."
							turn_sys_temp = turn_sys_temp[2:]
					else:
						turn_sys_temp = sys_temp

					
					if check_unknown_slots(turn_sys_temp, turn_vals):
						sent_sys_temp = sent_tokenize(turn_sys_temp)
						turn_sys_temp = ""
						for sent in sent_sys_temp:
							if not check_unknown_slots(sent, turn_vals):
								turn_sys_temp += sent
						if len(turn_sys_temp) == 0:
							continue

					turn_sys_temp, turn_vals = correct_comma_separation(turn_sys_temp, turn_vals)

					turn_identification += ":" + ",".join(sorted([l[0] + ("?" if l[1] == "?" else "") if isinstance(l, list) else str(l) for l in turn_vals]))
					if turn_identification not in answ_by_acts:
						answ_by_acts[turn_identification] = dict()
					answ_by_acts[turn_identification]["%s_B%s" % (act_key, dialogue_turn_id)] = turn_sys_temp
			else:
				print("\t%s" % (str(turn_acts)), file=conv_file)
		print("-"*100, file=conv_file)
		print("="*100, file=conv_file)
		counter += 1
		# if counter > 10:
		# 	sys.exit(1)

	for key, vals in answ_by_acts.items():
		to_remove = list()
		for k, sent in vals.items():
			if "<Open=" in sent: # Remove everything with slot '<Open=...>' because it does not contain valuable information 
				to_remove.append(k)
			if len(sent.strip()) == 0:
				to_remove.append(k)
		[vals.pop(k) for k in to_remove]

	print("Tokenizing...")
	unique_slots = list()
	to_remove = list()
	answ_lens = dict()
	for key, vals in answ_by_acts.items():
		if key not in answ_lens:
			answ_lens[key] = dict()
		for k, sent in vals.items():
			sent = correct_spelling(sent)
			sent_splits = SLOT_REGEX.split(sent)
			sent_splits = [" ".join(word_tokenize(s)) for s in sent_splits]

			slots = find_slots(sent)
			slots = ["<%s=\"%s\">" % (s.split("=")[0][1:], " ".join(word_tokenize(s.split("=")[1][1:-2]))) for s in slots]

			tokenized_sent = " ".join([(slots[i-1] if i > 0 else "") + " " + sent_splits[i] for i in range(len(sent_splits))])
			# if len(slots) > 0:
			# 	print("-"*50 + "\nInput sentence: %s\nTokenized sentence: %s" % (sent, tokenized_sent))
			tokenized_sent = tokenized_sent.strip()
			corrected_sent = perform_final_check(tokenized_sent, key)
			if corrected_sent is not None:
				if corrected_sent == tokenized_sent or key.split(":")[1] == "none":
					answ_by_acts[key][k] = corrected_sent
					answ_lens[key][k] = get_len_of_sent(corrected_sent)
				else:
					# FIND KEY AGAIN!
					slot_keys = sorted([s.split("=")[0][1:] for s in find_slots(corrected_sent)])
					prev_slots = key.split(":")[-1].split(",")
					slots_to_remove = []
					for pslot in prev_slots:
						if "?" in pslot:
							continue
						elif pslot in slot_keys:
							slot_keys.remove(pslot)
						else:
							slots_to_remove.append(pslot)
					[prev_slots.remove(x) for x in slots_to_remove]
					new_key = key.split(":")[0] + ":" + ",".join(prev_slots)
					if new_key != key:
						to_remove.append((key, k))
						print("Old sentence: %s\nNew sentence: %s\nOld key: %s\nNew key: %s" % (tokenized_sent, corrected_sent, key, new_key))
					answ_by_acts[new_key][k] = corrected_sent
					if new_key not in answ_lens:
						answ_lens[new_key] = dict()
					answ_lens[new_key][k] = get_len_of_sent(corrected_sent)
				unique_slots += [s.lower() for s in slots]
			else:
				to_remove.append((key, k))
		unique_slots = list(set(unique_slots))

	print("Removing %i sentences" % (len(to_remove)))
	[answ_by_acts[key].pop(k) for key, k in to_remove]

	mean_lens = {key: (mean(val_dict.values()) if len(val_dict.values()) > 0 else -1) for key, val_dict in answ_lens.items()}
	median_lens = {key: (median(val_dict.values()) if len(val_dict.values()) > 0 else -1) for key, val_dict in answ_lens.items()}
	stdev_lens = {key: (stdev(val_dict.values()) if len(val_dict.values()) > 1 else -1) for key, val_dict in answ_lens.items()}
	for key in answ_lens.keys():
		for subkey in answ_lens[key].keys():
			answ_lens[key][subkey] = (answ_lens[key][subkey], int(abs(answ_lens[key][subkey]-median_lens[key]) > stdev_lens[key]*2.5) if stdev_lens[key] > 0 else 0)
		answ_lens[key]["additional_information"] = {"mean": mean_lens[key], "median": median_lens[key], "stdev": stdev_lens[key]}
	with open("answ_lens.json", "w") as f:
		json.dump(answ_lens, f, indent=4, sort_keys=True)

	rem_counter = 0
	old_keys = list(answ_by_acts.keys())
	for key in old_keys:
		old_sub_keys = list(answ_by_acts[key].keys())
		for k in old_sub_keys:
			if answ_lens[key][k][1] == 1:
				answ_by_acts[key].pop(k)
				rem_counter += 1
	print("Removed %i sentence because of being too long." % (rem_counter))

	unique_slot_dict = dict()
	for slot in unique_slots:
		key = slot.split("=")[0][1:]
		if key not in unique_slot_dict:
			unique_slot_dict[key] = list()
		unique_slot_dict[key].append(slot.split("=")[-1][:-1])

	keys_to_remove = ["Train-Inform:none", "general-welcome:none", "general-greet:none", "general-bye:none"]
	keys_to_remove += [k for k in answ_by_acts.keys() if k.count("?") > 1]
	keys_to_remove += [k for k in answ_by_acts.keys() if k.count("Choice") > 1]
	keys_to_remove = list(set(keys_to_remove))
	for k in keys_to_remove:
		print("Removing key %s with %i elements..." % (k, len(answ_by_acts[k].keys())))
		_ = answ_by_acts.pop(k)
		if k in answ_by_acts:
			print("Could not remove key %s..." % k)
			sys.exit(1)

	print("="*50)
	print("#Examples per dialogue actions")
	print("-"*50)
	answ_lens = [(key, len(val)) for key, val in answ_by_acts.items()]
	answ_lens = [e for e in answ_lens if e[1] > 2]
	answ_lens = sorted(answ_lens, key=lambda x: x[1], reverse=True)
	for key, val_len in answ_lens[:5]:
		print("%s - %i" % (key, val_len))
	print("-"*50)
	print("Overall %i examples and %i different dialogue actions" % (sum([e[1] for e in answ_lens]), len(answ_lens)))
	print("="*50)

	answ_by_acts = {key: val for key, val in answ_by_acts.items() if len(val) > 2}

	with open("unique_slots.json", "w") as f:
		json.dump(unique_slot_dict, f, indent=4, sort_keys=True)

	with open("answers_by_acts.json", "w") as f:
		json.dump(answ_by_acts, f, indent=4, sort_keys=True)

def export_split_dataset(num_sents_per_exmp=5, num_incl_acts=100, num_excl_acts=100):
	with open("answers_by_acts.json", "r") as f:
		answ_by_acts = json.load(f)

	train_acts = answ_by_acts
	val_acts = dict()

	excl_acts = [key for key, vals in train_acts.items() if len(vals) <= num_sents_per_exmp + 1 and len(vals) >= num_sents_per_exmp - 1]
	shuffle(excl_acts)
	excl_acts = excl_acts[:min(len(excl_acts), num_excl_acts)]
	print("Number of exclusive actions: %i" % (len(excl_acts)))

	for k in excl_acts:
		val_acts[k] = train_acts.pop(k)

	incl_acts = [key for key, vals in train_acts.items() if len(vals) > num_sents_per_exmp*2 and len(vals) < num_sents_per_exmp*4]
	shuffle(incl_acts)
	incl_acts = incl_acts[:min(len(incl_acts), num_incl_acts)]
	print("Number of inclusive actions: %i" % (len(incl_acts)))

	for k in incl_acts:
		val_acts[k] = dict()
		incl_keys = list(train_acts[k].keys())[:num_sents_per_exmp]
		for act_key in incl_keys:
			val_acts[k][act_key] = train_acts[k].pop(act_key)

	with open("train.json", "w") as f:
		json.dump(train_acts, f, indent=4, sort_keys=True)

	with open("val.json", "w") as f:
		json.dump(val_acts, f, indent=4, sort_keys=True)

	with open("test.json", "w") as f:
		json.dump({}, f, indent=4, sort_keys=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--only_split", help="If chosen, the dataset will not be preprocessed, but an already saved version is used.", action="store_true")
	parser.add_argument("--debug", help="If chosen, a smaller fraction of the dataset will be used.", action="store_true")
	args = parser.parse_args()

	if not args.only_split:
		preprocess_dataset(debug=args.debug)
	export_split_dataset()

