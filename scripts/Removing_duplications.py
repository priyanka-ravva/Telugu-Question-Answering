# -*- encoding: utf-8 -*-
t = open("OVER_ALL_QA_DATA.txt","r").readlines()
p = open("Result_Q_10Ans_Tel_Combine_all.txt","r").readlines()

#from indic_transliteration import sanscript
#from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


Target_Questions = []
Target_Answers 	 = []


for line in t:
	line = line.strip()
	line = line.split(":::")
	Target_Answers.append(line[0].strip())
	Target_Questions.append(line[1].strip())


print(len(Target_Answers),len(Target_Questions))
print("\n#########################################\n")

m = open("Result_Q_10Ans_Tel_Combined_all_with_Unique_queries.txt","w")
#m = open("percentage_40.txt","w")

Predicted_Questions = []
Predicted_Answers   = []
print("# of lines in a File: ",len(p))

for line in p:
	input_sent = line.strip()
	line = line.strip()
	line = line.split("@@")
	#print(line[1])
	ans = line[1].split("$$$")
	#print(ans,len(ans))
	#break
	if not(line[0] in Predicted_Questions):
		Predicted_Questions.append(line[0])
		Predicted_Answers.append(ans)
		m.write(input_sent+"\n")

print(len(Predicted_Answers),len(Predicted_Questions))
print("\n#########################################\n")

m.close()
