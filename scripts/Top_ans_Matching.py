# -*- encoding: utf-8 -*-
#t = open("Question_Answer_pairs.txt","r").readlines()
t = open("OVER_ALL_QA_DATA.txt","r").readlines()
p = open("Result_Q_10Ans_Tel_Combined_all_with_Unique_queries.txt","r").readlines()

import numpy as np

#### MRR function #######
def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])




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


Predicted_Questions = []
Predicted_Answers   = []

for line in p:
	line = line.strip()
	line = line.split("@@")
	#print(line[1])
	ans = line[1].split("$$$")
	#print(ans,len(ans))
	#break
	Predicted_Questions.append(line[0])
	Predicted_Answers.append(ans)

print(len(Predicted_Answers),len(Predicted_Questions))
print("\n#########################################\n")


ff = open("exact_match_.txt","w")

hits = 0.0
hits_similarity_based = 0.0
thr = 0.60
count = 0

y_pred = []
y_pred_similarity = []
#################### Exact Match ###########################
for p_i, sample in enumerate(Predicted_Questions):
	n = 0
	for t_i,target in enumerate(Target_Questions):
		#print(sample.strip(),target.strip())
		if(sample.strip() == target.strip() ):
			n+=1
			tar_ans = Target_Answers[t_i].replace("%", "")
			pred_ans = [pred.replace("%", "").strip() for pred in Predicted_Answers[p_i]]
			pred_ans = [pred.replace("OOVA ", "").strip() for pred in pred_ans]  ### for Random answers...
			if(tar_ans in pred_ans): #### Exact Answer matching
				#ff.write(str(tar_ans)+"__"+str(pred_ans)+"__"+str(len(tar_ans))+str(len(pred_ans))+"\n")
				#print(tar_ans)
				#print("p: ",pred_ans)
				ind = pred_ans.index(tar_ans)+1
				#print(ind,len(pred_ans))
				temp = [0 for ans in pred_ans]
				#print("temp : ",temp)
				temp[ind] = 1
				#print("temp_ : ",temp)
				#print("\n\n")
				y_pred.append(temp)
				hits+=1
				break
	if(n==0):
		count+=1
	#break

print("Question does not have Target answers : ",count)
ff.close()
print("\n")
######################### Similarity Based ###############################

for p_i, sample in enumerate(Predicted_Questions):
	for t_i,target in enumerate(Target_Questions):
		if(sample.strip() == target.strip() ):
			tar_ans = Target_Answers[t_i].replace("%", "")
			pred_ans = [pred.replace("%", "").strip() for pred in Predicted_Answers[p_i]]
			pred_ans = [pred.replace("OOVA ", "").strip()for pred in pred_ans]  ### for Random answers...
			scores = [ similar(tar_ans,ans) for ans in pred_ans ]
			
			res = [1 for s in scores if(s >= thr)]
			if(sum(res)>=1):
				#print(p_i)
				#print(sample)
				hits_similarity_based+=1

				ind = np.argmax(scores)+1 ###pred_ans.index(tar_ans)+1

				#print(ind,len(pred_ans))
				temp = [0 for ans in pred_ans]
				#print("temp : ",temp)
				temp[ind] = 1
				#print("temp_ : ",temp)
				#print("\n\n")
				y_pred_similarity.append(temp)


				break




print("Exact Match:( hits - Out of all ), ACcuracy : ",(hits,len(Predicted_Questions)-count), (hits/(len(Predicted_Questions)-count))*100)
print("\n\n")

print("Similarity Method : ( hits - Out of all ), ACcuracy : ",(hits_similarity_based,len(Predicted_Questions)-count), (hits_similarity_based/(len(Predicted_Questions)-count))*100)
print("\n\n")

s = mean_reciprocal_rank(y_pred)
print("MRR Score: ",s)
ss = mean_reciprocal_rank(y_pred_similarity)
print("similarity based MRR score : ",ss)
