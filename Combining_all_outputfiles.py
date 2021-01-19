# -*- encoding: utf-8 -*-

### Note: if more files of 'Result_Q_10Ans_Tel.txt' files for a category/whole_queries then iterate over all Result_Q_10Ans_Tel.txt files to get accuracy. Because QA system may get struck due to internet slow, timelimit,  any other reasons.. 

f = open("Result_Q_10Ans_Tel_Combine_all.txt","w")
input_file = "output_files/Result_Q_10Ans_Tel.txt" 
res = open(input_file,"r").readlines()
all_samples_sum += len(res)

for line in res:
	line = line.strip()
	f.write(line+"\n")

f.close()
print("all_samples_sum :  ",all_samples_sum)



### removing the duplications:
t = open("OVER_ALL_QA_DATA.txt","r").readlines()
p = open("Result_Q_10Ans_Tel_Combine_all.txt","r").readlines()

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
#m = open("time_50.txt","w")


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

