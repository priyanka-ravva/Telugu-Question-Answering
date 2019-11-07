import os,sys
f = open("whole_questions.txt","w")
dir_path = "/home/priyanka/Desktop/QA/Icon/svm_model/github/Questions/"
list_files = os.listdir(dir_path)

print(list_files)

for filename in list_files:
	fname = open(dir_path+filename,"r").readlines()
	for line in fname:
		f.write(line.strip()+"\n")

f.close()
