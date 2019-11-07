f = open("Result_Q_10Ans_Tel_Combine_all.txt","w")
directory_path = "/home/priyanka/Desktop/QA/Icon/svm_model/output_files/PERS/person_rb_K_40/"


num_folders = 50

all_samples_sum = 0

for i in range(num_folders):
	input_file = directory_path+str(i+1)+"/Result_Q_10Ans_Tel.txt"
	print("input_file : ",input_file)
	res = open(input_file,"r").readlines()
	all_samples_sum += len(res)
	for line in res:
		line = line.strip()
		f.write(line+"\n")


f.close()
print("all_samples_sum :  ",all_samples_sum)
	
