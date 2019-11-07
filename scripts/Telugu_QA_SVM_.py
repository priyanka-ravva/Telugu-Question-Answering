# -*- coding: utf-8 -*-
# -*- coding: latin-1 -*-
import urllib
import urllib2
from bs4 import BeautifulSoup
import re, math
import sys
from collections import Counter
from pycorenlp import StanfordCoreNLP
import json
import unicodedata
from sklearn import svm
import operator
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import metrics
import random
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tag import StanfordNERTagger
import itertools
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import itertools
from collections import Counter


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

########### Classifiers ##############
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



import warnings
warnings.filterwarnings('ignore')

prelink="https://www.bing.com/search?"


def Question_Type(qa):
	Whole_output.append(qa+"\n")	
	qq=qa.split(" ")

	per=["ఎవరు","పేరు ఏమిటి","పేరు","రాసినవారు","కనుగొన్నవారు","రచించినవారు","ఎవరి","నిర్వహించినవారు","ఎవరికి","పొందినవారు","పారంగతుఁడు","పూర్తి పేరు"," ఏ దేవుని","పేరేమిటి","తొలినటి"]
	org=[" ఏ సంస్థ","సంస్థ"," సంస్థ పేరు","కంపెనీ","సంస్థతో"]
	loc=["ఏది","ఎక్కడ","ఏ దేశ","ఏ రాష్ట్రంలో","రాష్ట్రంలో","గ్రామంను","కార్యాలయం","గ్రామం"]
	date=["ఎప్పుడు"," ఏ నెలలో","పుట్టిన రోజు","పుట్టిన","రోజు","తేదీ","డే","సంవత్సరంలో","సంవత్సరాల","సంవత్సరం"]

	time=["ఏ ","ఏ రోజున","గంటలు","సమయం","పుట్టినరోజు ","సమయంలో","పుట్టిన రోజు","పుట్టిన","రోజు","తేదీ","డే","కాలం ","సంవత్సరం"]
	#money=["శాతం","ఎంత శాతం"]
	number=["ఎన్ని సార్లు","సంఖ్య","ఎన్నో ","ఎన్నోవ","1","2","5","10","6","3","4","7","8","9","0"]
	percentage=["%","ఎంత శాతం","శాతం"]

	for item in per:
		for wd in qq:
			if(re.search(item,wd)):
				ans.append("PERSON")
	for item in org:
		for wd in qq:
			if(re.search(item,wd)):
				ans.append("ORGANIZATION")
	for item in loc:
		for wd in qq:
			if(re.search(item,wd)):
				ans.append("LOCATION")

	for item in date:
		for wd in qq:
			if(re.search(item,wd)):
				ans.append("DATE")

	for item in time:
		for wd in qq:
			if(re.search(item,wd)):
				ans.append("TIME")

	'''
	for item in money:
		for wd in qq:
			if(re.search(item,wd)):
				ans.append("MONEY")
	'''

	for item in percentage:
		for wd in qq:
			if(re.search(item,wd)):
				ans.append("PERCENTAGE")

	for item in number:
		for wd in qq:
			if(re.search(item,wd)):
				ans.append("NUMBER")

	return ans



def translate(to_translate, to_langage="auto", langage="auto"):
	agents = {'User-Agent':"Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727; .NET CLR 3.0.04506.30)"}
	
	before_trans = 'class="t0">'
	link = "http://translate.google.com/m?hl=%s&sl=%s&q=%s" % (to_langage, langage, to_translate.replace(" ", "+"))
	request = urllib2.Request(link, headers=agents)
	#print("request: ",request)
	page = urllib2.urlopen(request).read()
	result = page[page.find(before_trans)+len(before_trans):]
	result = result.split("<")[0]
	quest.append(result)
	Whole_output.append(result)        
	return result



def translate_2(to_translate, to_langage="auto", langage="auto"):
        import goslate
        gs = goslate.Goslate()
        print(gs.translate(to_translate, 'en'))
        result = gs.translate(to_translate, 'en')
        return result




def qquery(qu):
	st=qu
	print("Top 10 URLS:\n----------------------------------------------------------------\n\n")
	query_args = { 'q':st, 'num':'18' }
	encoded_args = urllib.urlencode(query_args)
	url = prelink + encoded_args
	#print(url)
	resp = urllib2.urlopen(url)
	soup = BeautifulSoup(resp, from_encoding=resp.info().getparam('charset'))
	if not(soup.findAll('li', attrs={'class':'b_algo'})):
		#print("Not able to find urls, Some Error came call back the function again....\n")
		qquery(qu)

	for listt in soup.findAll('li', attrs={'class':'b_algo'}):
		head_links = listt.find('h2')
		for link in head_links.find_all('a', href=True):
			Top_10_urls.append(link['href'])
			print(link['href'].encode('utf-8')+"\n")
			Whole_output.append(link['href'].encode('utf-8')+"\n")
	return Top_10_urls



def Sentence_Extraction():
	Numlinks=len(Top_10_urls)
	#sen=open("Whole_sentence.txt","w")
	for i in range(Numlinks):
		try:
			nP=0
			content=urllib2.urlopen(Top_10_urls[i]) ############ May be Here you may get ERROR
			htmlSource=content.read()	############ May be Here you may get ERROR
			content.close()
			contentt = BeautifulSoup(htmlSource)
			
			for para in contentt.findAll(['p']):
				pa=para.text
				whole_sent.append(pa.strip()) #all sentence writing into one list line by line
				#sen.write(pa.encode('UTF-8')+"\n") # all sentence writing into one file line by line
				nP=nP+1 # number of imp content paragraphs in each URL...........
				#print(nP)
		except IOError as e:
			print(" ")	   		
		except ValueError:
	    		print(" ")
		except:
	    		print(" ")
	    		raise

	#sen.close()
	print("\n\nTotal extracted lines :"+str(len(whole_sent))+"\n\n")


def Finding_Similarity_Matrix(q):
	cc=0
	for i in range(len(whole_sent)):
		stmt=whole_sent[i].split(".")
		for j in range(len(stmt)):
			cc=cc+1
			all_sen.append(stmt[j])	
	all_sen1=list(set(all_sen))

	WORD = re.compile(r'\w+')
	def get_cosine(vec1, vec2):
	     intersection = set(vec1.keys()) & set(vec2.keys())
	     numerator = sum([vec1[x] * vec2[x] for x in intersection])

	     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
	     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
	     denominator = math.sqrt(sum1) * math.sqrt(sum2)

	     if not denominator:
                 return(0.0)
	     else:
                 return(float(numerator) / denominator)

	def text_to_vector(text):
	     words = WORD.findall(text)
	     return(Counter(words))

	text1 =q  # given user query
	x=0
	
	for n in range(len(all_sen)):
		text2=all_sen[n]
		vector1 = text_to_vector(text1)
		vector2 = text_to_vector(text2)

		cosine = get_cosine(vector1, vector2)
		x=x+1
	
		consine_list.append(cosine)


def Most_Relative_Sentence():
	
	num=10 ##number of most relative sentences considering
	if(len(whole_sent)<num):
		num = len(whole_sent)
	tt= (sorted(consine_list, reverse=True)[:num])
	for k in range(num):
		l=[i for i,j in enumerate(consine_list) if j==tt[k]]
		myString = all_sen[l[0]].strip()
		sent = ' '.join(myString.split())
		Top_similarity_sent.append(sent)
		Whole_output.append(sent+"\n")
	return Top_similarity_sent

	
def call_back_ner_funt(text):
	nlp = StanfordCoreNLP('http://corenlp.run/')
	res = nlp.annotate(text,properties={'timeout': '1500000',"annotators": "tokenize,ssplit,pos,ner", 'outputFormat': 'json',})
	return res




import Ner_script_spacy

def get_NER(qc,query):
	nlp = StanfordCoreNLP('http://corenlp.run/') # normal corenlp server through internet
	#nlp = StanfordCoreNLP('http://localhost:9000')
	#nlp = StanfordCoreNLP('http://10.2.6.65:9099/') ## Desktop
	#nlp = StanfordCoreNLP('http://10.4.16.160:9094/') ## Lab server

	print("ner qc : ",qc,query)
	query_tokens = word_tokenize(query)


	if(qc[0] == 'LOCATION'):
		qc.append("CITY")
		qc.append("COUNTRY")
		qc.append("STATE_OR_PROVINCE")

		###### For Spacy #########
		qc.append("LOC")
		qc.append("GPE")
		#qc.append("NORP")
		qc.append("FAC")
		qc.append("ORG") ### bcz we have very less samples on ORGA in QC part so

	if(qc[0] == 'ORGANIZATION'):
		###### For Spacy #########
		qc.append("FAC")
		qc.append("ORG") ### bcz we have very less samples on ORGA in QC part so

	if(qc[0] == 'NUMBER'):
		##### For Spacy ##########
		qc.append("PERCENT")
		qc.append("ORDINAL")
		qc.append("CARDINAL")

	if(qc[0] == 'DATE'):
		qc.append("DURATION")
		qc.append('TIME')
		qc.append("SET")

	##### If TIME & DATE COMBINED THEN NO NEED TO WRITE BELLOW 5LINES SNIPPET
	if(qc[0] == 'TIME'):
		qc.append("DURATION")
		qc.append('DATE')
		qc.append("SET")

	print("In NER Section QC : ",qc)
	###################### Only Spacy NER Tool for Query ###############################
	doc = Ner_script_spacy.NER_Spacy_funct(quest[0].decode('utf8'))
	for entity in doc.ents:
		#print(entity.text,entity.label_)
		q_ner.append(entity.label_)
		Whole_output.append(entity.label_+"\n") #### Does this Spacy has Any timeout Error...??


	print("\n Fiding Same type NERs in Top 10 Sentences:\n")
	for ii in range(len(Top_similarity_sent)):
		text1=Top_similarity_sent[ii]
		text=unicodedata.normalize('NFKD',text1).encode('ascii','ignore')
		#text = text1
		print("text: ",text)
		nerr=[]
		o_text=[]
		new_ner_o_text=[]
		chunk_ner=[]

		################# ONly Spacy NER Tool for Sentences #####################
		doc = Ner_script_spacy.NER_Spacy_funct(text.decode('utf8'))		
		for entity in doc.ents:
			nerr.append(entity.label_) ####### NEED TO MODIFY HERE FOR CHUNKING Pbm Bcz we need Sequence
			o_text.append(entity.text)
			if(entity.label_ !="O"):
				random_ans.append(entity.text)

		#print("\n############# NER Preprocessing ###################\n")
		print("\n")
		for b in range(len(o_text)):
			new_ner_o_text.append(o_text[b])

		for k in range(len(qc)):
			for z in range(len(new_ner_o_text)):
				if(new_ner_o_text[z]!="NULL"):
					if(new_ner_o_text[z]!="."):
						if(nerr[z]==qc[k]):  # v.v imp condition
							chunk_ner.append(new_ner_o_text[z])
							final_answers.append(new_ner_o_text[z])

	if(len(random_ans)==0):
		print("random_ans: ",random_ans)
		print("NEED Figure out some Solution When NER is not recognizing any word...!!!!!!!!!!")
		stop_words = set(stopwords.words('english')) 
		Tokens = []
		for sent in Top_similarity_sent: ### Here also we can take some sentences from K-ranked sentences
			Tokens.append(word_tokenize(sent))

		Tokens = list(itertools.chain(*Tokens))	
		filtered_sent_tokens = [w for w in Tokens if not w in stop_words] 

		d = Counter(filtered_sent_tokens)
		words = [pair[0] for pair in sorted(d.items(), key=lambda item: item[1],reverse=True)]
		print(words)

		'''
		sent = Top_similarity_sent[0]
		sent_tokens = word_tokenize(sent)
		for tk in sent_tokens:
			random_ans.append(tk)
		'''
		print("words len : ",len(words))
		limit = 10
		if(len(words)<limit):
			limit = len(words)

		for ra_i in range(limit): ##### Taking 10 random answers from top K-ranked sentences by ignoring the stopwords.
			random_ans.append(words[ra_i])			

	max_final_ans=[]
	for i in range(len(final_answers)):
		max_final_ans.append(final_answers.count(final_answers[i]))
	ss=sorted(range(len(max_final_ans)),key=max_final_ans.__getitem__,reverse=True)

	print("\n")
	print("Answers Set : ",random_ans)
	print("\n\nAfter Chunking of NERs:\n")
	print("final_answers : ",final_answers)
	print("max_final_ans : ",max_final_ans)
	print("lenghts of [AnswersSet, final_answers, max_final_ans,ss] : ",len(random_ans),len(final_answers),len(max_final_ans),len(ss))
	optional_print=[]
	print("query_tokens : ",query_tokens)
	print("\n------------- > > Output < < --------------\n")
	#p_n=10	
	if(len(ss)>=10):
		p_n=len(ss)
	else:
		p_n=len(ss)
	z=0
	for i in range(len(ss)):
		j=ss[i]
		x=[]
		x.append(final_answers[j])
		if((set(optional_print).intersection(x))):
			print(" ")			
		else:
			if not(final_answers[j] in query_tokens):
				#print("else ans not in Q : ",final_answers[j])
				optional_print.append(final_answers[j])
				label_ans.append(final_answers[j])			
			
			z=z+1
	if(z==0 and (len(final_answers)!=0)):		
		print(random.choice(final_answers))

	print("Possible Predictable label_answers : ",label_ans)
	
	
	
def writing(qa):

	print("Answer Generation Phase....\n")
	for i in range(len(label_ans)):
		print(str(i+1)+" : "+label_ans[i])
		print("\t")
	
	print("\n----------------------------Best Answer Displaying in Telugu ---------------------------------\n\n")
	if(len(label_ans)!=0):
		pred_ans = translate(label_ans[0],'te')
		print(translate(label_ans[0],'te'))

	elif(len(label_ans)==0):
		print("\tRandom Answer Because Here answer is not getting from Top Ranked Sentences:\n")
		pred_ans = translate("OOV "+random.choice(random_ans),'te')
		print(translate(random.choice(random_ans),'te'))
		
	print("\n-------------------------------------------------------------\n\n")

	t = open("output_files/Result_Q_10Ans_Eng.txt","a")
	tt = open("output_files/Result_Q_10Ans_Tel.txt","a")
	rb=open("output_files/Q_Possible_answers.txt","a")

	rb.write(qa+"@@")
	t.write(qa+"@@")
	tt.write(qa+"@@")

	for i in range(len(label_ans)):
		rb.write((label_ans[i])+"$$$")
		t.write((label_ans[i])+"$$$")
		tt.write(translate(label_ans[i],"te")+"$$$")
		if(len(label_ans)==0): ### Generating random 10 answers Here We can take one answer
			rb.write("OOVA"+random.choice(random_ans)+"$$$")
			t.write("OOVA"+random.choice(random_ans)+"$$$")
			tt.write(translate("OOVA"+random.choice(random_ans),'te')+"$$$")
			#break

	rb.write("\n")
	rb.close()

	t.write("\n")
	t.close()

	tt.write("\n")
	tt.close()

	return pred_ans


def NER_Other_way(qc):
	print("qc : ",qc)
	stanford_ner_tagger = StanfordNERTagger(
    '/DATA1/USERS/priyanka/Natural_Language_Processing/Open_domain_QA/NER/stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
    '/DATA1/USERS/priyanka/Natural_Language_Processing/Open_domain_QA/NER/stanford_ner/' + 'stanford-ner-3.9.2.jar')

	Possible_answers = []
	All_answers      = []

	for i in range(len(Top_similarity_sent)):
		article = Top_similarity_sent[i]
		#print("Top Sent -- > ",article)
		results = stanford_ner_tagger.tag(article.split())
		print('Original Sentence: %s' % (article))
		for result in results:
			tag_value = result[0]
			tag_type = result[1]
		if(tag_type != 'O'):
			print('Type: %s, Value: %s' % (tag_type, tag_value))
			print("\n\n")
			All_answers.append(tag_value)
			if(tag_type == qc[0]):
				Possible_answers.append(tag_value)

	print("Possible_answers : ",Possible_answers)
	print("\n\n")
	print("All_answers : ",All_answers)
	print("\n\n")


def Question_Classification(user_input):
	qa=open("counter_vectors.txt","a")
	qa.write("    :"+user_input+"\n")
	qa.close()
	f=open("counter_vectors.txt","r")
	classes=[]
	queries=[]
	new_label=[]
	for line in f:
		line=line.rstrip('\n')
		classes.append((line.split()[0]).split(":")[0])
		lb=(line.split()[0]).split(":")[0]
		#print(lb)
		if(lb=="PERS"):
			new_label.append(1)
		if(lb=="LOCA"):
			new_label.append(2)
		if(lb=="ORGA"):
			new_label.append(3)
		if(lb=="DATE"):
			new_label.append(4)
		if(lb=="TIME"):
			new_label.append(5)
		if(lb=="PERC"):
			new_label.append(6)
		if(lb=="NUMB"):
			new_label.append(7)
		queries.append(line[5:])

	vectorizer =TfidfVectorizer(min_df=1)
	#vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(classes)
	Y = vectorizer.fit_transform(queries)

	X_ = Y[:len(new_label)]
	Y_ = new_label[:len(new_label)]
	X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.33, random_state=42)

	test_Labels=new_label[:339]
	test_Features=Y[:339].toarray()

	train_Labels=new_label[339:(len(new_label))]
	train_Features=Y[339:(len(new_label))].toarray()

	test=Y[len(queries)-1].toarray()

	print("\nTraining Started for Question Classification........\n")
	#nn=MLPClassifier(activation='tanh',solver='sgd',hidden_layer_sizes=(100,80,50,8),random_state=4,alpha=0.1,batch_size=71)
	#nn = LinearSVC(random_state=0, tol=1e-5)
	nn = LinearSVC(random_state=0, tol=0.3,loss="squared_hinge",multi_class="crammer_singer")

	print(nn)
	print("--------------------------------------\n\n")
	nn.fit(X_train,y_train)
	pred2=nn.predict(X_test)

	hits=0.00
	for i in range(0,len(y_test)):
		if y_test[i]==pred2[i]:
			hits=hits+1
			
	print("\nNumber of hits = ",hits)
	print("\n")
	print("The accuracy is ",((hits/len(y_test))*100.0))
	print("\n###################################################\n\n\n")

	result=nn.predict(test) 	
	if(result==1):
		ans_type.append("PERSON")

	elif(result==2):
		ans_type.append("LOCATION")

	elif(result==3):
		ans_type.append("ORGANIZATION")

	elif(result==4):
		ans_type.append("DATE")

	elif(result==5):
		ans_type.append("TIME")

	elif(result==6):
		ans_type.append("PERCENTAGE")

	else:
		ans_type.append("NUMBER")

	fp=open("counter_vectors.txt","rb")
	lines = fp.readlines()
	lines = lines[:-1]
	fp.close()
	Target_names = ['PERSON','LOCATION','ORGANIZATION','DATE','TIME','PERCENTAGE','NUMBER']
	fs=open("counter_vectors.txt","w")
	for i in lines:
		fs.write(i)
	fs.close()

	expected=y_test
	predicted=pred2
	print(metrics.classification_report(expected, predicted))
	#print(metrics.confusion_matrix(expected, predicted))
	#print("\n..............................................................\n")
	return(ans_type)




def cleanup_funct():
	Top_10_urls=[]
	whole_sent=[]
	consine_list=[]
	all_sen=[]
	Top_similarity_sent=[]
	ans=[]
	Sentence_NER=[]
	ner=[]
	quest=[]
	q_ner=[]
	q_text=[]
	Whole_output=[]
	final_answers=[]
	ss=[]
	label_ans=[]
	ans_type=[]
	all_sen1=[]
	random_ans=[]
	return (Top_10_urls,whole_sent,consine_list,all_sen,Top_similarity_sent,ans,
			Sentence_NER,ner,quest,q_ner,q_text,Whole_output,
			final_answers,label_ans,ss,ans_type,all_sen1, random_ans)


############################################# Calling QA System Step by Step #########################################################
out = open("output_files/Accuracy.txt","w")
out_m = open("output_files/Accuracy_of_All_Queries.txt","a")
input_file = open("Category_wise_Questions/location.txt","r").readlines()
Questions = []

for line in input_file:
	line=line.strip()
	Questions.append(line.strip())


for n,qa in enumerate(Questions):
	(Top_10_urls,whole_sent,consine_list,all_sen,Top_similarity_sent,ans,Sentence_NER,ner,quest,q_ner,q_text,Whole_output,final_answers,
    label_ans,ss,ans_type,all_sen1,random_ans ) = cleanup_funct()

	answer=Question_Type(qa) # calling to get Rule Based Query Type
	print("\n---------------------------------------------------------\n\n")
	print("Query Type:\n")
	print(answer)
	q=translate(qa,'en') # translate te-->eng
	print("\n--------------------------------------------------------\n\n")
	print("Telugu Sentence:\t"+qa+"\n")
	print("English Sentence:\t"+q+"\n")
	q_keywords=q
	tt=q_keywords.split()
	print("\n\n")
	print("Top 10 URLS:\n----------------------------------------------------------------\n\n")
	Top_10_urls = qquery(q) # calling get Top 10 URLS
	print("---------------------------------------------------------\n\n")
	Sentence_Extraction()  # calling to get whole content from 10 urls(well define sentences only)
	Finding_Similarity_Matrix(q) # calling find similarity matri between query and all sentences
	Top_similarity_sent = Most_Relative_Sentence() # calling to print higest top 5 similarity senteces.
	print("Top_similarity_sent : ",Top_similarity_sent)
	print("\n---------------------------------------------------------\n\n")
	qc=Question_Classification(qa) # user query question classification
	print("\n Question Classification Prediction for User Query:\n")
	print('qc :',qc[0])

	################## Rule based Classifier and NN classifer #########################
	if(len(answer)!=0):
		for cl in answer:		
			qc.append(cl)
	print("\n---------------------------------------------------------\n\n")
	print("before NER QC : ",qc)
	get_NER(qc,q) # calling to get NERS
	result = writing(qa)
	print("predicted answer : ",result)
	out.write(str(list(set(qc)))+"::"+str(result)+"::"+qa+"\n")
	out_m.write(str(list(set(qc)))+"::"+str(result)+"::"+qa+"\n")

	print("qa: ",qa)
	with open("Category_wise_Questions/location.txt", "r") as f:
		lines = f.readlines()
	with open("Category_wise_Questions/location.txt", "w") as f:
		for line in lines:
		    if line.strip("\n") != qa.strip("\n"):
		        f.write(line)
	f.close()

out_m.close()
out.close()

