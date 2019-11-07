# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import spacy
nlpp = spacy.load('en_core_web_sm')

#print("hello world")
def NER_Spacy_funct(text_sent):
	doc = nlpp(text_sent)
	#print("doc: ",doc)
	print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Calling NER Spacy Function @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
	for entity in doc.ents:
		print("spacy ner output: ",entity.text,entity.label_)
	print("\n-------------------------------\n")
	return doc



