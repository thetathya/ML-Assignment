# import os
# import math
# import numpy as np
# from collections import Counter

# # make prior probabilities dynamic

# sizett = 463
# x = os.listdir("assignment3_train\\train")
# spamwc={}
# hamwc = {}
# for i in x:
# 	y = os.listdir("assignment3_train\\train\\" + i)
# 	if i=="spam":
# 		for j in y:		
# 			f = "assignment3_train\\train\\" + i + "\\" + j
# 			file=open(f,"r", errors = 'ignore')
# 			for word in file.read().split():
# 				if word not in spamwc:
# 					spamwc[word] = 1
# 				else:
# 					spamwc[word] += 1
# 	else:
# 		for j in y:		
# 			f = "assignment3_train\\train\\" + i + "\\" + j
# 			file=open(f,"r", errors = 'ignore')
# 			for word in file.read().split():
# 				if word not in hamwc:
# 					hamwc[word] = 1
# 				else:
# 					hamwc[word] += 1

# totalwc = Counter(hamwc) + Counter(spamwc) 
# totalw_s = sum(spamwc.values())
# totalw_h = sum(hamwc.values())
# novoc = len(totalwc)
# cs = 0
# ch = 0
# cst = 0
# cht = 0
# # Naive Bayes
# for i in x:
	
# 	y = os.listdir("assignment3_test\\test\\" + i)
# 	for j in y:
# 		test_sh = {}
# 		f = "assignment3_test\\test\\" + i + "\\" + j
# 		file=open(f,"r", errors = 'ignore')
# 		for word in file.read().split():
# 			if word not in test_sh:
# 				test_sh[word] = 1
# 			else:
# 				test_sh[word] += 1
# 		prob_s = math.log(123/sizett)
# 		prob_h = math.log(340/sizett)
# 		# print(prob_s, prob_h)
# 		for k in test_sh:
# 			if spamwc.get(k) != None:
# 				prob_s = prob_s + math.log((spamwc.get(k)+1)/((totalw_s)+(novoc)))
# 			else:
# 				prob_s = prob_s + math.log((1)/((totalw_s)+(novoc)))
# 			if hamwc.get(k) != None:
# 				prob_h = prob_h + math.log((hamwc.get(k)+1)/((totalw_h)+(novoc)))
# 			else:
# 				prob_h = prob_h + math.log((1)/((totalw_h)+(novoc)))

# 		if prob_s > prob_h:
# 			cs = cs + 1
# 			if i=="spam":
# 				cst = cst + 1
# 		elif prob_h > prob_s:
# 			ch = ch + 1
# 			if i=="ham":
# 				cht = cht + 1

# print(cst/cs,cht/ch)


# # Logistic Regression
# ltotalwc = list(totalwc.keys())
# mat = np.zeros((sizett+1,len(ltotalwc)+1))
# ind = 0
# calc = 0
# for i in x:
# 	y = os.listdir("assignment3_train\\train\\" + i)
# 	print(ind)
# 	for j in y:
# 		logwc = {}
# 		f = "assignment3_train\\train\\" + i + "\\" + j
# 		file=open(f,"r", errors = 'ignore')
# 		for word in file.read().split():
# 			if word not in logwc:
# 				logwc[word] = 1
# 			else:
# 				logwc[word] += 1
# 		for k in logwc:
# 			mat[ind][ltotalwc.index(k)] = logwc[k]
# 		if i=="spam":
# 			# print(i)
# 			calc = calc + 1
# 			mat[ind][len(ltotalwc)] = 1
# 		ind = ind + 1
# print(calc)
# print(mat[341])










import os
import math
import numpy as np
from collections import Counter
import time

sizett = 463
x = os.listdir("assignment3_train\\train")
spamwc={}
hamwc = {}
totalwc = {}
for i in x:
    y = os.listdir("assignment3_train\\train\\" + i)
    if i=="spam":
        for j in y:
            f = "assignment3_train\\train\\" + i + "\\" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in spamwc and word.isalpha():
                    spamwc[word] = 1
                    totalwc[word] = 1
                elif word.isalpha():
                    spamwc[word] += 1
                    totalwc[word] += 1
    else:
        for j in y:
            f = "assignment3_train\\train\\" + i + "\\" + j
            file=open(f,"r", errors = 'ignore')
            for word in file.read().split():
                if word not in hamwc and word.isalpha():
                    hamwc[word] = 1
                    totalwc[word] = 1
                elif word.isalpha():
                    hamwc[word] += 1
                    totalwc[word] += 1

# Naive Bayes
totalw_s = sum(spamwc.values())
totalw_h = sum(hamwc.values())
novoc = len(totalwc)
cs = 0
ch = 0
cst = 0
cht = 0
for i in x:
    y = os.listdir("assignment3_test\\test\\" + i)
    for j in y:
        test_sh = {}
        f = "assignment3_test\\test\\" + i + "\\" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in test_sh and word.isalpha():
                test_sh[word] = 1
            elif word.isalpha():
                test_sh[word] += 1
        prob_s = math.log(123/sizett)
        prob_h = math.log(340/sizett)
        # print(prob_s, prob_h)
        for k in test_sh:
            if spamwc.get(k) != None:
                prob_s = prob_s + math.log((spamwc.get(k)+1)/((totalw_s)+(novoc)))
            else:
                prob_s = prob_s + math.log((1)/((totalw_s)+(novoc)))
            if hamwc.get(k) != None:
                prob_h = prob_h + math.log((hamwc.get(k)+1)/((totalw_h)+(novoc)))
            else:
                prob_h = prob_h + math.log((1)/((totalw_h)+(novoc)))

            if prob_s > prob_h:
                cs = cs + 1
                if i=="spam":
                    cst = cst + 1
            elif prob_h > prob_s:
                ch = ch + 1
                if i=="ham":
                    cht = cht + 1

print(cst/cs,cht/ch,(cst+cht)/(cs+ch))








# Logistic regression


# Creates the matrix of where words are column and files are row
ltotalwc = list(totalwc.keys())
mat = np.zeros((sizett,len(ltotalwc)+1))
ind = 0
for i in x:
    y = os.listdir("assignment3_train\\train\\" + i)
    for j in y:
        logwc = {}
        f = "assignment3_train\\train\\" + i + "\\" + j
        file=open(f,"r", errors = 'ignore')
        for word in file.read().split():
            if word not in logwc and word.isalpha():
                logwc[word] = 1
            elif word.isalpha():
                logwc[word] += 1
        for k in logwc:
            mat[ind][ltotalwc.index(k)] = logwc[k]
        if i=="spam":
            mat[ind][len(ltotalwc)] = 1
        ind = ind + 1


# To calculate probability that is required to count weights, here i am skipping some values greater than 710 to avoid overfow error.
def prob(w,x):
    s = 0
    for i in range(len(x)):
        s = s + (w[i]*x[i])
    try:
        p = math.exp(w[0]+s)/(1 + math.exp(w[0]+s))
    except:
        p = 1
    return p


w = np.ones(len(totalwc)+1)
probab = np.ones(mat.shape[0])
print(mat.shape[0])
for i in range(mat.shape[0]):
	probab[i] = prob(w,mat[i])

print(probab)