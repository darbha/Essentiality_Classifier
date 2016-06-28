from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import scipy
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import operator
import itertools
import csv
import numpy as np


###Here we compare our classifier's performance with that of Seringhaus###

def main():

#########Calculating AUC for our classifier first - train on cere, test on pombe###

#read in the training and target files
	train = genfromtxt(open('filtered_properties_RF_cerevisiae.csv','r'))
	train_ess = genfromtxt(open('filtered_ess_RF_cerevisiae.csv','r'))
	#print train.shape, train_ess.shape
	

#read in the test files
	realtest = genfromtxt(open('filtered_props_RF_pombe.csv','r'))
	realtest_ess = genfromtxt(open('filtered_ess_RF_pombe.csv','r'))
	#print realtest.shape, realtest.shape
	
###random forest code
	rf = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=1)
	rf.fit(train, train_ess)
	

######Save output - classes 1 or 0#################
	
	savetxt('predicted_output_cereonpombe', rf.predict(realtest), delimiter = ',', fmt = '%f')
	

######Calculate AUC###############

	probas = rf.predict_proba(realtest)
	fpr, tpr, thresholds = metrics.roc_curve(realtest_ess, probas[:, 1])
	ROC = metrics.auc(fpr, tpr)
	print "\nAUC for our classifier tested on pombe = ", ROC, "\n"
	
	
####Calculating AUC for Seringhaus classifier that was trained on cerevisiae and tested on pombe######

	expt_ess = genfromtxt(open('filtered_expt_ess_pombe.csv','r'))
	pred_ess = genfromtxt(open('filtered_seringhaus_pred_ess_pombe.csv', 'r'))
	fpr1, tpr1, thresholds1 = metrics.roc_curve(expt_ess, pred_ess)
	ROC1 = metrics.auc(fpr1, tpr1)
	print "\nAUC for Seringhaus et al classifier tested on pombe = ", ROC1, "\n"		

	
#####Create and save ROC curve####################	
	
	ppombe = PdfPages('seringhaus_pombe_plot_comparison.pdf')
	plt.figure(figsize=(10,8))
	
	plt.plot(fpr,tpr,label="AUC of our classifier= %.3f" % ROC)
	plt.plot(fpr1,tpr1,label="AUC of Seringhaus et al's classifier= %.3f" % ROC1)
	plt.plot([0, 1], [0, 1], 'k--',label="Random AUC = %s" % 0.5)
	#plt.title('ROC curve for Classifier trained on S. cerevisiae and tested on S. pombe')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.show()
	#plt.savefig(ppombe, format='pdf')
	#ppombe.close()
	
############

if __name__ == "__main__":
	main()
