import sys
import os
import csv
import pandas as pd
from scipy.stats import ttest_ind
from numpy import genfromtxt, savetxt
from scipy.special import stdtr
from sklearn.ensemble import RandomForestClassifier
import scipy
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import operator
import itertools
import scipy.stats




#####	Mann-Whitney	############


def main():

######read in training files################
	
	prop_dataset = genfromtxt(open('filtered_properties_core_zscore.csv','r'))
	ess_dataset = genfromtxt(open('filtered_ess_core_zscore.csv','r'))
	

######Split data into training and test sets######################
	
	prop_train, prop_test, ess_train, ess_test = train_test_split(prop_dataset, ess_dataset, test_size=.3)
	
######Random forest code#####################	
	
	rf = RandomForestClassifier(n_estimators=100, min_samples_split=2, n_jobs=1)
	rf.fit(prop_train, ess_train)
	
######Save output - classes 1 or 0#################
	
	savetxt('output_cerevisiae_core', rf.predict(prop_test), delimiter = ',', fmt = '%f')


######Calculate AUC###############

	probas = rf.predict_proba(prop_test)
	x=probas[ess_test==1,1]
	y=probas[ess_test==0,1]
	u,p = scipy.stats.mannwhitneyu(x,y)
	print "p-value = ", p

if __name__ == "__main__":
	main()
		

# n_estimators : integer, optional (default=10) - The number of trees in the forest.
# min_samples_split : integer, optional (default=2) - The minimum number of samples required to split an internal node. Note: this parameter is tree-specific.
# n_jobs : integer, optional (default=1) - The number of jobs to run in parallel for both fit and predict. 
#If -1, then the number of jobs is set to the number of cores.		
		



#####		Z-Score		#####

cer_data = pd.read_csv("cerevisiae_scores_final_header.csv")
headers = cer_data.columns.values
to_normalize = [ 'CAI','Disorder', 'Ala', 'Cys', 'Asp', 'Glu', 'Phe',
       'Gly', 'His', 'Ile', 'Lys', 'Leu', 'Met', 'Asn', 'Pro', 'Gln',
       'Arg', 'Ser', 'Thr', 'Val', 'Trp', 'Tyr','Aromaticity Score']
for property in to_normalize:
	cer_data[property] = (cer_data[property] - cer_data[property].mean())/cer_data[property].std()      

cer_data.to_csv("Zscore_cerevisiae_props.csv")

remove_from = 0
remove_to = 1

with open("Zscore_cerevisiae_props.csv", 'r') as fp_in, open("Zscore_cerevisiae_props_ready.csv", 'w') as fp_out:
	reader = csv.reader(fp_in)
	writer = csv.writer(fp_out)
	for row in reader:
		del row[remove_from:remove_to]
		writer.writerow(row)

		
	
		
#####	t-Test	######		


aro = genfromtxt(open('avg_freq_aroma.csv','r'))
nonaro = genfromtxt(open('avg_freq_nonaroma.csv','r'))

t, p = ttest_ind(aro, nonaro, equal_var=False)
print "ttest_ind: t = %f  p = %f" % (t, p)





#####	Spearman correlation	#####

cer_data = pd.read_csv("Complete_data_with_header.csv")
spearman = cer_data.corr(method = "spearman")
#print spearman

spearman.to_csv('spearman_correlation_for_cerevisiae.csv', sep = '\t')