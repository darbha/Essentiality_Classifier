#from sklearn import preprocessing
#prop_data = np.loadtxt('/Users/jigishadarbha/Desktop/classifier/workingscripts/filtered_properties.csv')
#prop_target = np.loadtxt('/Users/jigishadarbha/Desktop/classifier/workingscripts/filtered_ess.csv') 
#print prop_data[0:5]


from sklearn.ensemble import RandomForestClassifier
import numpy as np
from numpy import genfromtxt, savetxt
import scipy
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import operator
import itertools
from collections import defaultdict

def main():

######read in training files################
	
	prop_dataset = genfromtxt(open('/Users/jigishadarbha/Desktop/classifier/workingscripts/filtered_properties_core_final.csv','r'))
	ess_dataset = genfromtxt(open('/Users/jigishadarbha/Desktop/classifier/workingscripts/filtered_ess_core_final.csv','r'))
	
		
######Calculate AUC###############
	criteria = ['gini','entropy']
	n_estimators = [10,100,1000]
	max_features = ['sqrt','log2']
	min_splits = [2,10,50]
	scores = defaultdict(list)
	for run in range(0,10) :
		print "Run", run
		prop_train, prop_test, ess_train, ess_test = train_test_split(prop_dataset, ess_dataset, test_size=.3)
		for s in min_splits :
			for c in criteria :
				for n in n_estimators :
					for m in max_features :
						rf = RandomForestClassifier(n_estimators=n, criterion=c, max_features=m,min_samples_split=s)
						rf.fit(prop_train, ess_train)
						probas = rf.predict_proba(prop_test)
						fpr, tpr, thresholds = metrics.roc_curve(ess_test, probas[:, 1])
						ROC = metrics.auc(fpr, tpr)
						scores[(c,n,m,s)].append(ROC)
						print run,c,n,m,s,ROC
	print "AVERAGE AUCs from 10 Runs"
	for c in scores :
		print "%s\t%s\t%s\t%s\t%s\t%s" % (c[0],c[1],c[2],c[3],np.mean(scores[c]),np.std(scores[c]))
#######Create and save ROC curve####################	
	
	pp = PdfPages('RF_Plots_cerevisiae_core_final.pdf')
	
	plt.plot(fpr,tpr,label="RandomForest AUC=%.3f" % ROC)
	plt.plot([0, 1], [0, 1], 'k--',label="Random = %s" % 0.5)
	plt.title('ROC curve for Classifier')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	#plt.show()
	plt.savefig(pp, format='pdf')
	
	
	
#######Compute feature importance#################
	
	feat_imp = rf.feature_importances_
	#print feat_imp
	

#######Convert feat_imp which is a numpy object to a list##############
	
	feat_imp_list = feat_imp.tolist()
	#print feat_imp_list
	
	
#######Read in names of features (separate file)#################	
	
	with open('Final_Features.txt', 'r') as h:
		feature_names = h.read()
		feature_names = feature_names.split("\t")
		#print feature_names
		

########Create a dictionary of features names and corresponding feature importance values#############
	
	name_to_number_dict = dict(zip(feature_names, feat_imp_list))
	#print name_to_number_dict

	
#########Sort and order the dictionary################
	
	sorted_dict = OrderedDict(sorted(name_to_number_dict.items(), key=lambda t:t[1], reverse = True))
	#print sorted_dict
	

########Print top 10 features with values##########################
	
	count = 0
	for x in sorted_dict:
		print("%d. %s (%f)" % (count + 1, x, sorted_dict[x]))
		count+=1
		if count == 10:
			break
	print "\n"
			
############Calculate standard deviation, sort features in descending order and store the top 10###############
	
	std = np.std([tree.feature_importances_ for tree in rf.estimators_],axis=0)
	indices = np.argsort(feat_imp)[::-1]
	indices = indices[0:10]
	
	
###########Plot the features - importance##############
	
	#print("Feature ranking:")
	#for f in range(10):
	#	print("%d. feature %d (%f)" % (f + 1, indices[f], feat_imp[indices[f]]))

	plt.figure()
	plt.title("Feature importances")
	plt.ylabel('RF score')
	plt.xlabel('Features')
	plt.bar(range(len(indices)), feat_imp[indices], color="r", yerr=std[indices], align="center")
	plt.xticks(range(10), np.asarray(feature_names)[indices], rotation = 45, fontsize = 12)
	plt.xlim([-1, 10])
	plt.savefig(pp, format='pdf')
	
	pp.close()


if __name__ == "__main__":
	main()
		
		


# n_estimators : integer, optional (default=10) - The number of trees in the forest.
# min_samples_split : integer, optional (default=2) - The minimum number of samples required to split an internal node. Note: this parameter is tree-specific.
# n_jobs : integer, optional (default=1) - The number of jobs to run in parallel for both fit and predict. 
#If -1, then the number of jobs is set to the number of cores.		
		
# Explaining if __name__ == "__main__":