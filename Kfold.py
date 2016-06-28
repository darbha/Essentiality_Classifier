import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import scipy
from matplotlib.backends.backend_pdf import PdfPages
from collections import OrderedDict
import operator
import itertools

#Read in properties and essentiality files
prop_dataset = genfromtxt(open('protein_props_cerevisiae.csv','r'))
ess_dataset = genfromtxt(open('essentiality_cerevisiae.csv','r'))

#Define number of folds and random forest classifier parameters
cross_val = StratifiedKFold(ess_dataset, n_folds=10)
rf = RandomForestClassifier(n_estimators=1000, min_samples_split=2)

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

#Train and test the classifier
for i, (train, test) in enumerate(cross_val):
    probas_ = rf.fit(prop_dataset[train], ess_dataset[train]).predict_proba(prop_dataset[test])
# Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(ess_dataset[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    #plt.plot(fpr, tpr, lw=1, label='AUC-ROC %d (area = %0.2f)' % (i, roc_auc))

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.0, 0.4), label='Random')

mean_tpr /= len(cross_val)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

pp = PdfPages('KFold_pombe.pdf')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
plt.savefig(pp, format='pdf')