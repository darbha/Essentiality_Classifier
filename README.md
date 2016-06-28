#Essentiality Classifier 
####(Classification of essential genes)


A Random Forests (RF) classifier is built to classify (yeast) genes as essential or nonessential, using only protein sequence properties. Input to train the classifier is protein properties and available essentiality information (experimental). >45 features used in studied. Only a few scripts discussed here.



####Some relevant scripts written to prepare datasets and train a RF classifier:


#####1. processing_cerevisiae_properties.py

  * The script processes and retains relevant protein properties (e.g. codon adaptation index, frequency of amino acids), and essentiality information (binary; 1 = essential and 0 = nonessential) downloaded from yeast genome database and OGEEDB, respectively. 

  * The input is a file with gene names and several associated annotations (properties), and another file with genes names and essentiality status along with other information. Irrelevant data is filtered out, and genes with properties and associated essentiality information is provided as output.


#####2. blast_paralog_detection.py

  * Apart from downloaded properties, other features of genes were added too. These were computed by me. An example is paralog detection. Paralogs are duplicate genes within a genome. 
  * The script filters all-against-all BLASTp results (input) as per method described by Gu et al 2002.
  * The output is count of paralogs per gene.


#####3. RBH.py

  * Another calculated property was the presence or absence of an ortholog in other selected organisms.
  * The script takes in 2 BLASTp results and returns best reciprocal hits.


#####4. disordered_proteins_cerevisiae.py

  * Maps data across files based on gene name and appends proteindisorder information to each gene.


#####5. Kfold.py

  * Trains and tests an RF classifier with 10-fold cross validation. The classifier is trained and tested within a single dataset (species) here.
  * Input of protein properties (downloaded, calculated) and essentiality info is provided. An AUC-ROC curve is generated.


#####6. parameter_search_cerevisiae.py

  * Various RF parameters are tested to detect the best combination of parameters for the study. 
  * Best number of estimators (trees), minimum split, criterion, maximum features are sought.
  * Classifier is trained and tested on single dataset. This script also calculates ROC curves to test performance and feature importance.


#####7. comparing_ourROC_with_seringhausROC.py

  * Script compares performance between 2 classifiers (RF and naive bayes of another study). 
  * RF classifier is trained on one dataset (*S. cerevisiae*) and tested on another (*S. pombe*).
  * ROC curve is generated for my classifier, and for another classifier, which was one of the first to do attempt predicting essentiality with machine learning.


#####8. statistical_tests.py

  * some statistical tests done during the study (Mann-Whitney, t-test, spearman correlation, z-score normalization)
