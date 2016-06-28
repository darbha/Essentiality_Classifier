import csv
#import csv module. implements classes to read and write tabular data in CSV format

import os
import re
import sys
from array import *
import collections


ess_count = 0
noness_count = 0
no_info = 0
with open('c_albicans_phenotypes.csv', 'rb') as f, open('edited_c_albicans_phenotypes.csv', 'wb') as f_out:
	writer = csv.writer(f_out)
	for row in f:
		row = row.split(',')
		if 'inviable' in row:
			Inv = [row[0], row[6], 'inviable']
			writer.writerow(Inv)
			ess_count+=1
 		elif 'viable' in row:
 			Vi = [row[0], row[6], 'viable']
			writer.writerow(Vi) 
 			noness_count+=1
 		else:
 			no_info+=1	
				
 	print "Essential genes = ", ess_count, "Non essential genes = ", noness_count, "No essentiality info = ", no_info
 	
 
syst = open('syst_mut.csv', 'w')
clas = open('classical_genetics.csv', 'w')
survey = open('large_scale.csv', 'w')						
total_count = 0
systematic_count = 0
large_scale_survey_count = 0
classical_genetics_count = 0
with open('edited_c_albicans_phenotypes.csv', 'rb') as f: 
	writer1 = csv.writer(syst)
	writer2 = csv.writer(clas)
	writer3 = csv.writer(survey)
	for row in csv.reader(f, delimiter=',', skipinitialspace=True):
		row = [string.strip(' ') for string in row]
		total_count+=1
		method = row[1]
		if method.startswith("systematic mutation"):
			#syst.write(str(row)+'\n')
			writer1.writerow(row)
			systematic_count+=1
		elif method.startswith("large-scale survey"):
			#survey.write(str(row)+'\n')
			writer3.writerow(row)
			large_scale_survey_count+=1
		elif method.startswith("classical genetics"):
			#clas.write(str(row)+'\n')
			writer2.writerow(row)
			classical_genetics_count+=1	
		else:
			print row		
	print "Total number of genes with known essentiality", total_count	
	print "Number of identifications by systematic mutation:", systematic_count	
	print "Number of identifications by large scale survey:", large_scale_survey_count
	print "Number of identifications by classical genetics:", classical_genetics_count 
	print systematic_count + large_scale_survey_count + classical_genetics_count	
	
syst.close()
clas.close()
survey.close()	
	
################## GETTING UNIQUE GENES ############		
######counting unique identifications by classical genetics method###
 
print "\n"
 
unique_entries1 = []	
countc = 0
 
with open('classical_genetics.csv', 'rb') as f, open('unique_classical_genetics.csv', 'w') as f_out: 
	writer = csv.writer(f_out)
	for row in csv.reader(f, delimiter=',', skipinitialspace=True): 
		if row[0] not in unique_entries1:
			unique_entries1.append(row[0])
			writer.writerow([row[0], row[2]])
			#print row
			countc+=1
 			
print "Number of UNIQUE identifications by classical genetics:", countc 

######counting unique identifications by systematic mutations####
 	
unique_entries2 = []	
counts = 0
 
with open('syst_mut.csv', 'rb') as f, open('unique_syst_mut.csv', 'w') as f_out: 
	writer = csv.writer(f_out)
	for row in csv.reader(f, delimiter=',', skipinitialspace=True): 
		if row[0] not in unique_entries2:
			unique_entries2.append(row[0])
 			writer.writerow([row[0], row[2]])
 			counts+=1
 		 			
print "Number of UNIQUE identifications by systematic mutations:", counts	

######counting unique identifications by large-scale survey####	
	
unique_entries3 = []	
countl = 0
 
with open('large_scale.csv', 'rb') as f, open('unique_large_scale.csv', 'w') as f_out:
	writer = csv.writer(f_out)
	for row in csv.reader(f, delimiter=',', skipinitialspace=True): 
		if row[0] not in unique_entries3:
			unique_entries3.append(row[0])
 			writer.writerow([row[0], row[2]])
 			countl+=1
 			
print "Number of UNIQUE identifications by large scale survey:", countl	

phenotypes = {}	

phenotype_files = ['unique_classical_genetics.csv','unique_syst_mut.csv','unique_large_scale.csv']

for filename in phenotype_files :
	with open(filename, 'r') as ucg:
		#For each phenotype file we store the essential and non-essential genes as a set
		#these sets are stored in a dictionary with the filename + ess or non_ess as the key
		shortname = filename.split(".")[0]
		ess_key = "%s_ess" % shortname
		non_ess_key = "%s_non_ess" % shortname
		phenotypes[ess_key] = set()
		phenotypes[non_ess_key] = set()
		for each_entry in csv.reader(ucg):
			if each_entry[1] == "inviable":
				phenotypes[ess_key].add(each_entry[0])
			else :
				phenotypes[non_ess_key].add(each_entry[0])	

for i in phenotypes :
	#i="CG+ess"
	for j in phenotypes :
		print i, j, len(phenotypes[i]),len(phenotypes[j]),len(phenotypes[i].intersection(phenotypes[j]))