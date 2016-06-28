import os
import csv
from math import exp
from collections import Counter
from collections import defaultdict
#here we are filtering the blast output as per Gu et al 2002 to retain paralogs.

protein_lengths = {}

with open('reordered_nidulans_properties_header.csv', 'r') as f:
	reader = csv.DictReader(f)
	for each_entry in reader:
		protein_lengths[each_entry['ORF']] = float(each_entry['Protein Length'])
    
        
# Below we filter paralogs as per Gu et al (2002)

n = 6   
paralog_count = defaultdict(int)
with open('allvsallnidulans.tsv', 'r') as blastoutput:
	reader2 = csv.reader(blastoutput, delimiter = "\t")
	
	for line in reader2:	
		if line[0] != line[1]:
			if line[0] in protein_lengths and line[1] in protein_lengths:
				length1 = protein_lengths[line[0]]
				length2 = protein_lengths[line[1]]
				max_length = max(length1, length2)
				if float(line[3]) >= 0.8 * max_length:
					L = float(line[3])
					identity = float(line[2])
					if L > 150:
						if identity >= 30:
							paralog_count[line[0]] += 1
					elif L <= 150:
						if identity >= n + 480 * L - 0.32 * (1 + exp(-L/1000)):
							paralog_count[line[0]] += 1
			
########### Counting paralogs ###########

paralogs = open('Anidulans_paralogs_colm.csv', 'w')		

writer3 = csv.writer(paralogs)
for p in paralog_count :
	writer3.writerow((p,paralog_count[p]))
   
paralogs.close()