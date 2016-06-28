import os, csv, re, sys

# uniprot_IDS = open('uniprot_ids_disorder.txt', 'w')

#####We are trying to get only the uniprot IDS here so that we can map to SCD IDs later##

with open('559292_disorder.tdt', 'r') as file:
 	for line in file:
 		line.strip()
 		line = line.translate(None, '\r\n').split('\t')
 		indices = 1,2,3,4,5,6
 		line = [i for j, i in enumerate(line) if j not in indices]
 		line = str(line).strip('[]')
 		line = line.replace("'","")
 		print line
 		uniprot_IDS.write(line+'\n')
 		
uniprot_IDS.close()		

########cerevisiae_disorder_mapped.csv contains uniprot IDs and respective ensembl protein IDs after mapping on uniprot####
###cerevisiae_disorder_mapped.csv and iza's csv file are stored as dictionaries###

with open('cerevisiae_disorder_mapped.csv', 'r') as map:
	map_dict = {row[0]: row[1] for row in list(csv.reader(map))}
	
with open('cerevisiae_disorder_iza.csv', 'r') as disorder:
	next(disorder, None)
	disorder_dict = {row[0]: row[1:] for row in list(csv.reader(disorder))}
	
for row in disorder_dict:
	if row in map_dict:
		disorder_dict[row].append(map_dict[row])
	else:
		disorder_dict[row].append(0)

merged = [(k,) + tuple(v) for k,v in disorder_dict.items()]

with open('cerevisiae_disorder_SGD.csv', 'w') as disorder_out:
	csv.writer(disorder_out).writerows(merged)
					
	
				

