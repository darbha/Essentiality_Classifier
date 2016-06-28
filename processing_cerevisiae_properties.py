import csv
import os
import sys
import re

with open('cerevisiae_properties.csv', 'r') as f, open('cerevisiae_corefeatures.csv', 'w') as f_out:
	reader = csv.reader(f)
	writer = csv.writer(f_out)
	for line in reader:
		indices = 4, 7, 8, 29, 30, 31, 32, 33, 34, 35
		edit_row = [i for j, i in enumerate(line) if j not in indices]
		#print edit_row
		writer.writerow(edit_row)
	

with open('myessentialcsv.csv', 'rb') as f, open('edited_myessential.csv', "wb") as f_out:
	essentiality_writer = csv.writer(f_out)
	for row in f:
			row = row.strip().split(',')
			indices = 0, 1, 2, 5, 6, 7, 8, 9
			edit_row = [i for j, i in enumerate(row) if j not in indices]
			#print edit_row
			essentiality_writer.writerow(edit_row)

#############Below, I am making a new file with protein ID and essentiality info##################

essentiality_status = {}	
with open('edited_myessential.csv') as edit_ess:
	for line in edit_ess:
		line = line.translate(None, '\r\n').split(',')
		essentiality_status[line[0]] = line[1]
		
ess_f = open('filtered_core_ess_cerevisiae.csv', 'w')
prop_f = open('filtered_core_properties_cerevisiae.csv', 'w')
id_f = open('filtered_core_ids_cerevisiae.csv', 'w')
missing_count = 0
with open('cerevisiae_corefeatures.csv') as edit_props:
	for line in edit_props:	
		line = line.replace('\r\n','')
		line = line.split(',')
		if line[0] in essentiality_status:
			essornot = essentiality_status[line[0]]
			if essornot == 'N':
				ess_f.write('0\n')
				prop_f.write(' '.join(line[1:])+'\n')
				id_f.write(line[0]+'\n')
			else:
				ess_f.write('1\n')
				prop_f.write(' '.join(line[1:])+'\n')
				id_f.write(line[0]+'\n')
		else:
			#print line[0] + ' or ' + line[1] +' are not in the list'
			missing_count += 1
			

ess_f.close()
prop_f.close()
id_f.close()
print missing_count, 'IDs from prop file are not in the essentiality file'

