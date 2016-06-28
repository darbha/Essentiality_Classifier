#here we are outputting bi-directional best blast hits - shared pairs between 2 blast output files. For example, in one round of blast, protein sequences 
# from S. cerevisiae are blasted against human protein database and for the next round of blast, human protein sequences are blasted against 
#cerevisiae database. We compare outputs from these blast runs to pick out only those that are each others' "best hits"



usage = """RBH blastoutput1 blastoutput2 RBHoutput"""

import sys, re

if len(sys.argv) < 3:
	print usage
	
debug = 9

#take the 2 blast output files and the output from this script as arguments 
blastoutput1 = sys.argv[1]
blastoutput2 = sys.argv[2]
outfl = sys.argv[3]	

#parse the first blast output file
file1 = open(blastoutput1, 'r')
D1 = {}
for line in file1:
	line.strip()
	elements = re.split('\t', line)
	queryID = elements[0]
	subjectID = elements[1]
	if (not(queryID in D1.keys())):
		D1[queryID] = subjectID     #pick the first hit
		
if debug: D1.keys()


#parse the second blast output file
file2 = open(blastoutput2, 'r')
D2 = {}
for line in file2:
	line.strip()
	elements = re.split('\t', line)
	queryID = elements[0]
	subjectID = elements[1]
	if (not(queryID in D2.keys())):
		D2[queryID] = subjectID     #pick the first hit
		
if debug: D2.keys()

#Now we pick shared pairs
sharedpairs = {}
for id1 in D1.keys():
	value1 = D1[id1]
	if value1 in D2.keys():
		if id1 == D2[value1]:	#a shared best reciprocal pair
			sharedpairs[value1] = id1

if debug: sharedpairs

#Create output file with a dictionary of shared pairs
outfile = open(outfl, 'w')
for k1 in sharedpairs.keys():
	line = k1 + '\t' + sharedpairs[k1] + '\n'
	outfile.write(line)
	
outfile.close()
file1.close()
file2.close()

print "The reciprocal best hits from", sys.argv[1], "and", sys.argv[2], "are in", sys.argv[3]	 			
			
			