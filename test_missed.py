fin = open("../MatchZoo_data/result/java/dssm.title.txt", "r")
qidset = set()
for line in fin:
	tokens = line.strip("\n").split("\t")
	qidset.add(tokens[0])

print(len(qidset))	
