fin = open("../MatchZoo_data/result/python/dssm.tit.txt", "r")
uniqueset = set([])

for line in fin:
	tokens = line.strip("\n").split("\t")
	uniqueset.add(tokens[0])

print(len(uniqueset))
