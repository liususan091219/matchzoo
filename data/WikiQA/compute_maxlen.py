fin = open("corpus_preprocessed.txt", "r")
qmax = -1
dmax = -1

for line in fin:
	tokens = line.strip("\n").split()
	if line.startswith("Q"):
		qmax = max(qmax, len(tokens) - 2)
	else:
		dmax = max(dmax, len(tokens) - 2)

print qmax, dmax
