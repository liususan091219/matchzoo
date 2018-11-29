from scipy import stats

array1 = []
array2 = []

lang = "javascript"
metric = "mrr"

fin = open("../MatchZoo_data/ttest/" + lang + "_" + metric + "_xue.txt", "r")

for line in fin:
	array1.append(float(line.strip("\n")))

fin = open("../MatchZoo_data/ttest/" + lang + "_" + metric + "_drmm.txt", "r")

for line in fin:
	array2.append(float(line.strip("\n")))

print stats.ttest_ind(array1, array2)
