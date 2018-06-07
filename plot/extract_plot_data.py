import re

data_path = "../../MatchZoo_data/training_scores/python/"
out_path = "../../MatchZoo_data/plot/python/"

#file_names = ["anmm.ques.10.40.glove50", "anmm.ques.20.40.glove50", "anmm.ques.10.120.glove50", "anmm.ques.20.120.glove50",  "anmm.ques.glove300",  "anmm.ques.10.40.50", "anmm.ques.10.40.300", "anmm.ques.10.120.300"]
#file_names = ["anmm.ques.10.40.glove50", "anmm.ques.20.40.glove50", "anmm.ques.10.120.glove50", "anmm.ques.20.120.glove50"]
file_names = ["anmm.tit.10.40.glove50", "anmm.tit"]

regex = re.compile("^test.*ndcg\@10\=(?P<score1>.*?)\s*ndcg\@100\=(?P<score2>.*?)\s*mrr\=(?P<score3>.*?)( )*$")

fout = open(out_path + "mrr.txt", "w")
fout2 = open(out_path + "ndcg10.txt", "w")
fout3 = open(out_path + "ndcg100.txt", "w")

for eachfile in file_names:
	fin = open(data_path + eachfile, "r")
	fout.write(eachfile + "\t")
	fout2.write(eachfile + "\t")
	fout3.write(eachfile + "\t")
	for line in fin:
		result = regex.search(line.strip("\r\n"))	
		if result:
			score1 = float(result.group("score1"))
			score2 = float(result.group("score2"))
			score3 = float(result.group("score3"))
			fout.write(str(score3) + " ")
			fout.flush()
			fout2.write(str(score1) + " ")
			fout2.flush()
			fout3.write(str(score2) + " ")
			fout3.flush()
	fout.write("\n")
	fout2.write("\n")
	fout3.write("\n")

fout.close()	
fout2.close()
fout3.close()
