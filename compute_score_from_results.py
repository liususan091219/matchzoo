import sys, math

def score2labellist(y_true, y_pred):
	idx2pred = dict([(x, y_pred[x]) for x in range(0, len(y_pred))])
	idx2true = dict([(x, y_true[x]) for x in range(0, len(y_true))])
	sortedidx2pred = sorted(idx2pred.items(), key = lambda x:x[1], reverse=True)
	labellist = []
	for i in range(0, len(sortedidx2pred)):
		idx = sortedidx2pred[i][0]
		thislabel = idx2true[idx]	
		labellist.append(str(thislabel))
	return labellist

def eval_score(y_true, y_pred, eval_name):
	labellist = score2labellist(y_true, y_pred)
	if eval_name == "mrr":
		return mrr(labellist)
	elif eval_name == "ndcg@10":
		return ndcg(labellist, 10)
	elif eval_name == "ndcg@100":
		return ndcg(labellist, 100)

def mrr(labellist):
        score =0
        scorecount = 0
        for i in range(0, len(labellist)):
                if labellist[i] == "1":
                        score += 1.0 / (i + 1.0)
                        scorecount += 1
        if scorecount == 0:
                return 0
        else:
                return score / scorecount

def ndcg(labellist, topK):
        dcg = idcg = 0.0
        for i in range(0, min(topK, len(labellist))):
                if labellist[i] == "1":
                        dcg += 1.0 / math.log(float(i + 2), 2.0)
        relcount = 0
        for i in range(0, len(labellist)):
                if labellist[i] == "1":
                        relcount += 1
        for i in range(0, min(topK, relcount)):
                idcg += 1.0 / math.log(float(i + 2), 2.0)
	return dcg / idcg

def load_qid2result(lang, component):
	fin = open("../MatchZoo_data/result/" + lang + "/dssm." + component + ".txt", "r")
	qid2qidlist = {}
	qid2gtlist = {}
	qid2qid2pred = {}
	qid2qid2gt = {}
	for line in fin:
		tokens = line.strip("\n").split("\t")
		qid1 = tokens[0]
		qid2 = tokens[2][1:]
		score = float(tokens[4])
		gt = tokens[6]
		qid2qidlist.setdefault(qid1, [])
		qid2qidlist[qid1].append(score)
		qid2gtlist.setdefault(qid1, [])
		qid2gtlist[qid1].append(gt)
		qid2qid2pred.setdefault(qid1, {})
		qid2qid2pred[qid1][qid2] = score
		qid2qid2gt.setdefault(qid1, {})
		qid2qid2gt[qid1][qid2] = gt
	return qid2qidlist, qid2gtlist, qid2qid2pred, qid2qid2gt

def load_qid_cosidf(lang):
	testset = set()
	fin = open("../MatchZoo_data/stackOF/data_" + lang + "/" + lang + "_test_qid.txt", "r")
	for line in fin:
		testset.add(line.strip("\n"))
	fin = open("../MatchZoo_data/stackOF/data_" + lang + "/" + lang + "_cosidf.txt", "r")
	qid2qid2list = {}
	qid2gtlist = {}
	fin.readline()
	for line in fin:
		tokens = line.strip("\n").split("\t")
		qid1 = tokens[0]		
		if qid1 not in testset:
			continue
		qid2 = tokens[1]
		gt = tokens[3]
		qid2qid2list.setdefault(qid1, [])
		qid2qid2list[qid1].append(qid2)
		qid2gtlist.setdefault(qid1, [])
		qid2gtlist[qid1].append(gt)
	return qid2qid2list, qid2gtlist

def load_idmap(lang, component):
	fin = open("../MatchZoo_data/stackOF/" + lang + "_" + component + "/idmap1.txt", "r")
	qid2idval1 = {}
	for line in fin:
		tokens = line.strip("\n").split("\t")
		qid = tokens[0]
		idval = tokens[1]
		qid2idval1[qid] = idval
	fin = open("../MatchZoo_data/stackOF/" + lang + "_" + component + "/idmap2.txt", "r")
	qid2idval2 = {}
	for line in fin:
		tokens = line.strip("\n").split("\t")
		qid = tokens[0]
		idval = tokens[1]
		qid2idval2[qid] = idval
	return qid2idval1, qid2idval2

def get_component_score(idmap1, idmap2, qid2qidlist, conc2score, conc2gts):
	qid2scorelist = {}
	qid2gtlist = {}
	for qid1 in qid2qidlist.keys():
		id1 = idmap1[qid1]
		qidlist = qid2qidlist[qid1]
		scorelist = []
		gtlist = []
		for qid2 in qidlist:
			id2 = idmap2[qid2]
			conc_id = id1 + "_" + id2
			score = conc2score[conc_id]
			gt = conc2gts[conc_id]
			scorelist.append(score)
			gtlist.append(gt)
		qid2scorelist[qid1] = scorelist
		qid2gtlist[qid1] = gtlist
	return qid2scorelist, qid2gtlist	

def compare_score_gt(qid2gtlist, qid2gtlist_2, qid2scorelist, metric):
	totalscore = 0
	scorecount = 0
	for qid in qid2gtlist.keys():
		gtlist = qid2gtlist[qid]
		gtlist_2 = qid2gtlist_2[qid]
		for i in range(0, len(gtlist)):
			gt1 = gtlist[i]
			gt2 = gtlist_2[i]
			assert gt1 == gt2
		scorelist = qid2scorelist[qid]
		thisscore = eval_score(gtlist, scorelist, metric)
		totalscore += thisscore
		scorecount += 1
	return totalscore / scorecount

def main(argv):
	qid2y_true = {}
	qid2y_pred = {}
	lang = sys.argv[1]
	metric = sys.argv[2]
	components = ["title", "question", "answer"]
	coeffs = [1.0, 0.25, 0.25]
	fout_debug = open("../MatchZoo_data/result/java/eval_score_2.txt", "w")
	qid2qid2sum = {}
	for i in range(0, 3):
		component = components[i]
		coeff = coeffs[i]
		_, _, qid2qid2pred, qid2qid2gt = load_qid2result(lang, component)
		for qid1 in qid2qid2pred.keys():
			for qid2 in qid2qid2pred[qid1].keys():
				qid2qid2sum.setdefault(qid1, {})
				qid2qid2sum[qid1].setdefault(qid2, 0)
				try:
					qid2qid2sum[qid1][qid2] += coeff * qid2qid2pred[qid1][qid2]
				except KeyError:
					import pdb
					pdb.set_trace()
					print qid1, qid2, component
					sys.exit(1)
	print(compute_score_final(qid2qid2sum, qid2qid2gt, metric))
	#for component in components:
	#	qid2qidlist, qid2gtlist = load_qid2result(lang, component)
	#	print(len(qid2qidlist))
	#	avgscore = 0.0
	#	scorecount = 0

def compute_score_final(qid2qid2pred, qid2qid2gt, metric):
	avgscore = 0
	scorecount = 0
	for qid1 in qid2qid2pred.keys():
		qid2pred = qid2qid2pred[qid1]
		qid2gt = qid2qid2gt[qid1]
		ylist = []
		gtlist = []
		for qid2 in qid2pred.keys():
			gtval = float(qid2gt[qid2])
			yval = float(qid2pred[qid2]) - gtval * 0.000001
			ylist.append(yval)
			gtlist.append(qid2gt[qid2])
		evalscore = eval_score(gtlist, ylist, metric)
		avgscore += evalscore
		scorecount += 1
	avgscore /= scorecount	
	return avgscore
#	print(avgscore), scorecount
	
if __name__=='__main__':
	main(sys.argv)	
