import math

def mrr(labellist):
    labellist = map(lambda x: str(x), labellist)
    score =0 
    scorecount = 0
    for i in range(0, len(labellist)):
        if labellist[i] == "1":
            score += 1.0 / (i + 1.0)
            scorecount += 1
    if scorecount == 0:
        print('bad')
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

