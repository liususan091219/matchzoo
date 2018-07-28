#!/usr/bin/env python
# coding: utf-8
from __future__ import  print_function

import os
import sys
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)

sys.path.append('../../matchzoo/inputs/')
sys.path.append('../../matchzoo/utils/')

from preparation import Preparation
from preprocess import Preprocess, NgramUtil


def filter_triletter(tri_stats, min_filter_num=5, max_filter_num=10000):
    tri_dict = {}
    tri_stats = sorted(tri_stats.items(), key=lambda d:d[1], reverse=True)
    for triinfo in tri_stats:
        if min_filter_num <= triinfo[1] <= max_filter_num:
            if triinfo[0] not in tri_dict:
                tri_dict[triinfo[0]] = len(tri_dict)
    return tri_dict

def write_idMap(idMap1, idMap2, word_dstdir):
	fout1 = open(word_dstdir + "idmap1.txt", "w")
	fout2 = open(word_dstdir + "idmap2.txt", "w")
	for qid1, id1 in idMap1.items():	
		fout1.write(qid1 + "\t" + id1 + "\n")
	for qid2, id2 in idMap2.items():
		fout2.write(qid2 + "\t" + id2 + "\n")
	fout1.close()
	fout2.close()

def read_dict(infile):
    word_dict = {}
    for line in open(infile):
        r = line.strip().split()
        word_dict[r[1]] = r[0]
    return word_dict


if __name__ == "__main__":
	lang = sys.argv[1]
	#component = sys.argv[2]
	rootdir = sys.argv[2]
	prepare = Preparation()
	srcdir = rootdir + "stackOF/data_" + lang + "/"
	word_dstdir = rootdir + "stackOF/" + lang + "_linear/"
	if not os.path.exists(word_dstdir):
		os.makedirs(word_dstdir)	
	relation_dstdir = word_dstdir 
	splitfiles = [srcdir + lang + "_train_qid.txt", srcdir + lang + "_valid_qid.txt", srcdir + lang + "_test_qid.txt"]
	corpus, rel_train, rel_valid, rel_test, idMap1, idMap2 = prepare.run_with_separate_linear(srcdir, splitfiles[0], splitfiles[1], splitfiles[2], lang)
	write_idMap(idMap1, idMap2, word_dstdir)
	print('total corpus : %d ...' % (len(corpus)))
	print('total relation-train : %d ...' % (len(rel_train)))
	print('total relation-valid : %d ...' % (len(rel_valid)))
	print('total relation-test: %d ...' % (len(rel_test)))
	prepare.save_corpus(word_dstdir + 'corpus.txt', corpus)
	
	prepare.save_relation_linear(relation_dstdir + 'relation_train.txt', rel_train)
	prepare.save_relation_linear(relation_dstdir + 'relation_valid.txt', rel_valid)
	prepare.save_relation_linear(relation_dstdir + 'relation_test.txt', rel_test)
	print('Preparation finished ...')
	
	preprocessor = Preprocess(word_stem_config={'enable': True}, word_filter_config={'min_freq': 0})
	dids, docs = preprocessor.run(word_dstdir + 'corpus.txt')
	preprocessor.save_word_dict(word_dstdir + 'word_dict.txt', True)
	preprocessor.save_words_stats(word_dstdir + 'word_stats.txt', True)
	
	fout = open(word_dstdir + 'corpus_preprocessed.txt', 'w')
	for inum, did in enumerate(dids):
	    fout.write('%s %s %s\n' % (did, len(docs[inum]), ' '.join(map(str, docs[inum]))))
	#    fout.write('%s %s\n' % (did, ' '.join(map(str, docs[inum]))))
	fout.close()
	print('Preprocess finished ...')
	
	# dssm_corp_input = dstdir + 'corpus_preprocessed.txt'
	# dssm_corp_output = dstdir + 'corpus_preprocessed_dssm.txt'
	word_dict_input = word_dstdir + 'word_dict.txt'
	triletter_dict_output = word_dstdir + 'triletter_dict.txt'
	word_triletter_output = word_dstdir + 'word_triletter_map.txt'
	word_dict = read_dict(word_dict_input)
	word_triletter_map = {}
	triletter_stats = {}
	for wid, word in word_dict.items():
	    nword = '#' + word + '#'
	    ngrams = NgramUtil.ngrams(list(nword), 3, '')
	    word_triletter_map[wid] = []
	    for tric in ngrams:
	        if tric not in triletter_stats:
	            triletter_stats[tric] = 0
	        triletter_stats[tric] += 1
	        word_triletter_map[wid].append(tric)
	triletter_dict = filter_triletter(triletter_stats, 5, 10000)
	with open(triletter_dict_output, 'w') as f:
	    for tri_id, tric in triletter_dict.items():
	        print(f, tri_id, tric, file=f)
	with open(word_triletter_output, 'w') as f:
	    for wid, trics in word_triletter_map.items():
	        print(wid, ' '.join([str(triletter_dict[k]) for k in trics if k in triletter_dict]), file=f)
	
	print('Triletter Processing finished ...')
	
