#!/usr/bin/env python
# coding: utf-8
'''
Generate bin sum used in the attention based neural matching model (aNMM)
'''
import os
import sys
sys.path.append('../../matchzoo/utils/')
sys.path.append('../../matchzoo/inputs/')
from preprocess import cal_binsum
from rank_io import *


if __name__ == '__main__':
    bin_num = int(sys.argv[1])
    srcdir = sys.argv[2] 
    embeddim = sys.argv[3]
    embedfile = srcdir + 'embed_glove_d' + embeddim + '_norm'
    corpusfile = srcdir + 'corpus_preprocessed.txt'

    relfiles = [ srcdir + 'relation_train.txt',
            srcdir + 'relation_valid.txt',
            srcdir + 'relation_test.txt'
            ]
    binfiles = [
            srcdir + 'relation_train.binsum-%d' % bin_num,
            srcdir + 'relation_valid.binsum-%d' % bin_num,
            srcdir + 'relation_test.binsum-%d' % bin_num
            ]
    embed_dict = read_embedding(filename = embedfile)
    print('read embedding finished ...')
    _PAD_ = len(embed_dict)
    embed_size = len(embed_dict[embed_dict.keys()[0]])
    embed_dict[_PAD_] = np.zeros((embed_size, ), dtype=np.float32)
    embed = np.float32(np.random.uniform(-0.2, 0.2, [_PAD_+1, embed_size]))
    embed = convert_embed_2_numpy(embed_dict, embed = embed)

    corpus, _ = read_data(corpusfile)
    print('read corpus finished....')
    for idx, relfile in enumerate(relfiles):
        binfile = binfiles[idx]
        rel = read_relation_linear(relfile)
        fout_tit = open(binfile + "_title.txt", 'w')
        fout_ques = open(binfile + "_question.txt", 'w')
        fout_ans = open(binfile + "_answer.txt", "w")
        for label, d1, d2, d3, d4 in rel:
            assert d1 in corpus
            assert d2 in corpus
            assert d3 in corpus
            assert d4 in corpus
            qnum = len(corpus[d1])
            d1_embed = embed[corpus[d1]]
            d2_embed = embed[corpus[d2]]
            d3_embed = embed[corpus[d3]]
            d4_embed = embed[corpus[d4]]
            curr_bin_sum_title = cal_binsum(d1_embed, d2_embed, qnum, bin_num)
            curr_bin_sum_ques = cal_binsum(d1_embed, d3_embed, qnum, bin_num)
            curr_bin_sum_ans = cal_binsum(d1_embed, d4_embed, qnum, bin_num)
            fout_tit.write(' '.join(map(str, curr_bin_sum_title.tolist())))
            fout_ques.write(' '.join(map(str, curr_bin_sum_ques.tolist())))
            fout_ans.write(' '.join(map(str, curr_bin_sum_ans.tolist())))
            fout_tit.write('\n')
            fout_ques.write('\n')
            fout_ans.write('\n')
        fout_tit.close()
        fout_ques.close()
        fout_ans.close()
    print 'generate bin sum finished ...'
