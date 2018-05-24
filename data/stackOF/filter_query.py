# coding: utf-8
"""
filter queries which have no right or wrong answers
Given dev.txt, dev.ref, dev-filtered.ref
Output dev-filtered.ref
"""
from __future__ import print_function

import os
import sys

if __name__ == '__main__':
<<<<<<< HEAD:data/stackOF/filter_query.py
    basedir = './stackOFCorpus/'
    filter_reffile = [basedir + 'stackOF-dev-filtered.ref', basedir + 'stackOF-test-filtered.ref']
    in_reffile = [basedir + 'stackOF-dev.ref', basedir + 'stackOF-test.ref']
    in_corpfile = [basedir + 'stackOF-dev.txt', basedir + 'stackOF-test.txt']
    outfile = [basedir + 'stackOF-dev-filtered.txt', basedir + 'stackOF-test-filtered.txt']
=======
    wikiqa_path = sys.argv[1]
    basedir = wikiqa_path + '/WikiQACorpus/'
    filter_reffile = [basedir + 'WikiQA-dev-filtered.ref', basedir + 'WikiQA-test-filtered.ref']
    in_reffile = [basedir + 'WikiQA-dev.ref', basedir + 'WikiQA-test.ref']
    in_corpfile = [basedir + 'WikiQA-dev.txt', basedir + 'WikiQA-test.txt']
    outfile = [basedir + 'WikiQA-dev-filtered.txt', basedir + 'WikiQA-test-filtered.txt']
>>>>>>> b2cc427e75f276a74dae5fcf2ec02ec52313f7d8:data/WikiQA/filter_query.py

    for i in range(len(filter_reffile)):
        fout = open(outfile[i], 'w')

        filtered_qids = set()
        for line in open(filter_reffile[i], 'r'):
            r = line.strip().split()
            filtered_qids.add(r[0])

        all_qids = []
        for line in open(in_reffile[i], 'r'):
            r = line.strip().split()
            all_qids.append(r[0])

        for idx,line in enumerate(open(in_corpfile[i], 'r')):
            if all_qids[idx] not in filtered_qids:
                continue
            print(line.strip(), file=fout)
        fout.close()

