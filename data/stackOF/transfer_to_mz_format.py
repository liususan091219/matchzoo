# coding: utf-8

import os
import sys

wikiqa_path = sys.argv[1]

<<<<<<< HEAD:data/stackOF/transfer_to_mz_format.py
basedir = './stackOFACorpus/'
=======
basedir = wikiqa_path +'/WikiQACorpus/'
>>>>>>> b2cc427e75f276a74dae5fcf2ec02ec52313f7d8:data/WikiQA/transfer_to_mz_format.py
dstdir = './'
infiles = [ basedir + 'stackOFA-train.txt', basedir + 'stackOFA-dev-filtered.txt', basedir + 'stackOFA-test-filtered.txt' ]
outfiles = [ dstdir + 'stackOFA-mz-train.txt', dstdir + 'stackOFA-mz-dev.txt', dstdir + 'stackOFA-mz-test.txt' ]

for idx, infile in enumerate(infiles):
    outfile = outfiles[idx]
    fout = open(outfile, 'w')
    for line in open(infile, 'r'):
        r = line.strip().split('\t')
        fout.write('%s\t%s\t%s\n' % (r[2], r[0], r[1]))
    fout.close()



