#!/usr/bin/env python
# coding: utf-8

import os
import sys


basedir = './stackOFACorpus/'
dstdir = './'
infiles = [ basedir + 'stackOFA-train.txt', basedir + 'stackOFA-dev-filtered.txt', basedir + 'stackOFA-test-filtered.txt' ]
outfiles = [ dstdir + 'stackOFA-mz-train.txt', dstdir + 'stackOFA-mz-dev.txt', dstdir + 'stackOFA-mz-test.txt' ]

for idx,infile in enumerate(infiles):
    outfile = outfiles[idx]
    fout = open(outfile, 'w')
    for line in open(infile, 'r'):
        r = line.strip().split('\t')
        fout.write('%s\t%s\t%s\n'%(r[2], r[0], r[1]))
    fout.close()



