#!/bin/bash
# help, dos2unix file
# download the wiki-qa dataset
wikiqa_path="/Data/work/xliu93/stackoverflow/MatchZoo_data/WikiQA/"
embed_path="/Data/work/xliu93/stackoverflow/MatchZoo_data/glove/glove_general/"
python_path="/Data/work/xliu93/stackoverflow/MatchZoo/data/WikiQA/"

cd $wikiqa_path
if [ ! "$(find ./ -name '*WikiQACorpus*' | head -1)" ]; then
	wget https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip
	unzip WikiQACorpus.zip
fi

# download the glove vectors
cd $embed_path
if [ ! -f glove.840B.300d.txt ]; then
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip
	unzip glove.840B.300d.zip
fi

if [ ! "$(find ./ -name '*glove.6B.*' | head -1)" ]; then
	wget http://nlp.stanford.edu/data/glove.6B.zip
	unzip glove.6B.zip
fi

# filter queries which have no right or wrong answers
cd $python_path
python filter_query.py $wikiqa_path

# transfer the dataset into matchzoo dataset format
python transfer_to_mz_format.py $wikiqa_path
# generate the mz-datasets

python prepare_mz_data.py $wikiqa_path

# generate word embedding
if [ ! -f $wikiqa_path/embed_glove_d300 ]; then
	python gen_w2v.py $embed_path/glove.840B.300d.txt $wikiqa_path/word_dict.txt $embed_path/embed_glove_d300
	python norm_embed.py $embed_path/embed_glove_d300 $embed_path/embed_glove_d300_norm
fi

if [ ! -f $wikiqa_path/embed_glove_d50 ]; then
	python gen_w2v.py $embed_path/glove.6B.50d.txt $wikiqa_path/word_dict.txt $embed_path/embed_glove_d50
	python norm_embed.py $embed_path/embed_glove_d50 $embed_path/embed_glove_d50_norm
fi

# generate data histograms for drmm model
# generate data bin sums for anmm model
# generate idf file
cd $python_path
cat $wikiqa_path/word_stats.txt | cut -d ' ' -f 1,4 > $wikiqa_path/embed.idf
python gen_hist4drmm.py 60 $wikiqa_path
python gen_binsum4anmm.py 20 $wikiqa_path # the default number of bin is 20

echo "Done ..."
