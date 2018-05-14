#!/bin/bash
# help, dos2unix file
# download the wiki-qa dataset
if [ ! "$(find ./ -name '*WikiQACorpus*' | head -1)" ]; then
	wget https://download.microsoft.com/download/E/5/F/E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip
	unzip WikiQACorpus.zip
fi

# download the glove vectors
data_path="/Data/work/xliu93/stackoverflow/MatchZoo_data/glove/glove_general/"
cd $data_path
if [ ! -f glove.840B.300d.txt ]; then
	wget http://nlp.stanford.edu/data/glove.840B.300d.zip
	unzip glove.840B.300d.zip
fi

if [ ! "$(find ./ -name '*glove.6B.*' | head -1)" ]; then
	wget http://nlp.stanford.edu/data/glove.6B.zip
	unzip glove.6B.zip
fi

current_dir="/Data/work/xliu93/stackoverflow/MatchZoo/data/WikiQA/"
# filter queries which have no right or wrong answers
cd $current_dir
python filter_query.py

# transfer the dataset into matchzoo dataset format
python transfer_to_mz_format.py
# generate the mz-datasets
python prepare_mz_data.py

# generate word embedding
if [ ! -f $data_path/embed_glove_d300 ]; then
	python gen_w2v.py $data_path/glove.840B.300d.txt word_dict.txt $data_path/embed_glove_d300
	python norm_embed.py $data_path/embed_glove_d300 $data_path/embed_glove_d300_norm
fi

if [ ! -f $data_path/embed_glove_d50 ]; then
	python gen_w2v.py $data_path/glove.6B.50d.txt word_dict.txt $data_path/embed_glove_d50
	python norm_embed.py $data_path/embed_glove_d50 $data_path/embed_glove_d50_norm
fi

# generate data histograms for drmm model
# generate data bin sums for anmm model
# generate idf file
cat word_stats.txt | cut -d ' ' -f 1,4 > embed.idf
python gen_hist4drmm.py 60
python gen_binsum4anmm.py 20 # the default number of bin is 20

echo "Done ..."
