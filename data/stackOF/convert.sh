lang=$1
component=$2
glove_dir="/Data/work/xliu93/stackoverflow/MatchZoo_data/glove"
data_dir="/Data/work/xliu93/stackoverflow/MatchZoo_data/stackOF/"${lang}_${component}
array=( 50 100 200 300 )
for i in "${array[@]}"
do
	python gen_w2v.py ${glove_dir}/glove_stackOF/gensim_glove_${i}d.txt ${data_dir}/word_dict.txt ${data_dir}/embed_glove_d${i}
	python norm_embed.py ${data_dir}/embed_glove_d${i} ${data_dir}/embed_glove_d${i}_norm
done
