currentdir=$PWD
parentdir="$(dirname "$currentdir")"
grandparentdir="$(dirname "$parentdir")"
rootdir="$(dirname "$grandparentdir")"/MatchZoo_data/

data_path=$rootdir/stackOF/$1_linear/
code_path=$grandparentdir/data/stackOF/
glove_dir=$rootdir/glove/

cd $code_path
python prepare_mz_data_linear.py $1 $rootdir

# generate word embedding

array=( 200 )
for i in "${array[@]}"
do
	fold_name="glove_stackOF"
	filename="gensim_glove_"${i}"d.txt"
    echo $fold_name
    python gen_w2v.py ${glove_dir}/${fold_name}/${filename} ${data_path}/word_dict.txt ${data_path}/embed_glove_d${i}
    python norm_embed.py ${data_path}/embed_glove_d${i} ${data_path}/embed_glove_d${i}_norm
done

# generate data histograms for drmm model
# generate data bin sums for anmm model
# generate idf file
cat ${data_path}word_stats.txt | cut -d ' ' -f 1,4 > ${data_path}embed.idf

# python gen_hist4drmm.py 60 $data_path 200
python gen_binsum4anmm_linear.py 20 $data_path 200 # the default number of bin is 20

echo "Done ..."
