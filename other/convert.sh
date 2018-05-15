python gen_w2v.py glove/gensim_glove_50d.txt word_dict.txt embed_glove_d50
python ../norm_embed.py embed_glove_d50 embed_glove_d50_norm
python gen_w2v.py glove/gensim_glove_100d.txt word_dict.txt embed_glove_d100
python ../norm_embed.py embed_glove_d100 embed_glove_d100_norm
python gen_w2v.py glove/gensim_glove_200d.txt word_dict.txt embed_glove_d200
python ../norm_embed.py embed_glove_d200 embed_glove_d200_norm
python gen_w2v.py glove/gensim_glove_300d.txt word_dict.txt embed_glove_d300
python ../norm_embed.py embed_glove_d300 embed_glove_d300_norm
