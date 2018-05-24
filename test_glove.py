from gensim.scripts.glove2word2vec import glove2word2vec

word2idx = {}
idx2word = {}

fin = open("../MatchZoo_data/stackOF/python_question/word_dict.txt", "r")

for line in fin:
	tokens = line.strip("\n").split()
	word = tokens[0]
	idx = tokens[1]
	word2idx[word] = idx
	idx2word[idx] = word

from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format("../MatchZoo_data/stackOF/python_question/embed_glove_d300", binary=False)

while True:
	newword = raw_input("prompt") 
	topwords = word_vectors.most_similar(positive=[word2idx[newword]], topn=10)
	for i in range(0, 10):
		eachword = topwords[i][0]
		print idx2word[eachword]
