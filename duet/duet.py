from __future__ import print_function
import sys
import os
import csv
import re
import time
import math
import numpy as np
import cntk as C

# uncomment the following if you want to run on cpu
#C.try_set_default_device(C.cpu(), acquire_device_lock=False)
C.cntk_py.set_fixed_random_seed(1)

class Sample:
    
    def __init__(self):
        self.query = ""
        self.docs = []
    
class DataReader:
    max_query_words = 10
    max_doc_words = 1000
    
    def __init__(self, data_file, ngraphs_file, num_docs, num_meta_cols, multi_pass):
        self.__load_ngraphs(ngraphs_file)
        self.data_file = open(data_file, mode='r')
        self.num_docs = num_docs
        self.num_meta_cols = num_meta_cols
        self.multi_pass = multi_pass
    
    def __load_ngraphs(self, filename):
        self.ngraphs = {}
        self.max_ngraph_len = 0
        with open(filename, mode='r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self.ngraphs[row[0]] = int(row[1]) - 1
                self.max_ngraph_len = max(self.max_ngraph_len, len(row[0]))
        self.num_ngraphs = len(self.ngraphs)

    def __read_samples(self, num_samples):
        labels = np.zeros((num_samples, self.num_docs), dtype=np.float32)
        samples = []
        meta = []
        mb_size = 0
        for i in range(num_samples):
            row = self.data_file.readline()
            if row == "":
                if self.multi_pass:
                    self.data_file.seek(0)
                    row = self.data_file.readline()
                else:
                    break
            cols = row.split('\t')
            curr_sample = Sample()
            curr_sample.query = re.sub('[^0-9a-z\t]+', ' ', cols[self.num_meta_cols].lower()).strip()
            for j in range(self.num_meta_cols+1, min(self.num_meta_cols+self.num_docs+1, len(cols))):
                curr_sample.docs.append(re.sub('[^0-9a-z\t]+', ' ', cols[j].lower()).strip())
            samples.append(curr_sample)
            labels[i][0] = np.float32(1)
            meta.append([cols[i] for i in range(0, self.num_meta_cols)])
            mb_size += 1
        return samples, labels, meta, mb_size
        
    def __get_interaction_features(self, samples):
        features = np.zeros((len(samples), self.num_docs, self.max_query_words, self.max_doc_words), dtype=np.float32)
        for sample_idx, sample in enumerate(samples):
            for doc_idx, doc in enumerate(sample.docs):
                for qw_idx, qword in enumerate(sample.query.split()):
                    if qw_idx == self.max_query_words:
                        break
                    for dw_idx, dword in enumerate(doc.split()):
                        if dw_idx == self.max_doc_words:
                            break
                        if qword == dword:
                            features[sample_idx, doc_idx, qw_idx, dw_idx] = np.float32(1)
        return features
        
    def __get_ngraph_features(self, samples):
        features_query = np.zeros((len(samples), self.num_ngraphs, self.max_query_words), dtype=np.float32)
        features_docs = np.zeros((len(samples), self.num_docs, self.num_ngraphs, self.max_doc_words), dtype=np.float32)
        for sample_idx, sample in enumerate(samples):
            # loop over query and docs -- doc_idx = 0 corresponds to query 
            for doc_idx in range(len(sample.docs)+1):
                doc = sample.query if doc_idx == 0 else sample.docs[doc_idx-1]
                max_words = self.max_query_words if doc_idx == 0 else self.max_doc_words
                for w_idx, word in enumerate(doc.split()):
                    if w_idx == max_words:
                        break
                    token = '#' + word + '#'
                    token_len = len(token)
                    for i in range(token_len):
                        for j in range(0, self.max_ngraph_len):
                            if i+j < token_len:
                                ngraph_idx = self.ngraphs.get(token[i:i+j])
                                if ngraph_idx != None:
                                    if doc_idx == 0:
                                        features_query[sample_idx, ngraph_idx, w_idx] += 1
                                    else:
                                        features_docs[sample_idx, doc_idx-1, ngraph_idx, w_idx] += 1
        return features_query, features_docs

    def get_minibatch(self, num_samples):
        samples, labels, meta, mb_size = self.__read_samples(num_samples)
        features_local = self.__get_interaction_features(samples)
        features_distrib_query, features_distrib_docs = self.__get_ngraph_features(samples)
        return features_local, features_distrib_query, features_distrib_docs, labels, meta, mb_size


def duet(features_local, features_distrib_query, features_distrib_docs, num_ngraphs, words_per_query, words_per_doc, num_docs):
    num_hidden_nodes = 300
    word_window_size = 3
    pooling_kernel_width_query = words_per_query - word_window_size + 1 # = 8
    pooling_kernel_width_doc = 100
    num_pooling_windows_doc = ((words_per_doc - word_window_size + 1) - pooling_kernel_width_doc) + 1 # = 899
                        
    duet_local    = C.layers.Sequential ([
                        C.layers.Convolution((1, words_per_doc), num_hidden_nodes, activation=C.tanh, strides=(1, 1), pad=False),
                        C.layers.Dense(num_hidden_nodes, activation=C.tanh),
                        C.layers.Dense(num_hidden_nodes, activation=C.tanh),
                        C.layers.Dropout(0.2),
                        C.layers.Dense(1, activation=C.tanh)])
                        
    duet_embed_q  = C.layers.Sequential ([
                        C.layers.Convolution((word_window_size, 1), num_hidden_nodes, activation=C.tanh, strides=(1, 1), pad=False),
                        C.layers.MaxPooling((pooling_kernel_width_query, 1), strides=(1, 1), pad=False),
                        C.layers.Dense(num_hidden_nodes, activation=C.tanh)])
                        
    duet_embed_d  = C.layers.Sequential ([
                        C.layers.Convolution((word_window_size, 1), num_hidden_nodes, activation=C.tanh, strides=(1, 1), pad=False),
                        C.layers.MaxPooling((pooling_kernel_width_doc, 1), strides=(1, 1), pad=False),
                        C.layers.Convolution((1, 1), num_hidden_nodes, activation=C.tanh, strides=(1, 1), pad=False)])
                        
    duet_distrib  = C.layers.Sequential ([
                        C.layers.Dense(num_hidden_nodes, activation=C.tanh),
                        C.layers.Dense(num_hidden_nodes, activation=C.tanh),
                        C.layers.Dropout(0.2),
                        C.layers.Dense(1, activation=C.tanh)])
    
    net_local       = [C.slice(features_local, 0, idx, idx+1) for idx in range(0, num_docs)]
    net_local       = [C.reshape(d, (1, words_per_query, words_per_doc)) for d in net_local]
    net_local       = [duet_local(d) for d in net_local]
    net_local       = [C.reshape(d, (1, 1)) for d in net_local]
    net_local       = C.splice(*net_local)
    
    net_distrib_q   = C.reshape(features_distrib_query, (num_ngraphs, words_per_query, 1))
    net_distrib_q   = duet_embed_q(net_distrib_q)
    net_distrib_q   = [net_distrib_q for idx in range(0, num_pooling_windows_doc)]
    net_distrib_q   = C.splice(*net_distrib_q)
    net_distrib_q   = C.reshape(net_distrib_q, (num_hidden_nodes * num_pooling_windows_doc, 1))
    
    net_distrib_d   = [C.slice(features_distrib_docs, 0, idx, idx+1) for idx in range(0, num_docs)]
    net_distrib_d   = [C.reshape(d, (num_ngraphs, words_per_doc, 1)) for d in net_distrib_d]
    net_distrib_d   = [duet_embed_d(d) for d in net_distrib_d]
    net_distrib_d   = [C.reshape(d, (num_hidden_nodes * num_pooling_windows_doc, 1)) for d in net_distrib_d]

    net_distrib     = [C.element_times(net_distrib_q, d) for d in net_distrib_d]
    net_distrib     = [duet_distrib(d) for d in net_distrib]
    net_distrib     = [C.reshape(d, (1, 1)) for d in net_distrib]
    net_distrib     = C.splice(*net_distrib)
                        
    net             = C.plus(net_local, net_distrib)
    
    return net

def train(train_file, ngraphs_file, num_docs, num_meta_cols):
    
    # initialize train data readers
    reader_train = DataReader(train_file, ngraphs_file, num_docs, num_meta_cols, True)
       
    # input variables denoting the features and label data
    features_local         = C.input_variable((reader_train.num_docs, reader_train.max_query_words, reader_train.max_doc_words), np.float32)
    features_distrib_query = C.input_variable((reader_train.num_ngraphs, reader_train.max_query_words), np.float32)
    features_distrib_docs  = C.input_variable((reader_train.num_docs, reader_train.num_ngraphs, reader_train.max_doc_words), np.float32)
    labels                 = C.input_variable((reader_train.num_docs), np.float32)

    # Instantiate the Duet neural document ranking model and specify loss function
    z = duet(features_local, features_distrib_query, features_distrib_docs, reader_train.num_ngraphs, reader_train.max_query_words, reader_train.max_doc_words, reader_train.num_docs)
    ce = C.cross_entropy_with_softmax(z, labels)
    pe = C.classification_error(z, labels)

    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = C.learning_rate_schedule(0.001, C.UnitType.minibatch)
    progress_printers = [C.logging.ProgressPrinter(freq=100, tag='Training', gen_heartbeat=False)]
    trainer = C.Trainer(z, (ce, pe), [C.sgd(z.parameters, lr=lr_per_minibatch)], progress_printers)

    # Get minibatches of training data and perform model training
    minibatch_size = 64
    minibatches_per_epoch = 32
    epochs = 4
    
    C.logging.log_number_of_parameters(ce)
    print()
    
    for i in range(epochs):
        for j in range(minibatches_per_epoch):
            train_features_local, train_features_distrib_query, train_features_distrib_docs, train_labels, train_meta, actual_mb_size = reader_train.get_minibatch(minibatch_size)
            trainer.train_minibatch({features_local : train_features_local, features_distrib_query : train_features_distrib_query, features_distrib_docs : train_features_distrib_docs, labels : train_labels})
        trainer.summarize_training_progress()

    return z

def eval(model, test_file, ngraphs_file, num_docs, num_meta_cols, score_file):
    
    minibatch_size = 64
    actual_mb_size = minibatch_size
    
    # initialize test data readers
    reader_test  = DataReader(test_file, ngraphs_file, num_docs, num_meta_cols, False)

    with open(score_file, mode='w') as f:
        while(actual_mb_size == minibatch_size):
            test_features_local, test_features_distrib_query, test_features_distrib_docs, test_labels, test_meta, actual_mb_size = reader_test.get_minibatch(minibatch_size)
            if actual_mb_size > 0:
                result = model.eval({model.arguments[0] : test_features_local, model.arguments[1] : test_features_distrib_query, model.arguments[2] : test_features_distrib_docs})
                result = result.reshape((actual_mb_size, num_docs))
                result = [row[0] for row in result]
                for idx in range(actual_mb_size):
                    f.write("{}\t{}\t{}\t{}\n".format(test_meta[idx][0], test_meta[idx][1], test_meta[idx][2], result[idx]))


def ComputeDCG(sorted_ranks):
    dcg = 0
    for pos, rating in enumerate(sorted_ranks):
        dcg += ((2^rating - 1)/math.log2(pos + 2))
    return dcg

def ComputeNDCGPerQuery(ideal_ratings, scored_ratings):
    ideal_ranks = sorted(ideal_ratings, reverse=True)
    model_ranks = [pair[1] for pair in sorted(scored_ratings, key=lambda tup: tup[0], reverse=True)]
    ideal_dcg = ComputeDCG(ideal_ranks)
    model_dcg = ComputeDCG(model_ranks)
    return model_dcg / ideal_dcg if ideal_dcg > 0 else 0

def ComputeNDCG(score_file, ndcg_pos):
    ndcg = 0
    curr_qid = -1
    ideal_ratings = []
    scored_ratings = []
    q_count = 0
    
    with open(score_file, mode='r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            q_id   = row[0]
            doc_id = row[1]
            rating = int(row[1])
            score  = float(row[1])
            
            if q_id != curr_qid:
                ndcg += ComputeNDCGPerQuery(ideal_ratings, scored_ratings)
                q_count += 1
                ideal_ratings = []
                scored_ratings = []
                
            curr_qid = q_id
            ideal_ratings.append(rating)
            scored_ratings.append((score, rating))
            
    ndcg += ComputeNDCGPerQuery(ideal_ratings, scored_ratings)
    q_count += 1
    
    return ndcg / q_count

data_path = "/Data/work/xliu93/stackoverflow/MatchZoo_data/duet_cntk/"
ngraphs_file = data_path + "ngraphs.txt"
train_file = data_path + "train.txt"
test_file = data_path + "test.txt"
score_file = data_path + "score.txt"

model = train(train_file, ngraphs_file, 2, 0)
eval(model, test_file, ngraphs_file, 2, 3, score_file)
ndcg = ComputeNDCG(score_file, 10)
print("test ndcg = {}".format(ndcg))
