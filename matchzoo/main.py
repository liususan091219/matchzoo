# -*- coding: utf8 -*-
from __future__ import print_function
import os, math
import sys
import time
import json
import argparse
import random
random.seed(49999)
import numpy
numpy.random.seed(49999)
import tensorflow
tensorflow.set_random_seed(49999)

from collections import OrderedDict

import keras
import keras.backend as K
from keras.models import Sequential, Model

from utils import *
import inputs
import metrics
from losses import *

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
sess = tensorflow.Session(config = config)

def score2labellist(y_true, y_pred):
	idx2pred = dict([(x, y_pred[x]) for x in range(0, len(y_pred))])
	idx2true = dict([(x, y_true[x]) for x in range(0, len(y_true))])
	sortedidx2pred = sorted(idx2pred.items(), key = lambda x:x[1], reverse=True)
	labellist = []
	for i in range(0, len(sortedidx2pred)):
		idx = sortedidx2pred[i][0]
		thislabel = idx2true[idx]	
		labellist.append(str(thislabel))
	return labellist

def eval_score(y_true, y_pred, eval_name, eval_func):
	labellist = score2labellist(y_true, y_pred)
	if eval_name == "mrr":
		return mrr(labellist)
	elif eval_name == "ndcg@10":
		return ndcg(labellist, 10)
	elif eval_name == "ndcg@100":
		return ndcg(labellist, 100)
	else:
		assert eval_name == "map" or eval_name.startswith("ndcg")
		return eval_func(y_true = y_true, y_pred = y_pred)

def mrr(labellist):
        score =0
        scorecount = 0
        for i in range(0, len(labellist)):
                if labellist[i] == "1":
                        score += 1.0 / (i + 1.0)
                        scorecount += 1
        if scorecount == 0:
                return 0
        else:
                return score / scorecount

def ndcg(labellist, topK):
        dcg = idcg = 0.0
        for i in range(0, min(topK, len(labellist))):
                if labellist[i] == "1":
                        dcg += 1.0 / math.log(float(i + 2), 2.0)
        relcount = 0
        for i in range(0, len(labellist)):
                if labellist[i] == "1":
                        relcount += 1
        for i in range(0, min(topK, relcount)):
                idcg += 1.0 / math.log(float(i + 2), 2.0)
	return dcg / idcg


def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']['setting']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, config['model']['model_path'])

        model = import_object(config['model']['model_py'], model_config)
        mo = model.build()
    return mo


def train(config, data_root, log_file):

    print(json.dumps(config, indent=2), end='\n')
    # read basic config
    global_conf = config["global"]
    optimizer = global_conf['optimizer']
    weights_file = data_root + str(global_conf['weights_file']) + '.%d'
    display_interval = int(global_conf['display_interval'])
    num_iters = int(global_conf['num_iters'])
    save_weights_iters = int(global_conf['save_weights_iters'])
    fout_log = open(data_root + log_file, "w")

    # read input config
    input_conf = config['inputs']
    share_input_conf = input_conf['share']


    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print('[Embedding] Embedding Load Done.', end='\n')

    # list all input tags and construct tags config
    input_train_conf = OrderedDict()
    input_eval_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'TRAIN':
            input_train_conf[tag] = {}
            input_train_conf[tag].update(share_input_conf)
            input_train_conf[tag].update(input_conf[tag])
        elif input_conf[tag]['phase'] == 'EVAL':
            input_eval_conf[tag] = {}
            input_eval_conf[tag].update(share_input_conf)
            input_eval_conf[tag].update(input_conf[tag])
    print('[Input] Process Input Tags. %s in TRAIN, %s in EVAL.' % (input_train_conf.keys(), input_eval_conf.keys()), end='\n')

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag != 'share' and input_conf[tag]['phase'] == 'PREDICT':
            continue
        if 'text1_corpus' in input_conf[tag]:
            datapath = data_root + input_conf[tag]['text1_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
        if 'text2_corpus' in input_conf[tag]:
            datapath = data_root + input_conf[tag]['text2_corpus']
            if datapath not in dataset:
                dataset[datapath], _ = read_data(datapath)
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')

    # initial data generator
    train_gen = OrderedDict()
    eval_gen = OrderedDict()

    for tag, conf in input_train_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[data_root + conf['text1_corpus']]
        conf['data2'] = dataset[data_root + conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        train_gen[tag] = generator(data_root,  config = conf )

    for tag, conf in input_eval_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[data_root + conf['text1_corpus']]
        conf['data2'] = dataset[data_root + conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        eval_gen[tag] = generator(data_root,  config = conf )

    ######### Load Model #########
    model = load_model(config)

    loss = []
    for lobj in config['losses']:
        if lobj['object_name'] in mz_specialized_losses:
            loss.append(rank_losses.get(lobj['object_name'])(lobj['object_params']))
        else:
            loss.append(rank_losses.get(lobj['object_name']))
    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
    model.compile(optimizer=optimizer, loss=loss)
    print('[Model] Model Compile Done.', end='\n')

    for i_e in range(num_iters):
        for tag, generator in train_gen.items():
            genfun = generator.get_batch_generator()
            print('[%s]\t[Train:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
            history = model.fit_generator(
                    genfun,
                    steps_per_epoch = display_interval,
                    epochs = 1,
                    shuffle=False,
                    verbose = 0
                ) #callbacks=[eval_map])
            print('Iter:%d\tloss=%.6f' % (i_e, history.history['loss'][0]), end='\n')

        for tag, generator in eval_gen.items():
            genfun = generator.get_batch_generator()
            print('[%s]\t[Eval:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
            res = dict([[k,0.] for k in eval_metrics.keys()])
            num_valid = 0
            for input_data, y_true in genfun:
                y_pred = model.predict(input_data, batch_size=len(y_true))
                if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                    list_counts = input_data['list_counts']
                    for k, eval_func in eval_metrics.items():
                        for lc_idx in range(len(list_counts)-1):
                            pre = list_counts[lc_idx]
                            suf = list_counts[lc_idx+1]
                            res[k] += eval_score(y_true[pre:suf], y_pred[pre:suf], k, eval_func)
          #                  res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])
                    num_valid += len(list_counts) - 1
                else:
                    for k, eval_func in eval_metrics.items():
                        res[k] += eval_score(y_true, y_pred, k, eval_func)
                 #       res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                    num_valid += 1
            generator.reset()
            print('Iter:%d\t%s' % (i_e, '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()])), end='\n')
            fout_log.write('Iter:%d\t%s' % (i_e, '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()])) + "\n")
            sys.stdout.flush()
        if (i_e+1) % save_weights_iters == 0:
            model.save_weights(weights_file % (i_e+1))
    fout_log.close()

def predict(config):
    ######## Read input config ########

    print(json.dumps(config, indent=2), end='\n')
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'], ), dtype=np.float32)
        embed = np.float32(np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed = embed)
    else:
        embed = np.float32(np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print('[Embedding] Embedding Load Done.', end='\n')

    # list all input tags and construct tags config
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print('[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys()), end='\n')

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag == 'share' or input_conf[tag]['phase'] == 'PREDICT':
            if 'text1_corpus' in input_conf[tag]:
                datapath = data_root + input_conf[tag]['text1_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
            if 'text2_corpus' in input_conf[tag]:
                datapath = data_root + input_conf[tag]['text2_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')

    # initial data generator
    predict_gen = OrderedDict()

    for tag, conf in input_predict_conf.items():
        print(conf, end='\n')
        conf['data1'] = dataset[data_root + conf['text1_corpus']]
        conf['data2'] = dataset[data_root + conf['text2_corpus']]
        generator = inputs.get(conf['input_type'])
        predict_gen[tag] = generator(
                                    #data1 = dataset[conf['text1_corpus']],
                                    #data2 = dataset[conf['text2_corpus']],
                                     config = conf )

    ######## Read output config ########
    output_conf = config['outputs']

    ######## Load Model ########
    global_conf = config["global"]
    weights_file = str(global_conf['weights_file']) + '.' + str(global_conf['test_weights_iters'])

    model = load_model(config)
    model.load_weights(weights_file)

    eval_metrics = OrderedDict()
    for mobj in config['metrics']:
        mobj = mobj.lower()
        if '@' in mobj:
            mt_key, mt_val = mobj.split('@', 1)
            eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
        else:
            eval_metrics[mobj] = metrics.get(mobj)
    res = dict([[k,0.] for k in eval_metrics.keys()])

    for tag, generator in predict_gen.items():
        genfun = generator.get_batch_generator()
        print('[%s]\t[Predict] @ %s ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), tag), end='')
        num_valid = 0
        res_scores = {}
        for input_data, y_true in genfun:
            y_pred = model.predict(input_data, batch_size=len(y_true) )

            if issubclass(type(generator), inputs.list_generator.ListBasicGenerator):
                list_counts = input_data['list_counts']
                for k, eval_func in eval_metrics.items():
                    for lc_idx in range(len(list_counts)-1):
                        pre = list_counts[lc_idx]
                        suf = list_counts[lc_idx+1]
                        res[k] += eval_score(y_true[pre:suf], y_pred[pre:suf], k, eval_func)
#                        res[k] += eval_func(y_true = y_true[pre:suf], y_pred = y_pred[pre:suf])

                y_pred = np.squeeze(y_pred)
                for lc_idx in range(len(list_counts)-1):
                    pre = list_counts[lc_idx]
                    suf = list_counts[lc_idx+1]
                    for p, y, t in zip(input_data['ID'][pre:suf], y_pred[pre:suf], y_true[pre:suf]):
                        if p[0] not in res_scores:
                            res_scores[p[0]] = {}
                        res_scores[p[0]][p[1]] = (y, t)

                num_valid += len(list_counts) - 1
            else:
                for k, eval_func in eval_metrics.items():
                    res[k] += eval_score(y_true, y_pred, k, eval_func)
                    #res[k] += eval_func(y_true = y_true, y_pred = y_pred)
                for p, y, t in zip(input_data['ID'], y_pred, y_true):
                    if p[0] not in res_scores:
                        res_scores[p[0]] = {}
                    res_scores[p[0]][p[1]] = (y[1], t[1])
                num_valid += 1
        generator.reset()

        if tag in output_conf:
            if output_conf[tag]['save_format'] == 'TREC':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            f.write('%s\tQ0\t%s\t%d\t%f\t%s\t%s\n'%(qid, did, inum, score, config['net_name'], gt))
            elif output_conf[tag]['save_format'] == 'TEXTNET':
                with open(output_conf[tag]['save_path'], 'w') as f:
                    for qid, dinfo in res_scores.items():
                        dinfo = sorted(dinfo.items(), key=lambda d:d[1][0], reverse=True)
                        for inum,(did, (score, gt)) in enumerate(dinfo):
                            f.write('%s %s %s %s\n'%(gt, qid, did, score))

        print('[Predict] results: ', '\t'.join(['%s=%f'%(k,v/num_valid) for k, v in res.items()]), end='\n')
        sys.stdout.flush()

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--model_file', default='./models/arci.config', help='Model_file: MatchZoo model file for the chosen model.')
    parser.add_argument("--data_root", default="/Data/work/xliu93/stackoverflow/MatchZoo_data/", help="data root")
    parser.add_argument("--log_file", default="", help = "log file")
    args = parser.parse_args()
    model_file =  args.model_file
    log_file = args.log_file
    data_root = args.data_root
    with open(model_file, 'r') as f:
        config = json.load(f)
    phase = args.phase
    if args.phase == 'train':
        train(config, data_root, log_file)
    elif args.phase == 'predict':
        predict(config)
    else:
        print('Phase Error.', end='\n')
    return

if __name__=='__main__':
    main(sys.argv)
