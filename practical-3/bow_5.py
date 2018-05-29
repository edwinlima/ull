# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import os


optim = 'adam'
batch_size = 64
tenacity = 5
epoch_size = 4


optim = 'rmsprop'
batch_size = 128
tenacity = 3
epoch_size = 30

# Set PATHs
PATH_TO_SENTEVAL = '/Users/edwinlima/git/SentEval/' 
PATH_TO_DATA = PATH_TO_SENTEVAL + '/data'
# PATH_TO_VEC = 'glove/glove.840B.300d.txt'
PATH_TO_VEC = PATH_TO_SENTEVAL + 'examples/fasttext/crawl-300d-2M.vec'
PATH_TO_VEC = './skipgram_ep_30_size_100_mincount_10_win_5.vec'
PATH_TO_VEC = './skipgram_ep_30_size_100_mincount_1_win_5.vec'
PATH_TO_VEC = './skipgram_ep_30_size_100_mincount_20_win_5.vec'
PATH_TO_VEC = './skipgram_ep_30_size_100_mincount_None_win_None.vec'
PATH_TO_VEC = './skipgram_europarl_ep_30_size_100_mincount_200_win_5.vec'
PATH_TO_VEC = './skipgram_europarl_ep_40_size_100_mincount_200_win_5.vec'
PATH_TO_VEC = './skipgram_europarl_ep_20_size_100_mincount_200_win_5.vec'
PATH_TO_VEC = './glove.840B.300d.txt'
emb_size = 300

#PATH_TO_VEC = './skipgram_ep_30_size_300_mincount_10_win_5.vec'
#emb_size = 300

logPath = './logging/'
fileName = os.path.splitext(PATH_TO_VEC)[0]  + "optim_%s_batch_%s_tenac_%s_ep_%s"%(optim, batch_size,tenacity, epoch_size)

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
sys.path.insert(0, PATH_TO_SENTEVAL + '/examples/')

import data


def prepare(params, samples):
    _, params.word2id = data.create_dictionary(samples)
    params.word_vec = data.get_wordvec(PATH_TO_VEC, params.word2id)
    # Make sure this matches the embedding size
    params.wvec_dim = emb_size
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
# try kfold=5?
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
#params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                 'tenacity': 3, 'epoch_size': 2}

params_senteval['classifier'] = {'nhid': 0, 'optim': optim, 'batch_size': batch_size,
                                 'tenacity': tenacity, 'epoch_size': epoch_size}

# Set up logger
logFormatter = logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
rootLogger = logging.getLogger()

fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']

    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']


    transfer_tasks = [
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC',
                      'MRPC', 'SICKEntailment', 'STS14']

 
    results = se.eval(transfer_tasks)
    print(results)
