import re
import os
import time
import pickle
import numpy as np
from os import listdir
from gensim import corpora, utils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models.wrappers.dtmmodel import DtmModel
from gensim.corpora import Dictionary
import random

import argparse
parser = argparse.ArgumentParser(description='Run DTM on a corpus of tokens.')
parser.add_argument('tokensDir', type=str, help='Input dir for tokens')
parser.add_argument('tempDir', type=str, help='Output dir for temp files')
parser.add_argument('dictionary', type=str, help='Input dictionary file name')
parser.add_argument('modelName', type=str, help='Model file name')
parser.add_argument('--subsample', type=int, help='optional parameter to sample the corpus')
parser.add_argument('--no_topics', type=int, help='number of topics to learn')

args = parser.parse_args()

class DTMcorpus(corpora.textcorpus.TextCorpus):

    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)

textPath = args.tokensDir
tokenFiles = listdir(textPath)
if args.subsample:
    tokenFiles = random.sample(tokenFiles, args.subsample)

print(len(tokenFiles))

def parseYear(filename):
    year = filename.split('-')[0]
    year = re.sub('\D', '', year)
    return int(year)

fileYearPairs = [(t, parseYear(t)) for t in tokenFiles if parseYear(t) > 1400]
yearMin = min([fyp[1] for fyp in fileYearPairs])
yearMax = max([fyp[1] for fyp in fileYearPairs])
interval = 20
fyp = sorted(fileYearPairs, key = lambda x: x[1])
print(len(fyp))

# eta logic
batchSize = 100
batches = [fyp[i:i + batchSize] for i in range(0, len(fyp), batchSize)]
timeElapsed = 0
lastTime = time.time()

documents = []
time_slices = [0 for i in range(yearMin, yearMax, interval)]
sliceYear = yearMin
_slice = 0

print("Starting corpus building...")
for batchIndex in range(1, len(batches)+1):
    batch = batches[batchIndex-1]
    
    for p in batch:
        filename = p[0]
        year = p[1]
        
        with open(textPath + filename, 'rb') as fp:
            documents.append(pickle.load(fp))

        if year - sliceYear < interval:
            time_slices[_slice] += 1

        else:
            _slice += 1
            time_slices[_slice] += 1
            sliceYear += 20

    batchTime = time.time() - lastTime
    timeElapsed += batchTime
    ETA = (timeElapsed/batchIndex) * (len(batches) - batchIndex)
    ETAstring = "{}:{}:{}".format( int(ETA / 3600), int( (ETA % 3600) / 60 ), int(ETA % 60))

    print("Batch {} of {} | Batch time: {} | ETA: {}".format(batchIndex, len(batches), batchTime, ETAstring))
    lastTime = time.time()

print(time_slices)

dictionary = Dictionary.load(args.dictionary)

corpus = DTMcorpus(documents, dictionary)

no_topics = 150
if args.no_topics:
    no_topics = args.no_topics

dtm_path = './dtm-linux64'
model = DtmModel(dtm_path, 
                 corpus, 
                 time_slices, 
                 num_topics=no_topics, 
                 id2word=dictionary, 
                 prefix=args.tempDir)
model.save(args.modelName)