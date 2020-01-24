import time
import pickle 
import gensim
# import pandas as pd
from os import listdir
# from multiprocessing import Pool
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from random import shuffle

import argparse
parser = argparse.ArgumentParser(description='Run LDA on a corpus of tokens.')
parser.add_argument('tokensDir', type=str, help='Input dir for tokens')
parser.add_argument('dictName', type=str, help='Dictionary file name')
parser.add_argument('modelName', type=str, help='Model file name')

args = parser.parse_args()

# ------------- MAIN ---------------

# --------- online lda ------------
print("Loading gensim dict...")
common_dictionary = Dictionary.load(args.dictName)
print("Loaded.")

batchSize = 500
docDir = args.tokensDir
docFiles = listdir(docDir)
shuffle(docFiles) # in place shuffling
batches = [docFiles[i:i + batchSize] for i in range(0, len(docFiles), batchSize)]
timeElapsed = 0

# create starting valeus
print('Creating initial lda model...')
docs = []
for file in batches[0]:
    with open(docDir + file, 'rb') as fp:
        doc = pickle.load(fp)
        docs.append(doc)

start_corpus = [common_dictionary.doc2bow(doc) for doc in docs]
lda = LdaModel(start_corpus, num_topics = 100, id2word = common_dictionary)
print("Created.")

print("Iterating through docs...")
lastTime = time.time()
for batchIndex in range(2, len(batches)+1):

    batch = batches[batchIndex-1]    
    docs = []

    for filename in batch:
        with open(docDir + filename, 'rb') as fp:
            doc = pickle.load(fp)
            docs.append(doc)
            
    other_corpus = [common_dictionary.doc2bow(doc) for doc in docs]
    lda.update(other_corpus)
    
    batchTime = time.time() - lastTime
    timeElapsed += batchTime
    ETA = (timeElapsed/(batchIndex-1)) * (len(batches) - (batchIndex-1))
    ETAstring = "{}:{}:{}".format( int(ETA / 3600), int( (ETA % 3600) / 60 ), int(ETA % 60))

    print("Batch {} of {} | Batch time: {:.3} | ETA: {}".format(batchIndex, len(batches), batchTime, ETAstring))
    lastTime = time.time()
    
lda.save(args.modelName)