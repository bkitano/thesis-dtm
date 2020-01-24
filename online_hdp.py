import time
import pickle 
import gensim
from os import listdir
from gensim.corpora.dictionary import Dictionary
from gensim.models import HdpModel
from random import shuffle
import argparse

parser = argparse.ArgumentParser(description='Run online HDP on a directory of tokens.')
parser.add_argument('corpusDir', type=str, help='Input directory for the corpus.' )
parser.add_argument('dictName', type=str, help='Input dictionary.' )
parser.add_argument('modelName', type=str, help='Output model name.')
args = parser.parse_args()

# ------------- MAIN ---------------

# --------- online lda ------------
print("Loading gensim dict...")
common_dictionary = Dictionary.load(args.dictName)
print("Loaded.")

batchSize = 500
docDir = args.corpusDir
docFiles = listdir(docDir)
shuffle(docFiles) # in place shuffling
batches = [docFiles[i:i + batchSize] for i in range(0, len(docFiles), batchSize)]
timeElapsed = 0

# create starting valeus
print('Creating initial hdp model...')
docs = []
for file in batches[0]:
    with open(docDir + file, 'rb') as fp:
        doc = pickle.load(fp)
        docs.append(doc)

start_corpus = [common_dictionary.doc2bow(doc) for doc in docs]
hdp = HdpModel(start_corpus, id2word = common_dictionary)
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
    print("length of other_corpus: {}".format(len(other_corpus)))

    hdp.update(other_corpus)
    print("m_num: {}".format(hdp.m_num_docs_processed))
    
    batchTime = time.time() - lastTime
    timeElapsed += batchTime
    ETA = (timeElapsed/(batchIndex-1)) * (len(batches) - (batchIndex-1))
    ETAstring = "{}:{}:{}".format( int(ETA / 3600), int( (ETA % 3600) / 60 ), int(ETA % 60))

    print("Batch {} of {} | Batch time: {:.3} | ETA: {}".format(batchIndex, len(batches), batchTime, ETAstring))
    lastTime = time.time()
    
hdp.save(args.modelName)