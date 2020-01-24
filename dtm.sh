DTM_DICT='./trimmed_split_vard.dct'
DTM_DIR='./vard_split_dtm/'
DTM_MODEL='./vard_split_dtm/dtm.model'
TOKENS_DIR='./vard_split_tokens/'

# execute the dtm script
python dtm.py $TOKENS_DIR $DTM_DIR $DTM_DICT $DTM_MODEL --subsample 1000
echo 'Finished DTM.'
