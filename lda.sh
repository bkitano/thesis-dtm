# set path names
TOKENS_DIR='./vard_split_tokens/'
TOKENS_DICT='./trimmed_split_vard.dct'
MODEL_DIR='./vard_trimmed_split_lda.model'

python online_lda.py $TOKENS_DIR $TOKENS_DICT $MODEL_DIR
echo 'Finished LDA'