#train models on different pretrained weights

#BART_PATH=/pretrained/bart.large.cnn/model.pt 
BART_PATH=/home/ubuntu/project/semsim/fairseq-semsim/checkpoints/semsim.pt
TOTAL_NUM_UPDATES=50000  
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=1024 # for gpu 16gb
UPDATE_FREQ=32

python train.py cnn_dm-bin_medium \
    --no-epoch-checkpoints \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion semantic_similarity_loss \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999 )" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --save-dir checkpoints/new_rewarder_model \
    --find-unused-parameters;
