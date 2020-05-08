#binarize preprocessing
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "cnn_dm/small_cnn_dm/train_small.bpe" \
  --validpref "cnn_dm/small_cnn_dm/val_small.bpe" \
  --destdir "cnn_dm-bin_small/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
