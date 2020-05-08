git clone https://github.com/icml-2020-nlp/semsim
cd /semsim/fairseq-semsim
pip install --editable .

#install gdown
pip install gdown

#download data
cd cnn_dm
gdown https://drive.google.com/uc?id=1goxgX-0_2Jo7cNAFrsb9BJTsrH7re6by
tar -czf cnn_dm.tar.gz .


#download model
cd ..
gdown https://drive.google.com/uc?id=1CNgK6ZkaqUD239h_6GkLmfUOGgryc2v9
