from Siamese_LSTM_python3.rewarder import Rewarder

rewarder = Rewarder()

target_txt = ["A truly wise man", "A truly wise woman", "A not truly wise man"]
sentence_txt = ["He is smart", "She is smart", "He is not smart"]

semsim_score = rewarder(target_txt, sentence_txt)

