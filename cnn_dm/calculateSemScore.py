import numpy as np
model_path = '/home/ubuntu/project/semsim/fairseq-semsim/cnn_dm/small_cnn_dm/test_small_new_rewarder.hypo'
#model_path = '/home/ubuntu/project/semsim/fairseq-semsim/cnn_dm/small_cnn_dm/output_jiayin'
target_path = '/home/ubuntu/project/semsim/fairseq-semsim/cnn_dm/test_small.target'
#target_path = '/home/ubuntu/project/semsim/fairseq-semsim/cnn_dm/target_jiayin'
model_output = []
with open(model_path) as f:
    for i in f.readlines():
        model_output.append(i.strip())


target_output = []
with open(target_path) as f:
    for i in f.readlines():
        target_output.append(i.strip())

assert len(model_output) == len(target_output)

from newLoss.Siamese_LSTM_python3 import rewarder
my_rewarder = rewarder.Rewarder()

#compare
scores = np.array([my_rewarder(model_output[i], target_output[i]) for i in range(len(model_output))])
np.save('compare.npy',scores)

print('model:', model_path)
print('mean scores:')
print(scores.mean())
print(scores.std())
print(scores.max())
print(scores.min())    
