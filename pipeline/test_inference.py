#test inferene, test different models and generate results

import torch
from fairseq.models.bart import BARTModel
path = '/home/ubuntu/project/semsim/fairseq-semsim/'
pathbart = '/home/ubuntu/project/semsim/fairseq-semsim/pretrained/bart.large.cnn/'
bart = BARTModel.from_pretrained(
    #pathbart,
    path + 'checkpoints/',
    #checkpoint_file='model.pt',
    #checkpoint_file='semsim.pt',
    checkpoint_file='checkpoint_best0.pt',
    data_name_or_path=path+'cnn_dm-bin'
)

bart.cuda()
bart.eval()
bart.half()
count = 1
bsz = 32 # for 12GB GPU memory
with open(path+'cnn_dm/test.source') as source, open(path+'cnn_dm/test_new_rewarder.hypo', 'w') as fout:
    sline = source.readline().strip()
    slines = [sline]
    for sline in source:
        if count > 0:
            if count % bsz == 0:
            	print(count,'processing')
            	with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)

            	for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
            	slines = []

            slines.append(sline.strip())
        count += 1
    if slines != []:
        hypotheses_batch = bart.sample(slines, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for hypothesis in hypotheses_batch:
            fout.write(hypothesis + '\n')
            fout.flush()

