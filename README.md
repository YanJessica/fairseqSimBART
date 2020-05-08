# Introduction

SimBART is a text summarization model that improves BART using own-created similarity score. The model is built on May 2020 version of [Fairseq](https://github.com/pytorch/fairseq) 
and May 2020 version of [semsim](https://github.com/icml-2020-nlp/semsim/tree/d0171cb2a430d8ecf6dd822584e2a78d15016800).


## Features:

Compared with BART, Our improvements are:

- Implement the Similarity score as a **loss measure**, called SimLoss, see [semantic_similarity_loss](fairseq/criterions/semantic_similarity_loss.py). 
- Implement the Similarity score as a **model evaluation metric**, see [calculateSemScore](cnn_dm/calculateSemScore.py)

Our project focuses on text summarization, which refers to generating a shortsummary consisting of a few sentences that capture salient ideas of a text.  The mostly used maximum-likelihood method which minimizes cross-entropy loss, maynot be appropriate for summarization task since such a loss is too strict to accountmultiple valid answers. We fix this problem by learning with document similarityloss. However, the computation of similarity score is non-differentiable with respectto the parameter in the summarization model. Therefore, our key achievementis to address this issue by introducing a two-stage similarity computation that is differentiable and can affect the updates of parameter in the BART model. 

For more detailed information, please refer to our [documentation](SimBART.pdf)

## Requirements and Pipline

* [PyTorch](http://pytorch.org/) version >= 1.2.0
* Python version >= 3.5
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

[start_code](start_code.sh) provides data preparation methods. 

[Model Pipeline](cnn_dm/pipeline) shows piplines to:
- Step1: BPE encoding for dataset
- Step2: Binarize dataset for Fairseq preprocessing
- Step3: Train model
- Step4: Test model


