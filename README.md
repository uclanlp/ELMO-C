# Efficient Contextual Representation Learning With Continuous Outputs
Thank you for checking out our repository! This repo contains code for the paper [Efficient Contextual Representation Learning With Continuous Outputs](https://transacl.org/ojs/index.php/tacl/article/view/1766). Please contact us if you have any questions!

We provide scripts for training our faster ELMo model, ELMo-C. Evaluating ELMo-C on downstream tasks can be done using AllenNLP and is not included in this repo. Please contact us if you are interested in that.

This repo borrows code from several repositeries, including but not limited to [AllenNLP](https://github.com/allenai/allennlp), [AdaptiveSoftmax](https://github.com/rosinality/adaptive-softmax-pytorch), and [BPE](https://github.com/rsennrich/subword-nmt). I would like to expresse my gratitdue for authors of these repositeries! I also want to acknowledge the help from authors of [Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs](https://arxiv.org/pdf/1812.04616.pdf) for dealing with numerical stability. Thanks!

# Dependencies

## Environments

Pytorch == 0.4.0

AllenNLP <= 0.8.0

[FastText](https://fasttext.cc/)

## Data

The training corpus of ELMo is the One-Billion-Word-Benmark [(Link)](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz). After decompressing, the original training set `training-monolingual.tokenized.shuffled/` has 99 data chunks. We take out the last chunk `news.en-00099-of-00100` as our validation set and move it into `val-monolingual.tokenized.shuffled/`. The vocabulary can be obtained from AllenNLP [(Link)](https://allennlp.s3.amazonaws.com/models/elmo/vocab-2016-09-10.txt).

We also need to get the FastText embedding as our pre-trained word embeddings [(Link)](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip).

After downloading the necessary files, the local data folder should contains: 
```
vocab-2016-09-10.txt
training-monolingual.tokenized.shuffled/
val-monolingual.tokenized.shuffled/
crawl-300d-2M-subword.bin
```

# Code

Assume that the folder `XX` is the parent directory of the code directory and this code directory is named as `elmo_c`. The following command trains our model:
```
export PYTHONPATH=$PYTHONPATH:XX/
cd XX/elmo_c
CUDA_VISIBLE_DEVICES=XXXX python -m elmo_c.stand_alone_scripts.train
```

Please see `config.json` and `stand_alone_scripts/parse_config.py` to specify data and model paths. 

The core model is in `source/models/complete_elmo.py`. The continous output layer and other kinds of output layers are implemented in `source/models/losses.py`. `source/models/custom_elmo_lstm.py`, `source/models/elmo_lstm.py`, and `source/models/elmo_own.py` are implementations of different kinds of sequence encoders (e.g. ELMo's LSTM, LSRM, SRU, QRNN, etc). `source/models/build_word_vectors.py` is responsible for pre-computing word embeddings for all words in a given vocabulary.
