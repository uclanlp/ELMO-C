import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy
import json
from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection
from allennlp.modules.elmo import batch_to_ids
import fastText
from timeit import default_timer as timer

from elmo_c.stand_alone_scripts.parse_config import *
from elmo_c.source.models import *
from elmo_c.source.helpers import *
from elmo_c.source.data import Corpus

import os
##########################################################

# Things needed for training and decoding
global_vocab = Vocab()
global_vocab.add_word("<pad>")
global_vocab.add_word("<s>")
global_vocab.add_word("</s>")
global_vocab.add_word("<unk>")

fasttext_model = None

if config["other_stuff"].get("restrict_vocab", False):
    input_cache_embedding_path = ""
    output_cache_embedding_path = ""
try:
    if (not os.path.exists(input_cache_embedding_path) and config["input_layer"]["name"] == "embedding" and config["input_layer"]["embedding_type"] == "common_crawl_open" and config["input_layer"]["embedding_type"] != "cnn") or config["dataset"]["emb_on_the_fly"]:
        fasttext_model = fastText.load_model(input_embedding_model_path)

    if fasttext_model is None:
        if not os.path.exists(output_cache_embedding_path) and config["output_layer"]["name"] == "semfit" and config["output_layer"]["embedding_type"] != "cnn":
            fasttext_model = fastText.load_model(output_embedding_model_path)
except:
    fasttext_model = None

################ Initalized BERT tokenizer
tokenizer = None

global_vocab = Corpus.create_vocabulary(dictionary = global_vocab, vocabulary_cache_file = cache_vocabulary_file, vocabulary_original_file = vocabulary_file)

if config["other_stuff"].get("restrict_vocab", False):
    global_vocab_idx = global_vocab.idx[:config["other_stuff"].get("restrict_vocab", False)]
    global_vocab = Vocab()
    global_vocab.add_word("<pad>")
    global_vocab.add_word("<s>")
    global_vocab.add_word("</s>")
    global_vocab.add_word("<unk>")
    for i in global_vocab_idx:
        global_vocab.add_word(i)
    
corpus_train = Corpus(train_file_path, global_vocab, fasttext_model = fasttext_model, config = config, batch_size = batch_size, bptt = bptt, input_embedding_model_path = input_embedding_model_path)
corpus_val = Corpus(valid_file_path, global_vocab, fasttext_model = fasttext_model, config = config, batch_size = batch_size, bptt = bptt, input_embedding_model_path = input_embedding_model_path)

###############################################################################
if config["input_layer"]["name"] == "embedding" and config["input_layer"]["embedding_type"] != "bpe":
    #try:
    input_word_vectors = build_word_vectors_cache(
            embedding_model_path = input_embedding_model_path, 
            word_list = global_vocab.idx, 
            embedding_type = config["input_layer"]["embedding_type"],
            config = config,
            cache_embedding_path = input_cache_embedding_path, 
            weight_file = config["input_layer"].get("weight_file", None), # If we want to use a weight file, need to specify
            fasttext_model = fasttext_model
            )
    input_word_vectors = torch.Tensor(input_word_vectors)
    #except:
    #input_word_vectors = None # This is just for testing not tying the word embedding and the softmax embedding
else:
    input_word_vectors = None

if config["output_layer"]["name"] == "semfit":
    if config["output_layer"]["embedding_type"] == config["input_layer"]["embedding_type"]: #The input embedding and output embedding are the same
        output_word_vectors = input_word_vectors
    else:
        output_word_vectors = build_word_vectors_cache(
        embedding_model_path = output_embedding_model_path, 
        word_list = global_vocab.idx, 
        embedding_type = config["output_layer"]["embedding_type"],
        config = config,
        cache_embedding_path = output_cache_embedding_path, 
        weight_file = config["output_layer"].get("weight_file", None), # If we want to use a weight file, need to specify
        fasttext_model = fasttext_model)
        output_word_vectors = torch.Tensor(output_word_vectors)
else:
    output_word_vectors = None

config["output_layer"]["vocab_size"] = len(global_vocab.idx)

print("Creating Models...")
save_elmo = InitializeWrapper(
    config = config,
    input_word_vectors = input_word_vectors,
    output_word_vectors = output_word_vectors
    ) #We keep this model around to save it later

elmo = TrueRunningModel(config = config, input_layer = save_elmo.embedding, rnn = save_elmo.rnn, loss = save_elmo.loss, additional_linear_layer = save_elmo.additional_linear_layer, device = device)

if config["dataset"]["emb_on_the_fly"]:
    corpus_train.elmo = save_elmo
    corpus_val.elmo = save_elmo

if config['other_stuff'].get("average_models_list", None):
    print("#### Avaraging Models!")
    elmo = average_model_memory_efficient(config["other_stuff"]["average_models_list"], config, input_word_vectors)
    print("# Saving models...")
    torch.save(elmo.state_dict(), models_path + model_name)
    print("Done!")
    assert(0)
print("Initialization finished.")
#assert(not output_word_vectors.is_cuda) # No Matter What, the ouput word vectors should be kept on CPU
