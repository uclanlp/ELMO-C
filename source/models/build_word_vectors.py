from copy import deepcopy
import os
import torch
from collections import Counter
import json
import numpy as np
import random
import logging
from torch.utils.data import Dataset, DataLoader
from timeit import default_timer as timer
import time
import fastText

from ..helpers import *

def create_word_vectors_fasttext_list(word_list, fasttext_model= None, save_path = None):
    word_vectors = np.ndarray((len(word_list), 300), dtype = float)
    if fasttext_model is None:
        fasttext_model = fastText.load_model(input_embedding_model_path)

    unk_vector = fasttext_model.get_word_vector("<unk>")
    for index, i in enumerate(word_list):
        vector = fasttext_model.get_word_vector(i).astype(float)
        if not np.any(vector):
            vector = unk_vector # In case FastText returns a zero vector
        word_vectors[index] = vector
    try:
        if save_path is not None or len(save_path) != 0:
            with open(save_path, "wb") as f:
                np.save(f, word_vectors)
            print("Written data to file {}".format(save_path))
    except:
       pass

    return word_vectors

def create_word_vectors_fasttext_mixed(config, word_list, save_path):
    vec_file = config["other_stuff"]["data_folder"] + config["input_layer"]["vec_file"]
    bin_file = config["other_stuff"]["data_folder"] + config["input_layer"]["bin_file"]
    #print('Loading embedding from text file...')
    word_dict = {}
    word_set = set(word_list)
    with open(vec_file) as f:
        for line in f:
            splited_line = line.split()
            if len(splited_line) < 300:
                continue
            word = " ".join(splited_line[:-300])
            if word in word_set:
                word_dict[word] = np.array([float(i) for i in splited_line[-300:]])
    print("Total {} words, find {} words in word embedding.".format(len(word_list), len(word_dict)))
    #print('Filling in missing embeddings...')        
    fasttext_model = fastText.load_model(bin_file)
    word_vectors = np.ndarray((len(word_list), 300), dtype = float)
    unk_vector = fasttext_model.get_word_vector("<unk>")
    use_unk = config["input_layer"].get("use_unk", False)
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
    if use_unk:
        print("Assigning the rest to <unk>.")
    else:
        print("Rerolve rest by FastText.")
    for index, i in enumerate(word_list):
        vector = word_dict.get(i, None)
        if vector is None:
            if use_unk and i not in special_tokens:
                vector = unk_vector
            else:
                vector = fasttext_model.get_word_vector(i).astype(float)
                if not np.any(vector):
                    vector = unk_vector # In case FastText returns a zero vector
        word_vectors[index] = vector
    try:
        if save_path is not None or len(save_path) != 0:
            with open(save_path, "wb") as f:
                np.save(f, word_vectors)
            #print("Written data to file {}".format(save_path))
    except:
       pass
    return word_vectors


def create_word_vectors_glove(config, word_list, save_path):
    vec_file = config["other_stuff"]["data_folder"] + config["input_layer"]["vec_file"]
    #print('Loading embedding from text file...')
    word_dict = {}
    word_set = set(word_list)
    with open(vec_file) as f:
        for line in f:
            splited_line = line.split()
            if len(splited_line) < 300:
                continue
            word = splited_line[0]
            if word in word_set:
                word_dict[word] = np.array([float(i) for i in splited_line[1:]])
    word_vectors = np.ndarray((len(word_list), 300), dtype = float)
    for index, i in enumerate(word_list):
        vector = word_dict.get(i, None)
        assert(vector)
        word_vectors[index] = vector
    try:
        if save_path is not None or len(save_path) != 0:
            with open(save_path, "wb") as f:
                np.save(f, word_vectors)
            #print("Written data to file {}".format(save_path))
    except:
       pass
    return word_vectors

def create_word_vectors_random(config, word_list, save_path):
    word_vectors = torch.distributions.Normal(0, 1).sample(torch.Size([len(word_list), 300])).numpy()
    #word_vectors = np.random.rand(len(word_list), 300)
    try:
        if save_path is not None or len(save_path) != 0:
            with open(save_path, "wb") as f:
                np.save(f, word_vectors)
            #print("Written data to file {}".format(save_path))
    except:
       pass
    return word_vectors

def create_word_vectors_random_trained(config, word_list, save_path):
    tmp_vocab = Vocab()
    tmp_vocab.load(config["input_layer"]["vocab_cache"])
    with open(config["input_layer"]["embedding_cache"], "rb") as f:
        tmp_embedding = np.load(f)

    word_vectors = np.random.randn(len(word_list), 300)
    
    unk_index = tmp_vocab.w2idx["<unk>"]
    counter = 0.0
    for index, i in enumerate(word_list):
        tmp_id = tmp_vocab.w2idx.get(i, unk_index)
        if tmp_id != unk_index:
            word_vectors[index] = tmp_embedding[tmp_id]
        if tmp_id == unk_index:
            counter += 1
    print("{} percent of random embedding. Loaded from {}".format(counter/ len(word_list), config["input_layer"]["embedding_cache"]))
    return word_vectors

def create_word_vectors_cnn(word_list, config, embedding_model_path, save_path = None, weight_file = None, use_cuda = True):
    
    cnn_model = create_and_load_CNN_model(config, embedding_model_path, weight_file)

    word_vectors = compute_embedding_with_model(cnn_model, word_list, config["input_layer"]["input_size"],  use_cuda)
    try:
        if save_path is not None or save_path != "":
            with open(save_path, "wb") as f:
                np.save(f, word_vectors)
            print("CNN: Written data to file {}".format(save_path))
    except:
        print("Cached Failed!")

    return word_vectors

def compute_embedding_with_model(model, word_list, dim, use_cuda = True):
    word_vectors = np.ndarray((len(word_list), dim), dtype = float)
    character_list = batch_to_ids([word_list])
    assert(character_list.dim() == 3)
    assert(character_list.size(0) == 1)
    assert(character_list.size(1) == len(word_list))
    print("There are {} words to cache.".format(len(word_list)))
    if use_cuda:
        model = model.cuda()
    cache_batch_size = 2000 # Do it in batch
    current = 0 
    counter = 0
    model.eval()
    while True:
        boundary = min(current + cache_batch_size, character_list.size(1))
        tmp_tensor = character_list[:, current:boundary, :]
        if use_cuda:
            tmp_tensor = tmp_tensor.cuda()
        output_embedding = model((tmp_tensor, None), forward_target = None, mask_forward = None, backward_target = None, mask_backward = None, hidden = None, return_hidden = False)
        assert(output_embedding.dim() == 3)
        assert(output_embedding.size(0) == 1)
        assert(output_embedding.size(1) == tmp_tensor.size(1))
        output_embedding = output_embedding.data.cpu().numpy()[0]
        word_vectors[current:boundary, :] = output_embedding
        current += cache_batch_size
        counter += 1
        if counter % 10:
            print(counter * cache_batch_size / len(word_list))
        if boundary == character_list.size(1):
            break
    return word_vectors

def create_and_load_CNN_model(config, model_save_address, weight_file = None):
    
    config_copy = deepcopy(config)
    config_copy["input_layer"]["name"] = "cnn" # Originally it could be "embedding"
    config_copy["input_layer"]["weight_file"] = weight_file
    config_copy["rnn_layer"]["name"] = "none"
    config_copy["rnn_layer"]["bidirectional"] = False
    config_copy["rnn_layer"]['custom_elmo'] = False

    config_copy["other_stuff"]["parallel_rnn_and_last_layer"] = False
    config_copy["other_stuff"]["output_representations"] = True
    config_copy["other_stuff"]["situation"] = 5
    
    from elmo_c.source.models.complete_elmo import InitializeWrapper, TrueRunningModel
    elmo_cnn = InitializeWrapper(
    config = config_copy,
    input_word_vectors = None,
    output_word_vectors = None
    )
    elmo_cnn = TrueRunningModel(config = config_copy, input_layer = elmo_cnn.embedding, rnn = elmo_cnn.rnn, loss = elmo_cnn.loss, additional_linear_layer = elmo_cnn.additional_linear_layer, device = None)

    if not config_copy["input_layer"]["weight_file"]:
        load_state_dict_from_file(elmo_cnn, model_save_address)
    return elmo_cnn

def build_word_vectors_cache(
            embedding_model_path, 
            word_list, 
             
            embedding_type,
            config,

            cache_embedding_path = "", # We do not need to always cache
            weight_file = None, # If we want to use a weight file, need to specify
            fasttext_model = None):

    if os.path.exists(cache_embedding_path): # we have the vectors
        print("Loading from embeddings cached file {}".format(cache_embedding_path))
        with open(cache_embedding_path, "rb") as f:
            data = np.load(f)
        if not config.get("restrict_vocab", False):
            assert(len(word_list) == data.shape[0])
        print("Loaded!")
        return data

    if embedding_type == "cnn":
        return create_word_vectors_cnn(word_list = word_list, config = config, embedding_model_path = embedding_model_path, weight_file = weight_file, save_path = cache_embedding_path)

    if embedding_type == "common_crawl_open":  
        return create_word_vectors_fasttext_list(word_list = word_list, fasttext_model = fasttext_model, save_path = cache_embedding_path)

    if embedding_type == "mixed":
        return create_word_vectors_fasttext_mixed(word_list = word_list, save_path = cache_embedding_path, config = config)

    if embedding_type == "glove":
        return create_word_vectors_glove(word_list = word_list, save_path = cache_embedding_path, config = config)

    if embedding_type == "random":
        return create_word_vectors_random(word_list = word_list, save_path = cache_embedding_path, config = config)

    if embedding_type == "random_trained":
        return create_word_vectors_random_trained(word_list = word_list, save_path = cache_embedding_path, config = config)

    assert(0) # Unsupported Embedding Type
