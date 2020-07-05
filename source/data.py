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

from .helpers import *
from .models.build_word_vectors import build_word_vectors_cache

from allennlp.modules.elmo import batch_to_ids

class Batch:
    '''
    There are two ways to construct Batch.

    For RNN (LSTM, SRU, QRNN), as we use BPTT, the forward sequences and backward sequences are not the same. So we need to provide both forward and backward sequences and their target.

    For Transformer, we use padding. So all we need to do it provide the raw sentence. Then Batch can disscet it into forward and backward sequence.
    '''
    def __init__(self, 

        vocab_instance = None,
        input_batches_padded = None, 
        target_forward_padded = None, 
        target_backward_padded = None, 
        input_batches_backward = None, 

        raw_data = None,

        config = None,

        gpu_number = -1):

        if raw_data is not None: # This is especially for transformer
            assert(input_batches_backward is None)
            assert(target_forward_padded is None)
            max_len = max([len(i) - 1 for i in raw_data])
            #print("{} {}".format(max_len, len(raw_data)))
            input_batches_padded = [i[:-1] for i in raw_data]
            target_forward_padded = [i[1:] for i in raw_data]

            self.input_batches_padded = (input_batches_padded, input_batches_backward)
            self.target_forward_padded = target_forward_padded
            self.target_backward_padded = target_backward_padded
            
            self.mask_forward = (input_batches_padded > 0).long()
            self.mask_backward = self.mask_forward # In this padde method, mask_forward == mask_backward

        elif input_batches_padded is not None:

            assert(input_batches_backward is not None)
            assert(target_backward_padded is not None)

            '''self.input_batches_word = []
            for i in input_batches_padded.cpu().numpy().tolist():
                self.input_batches_word.append([vocab_instance.idx[j] for j in i])
            self.input_word_backward=[]
            for i in input_batches_backward.cpu().numpy().tolist():
                self.input_word_backward.append([vocab_instance.idx[j] for j in i])'''
            
            self.target_forward_padded = target_forward_padded
            self.target_backward_padded = target_backward_padded

            self.input_batches_padded = (input_batches_padded, input_batches_backward)
            self.mask_forward = (input_batches_padded > 0).long()
            self.mask_backward = (input_batches_backward > 0).long()
        
        if "cnn" in config["input_layer"]["name"]: # only if we use CNN on the input side   
            self.input_batches_word = []
            for i in input_batches_padded.cpu().numpy().tolist():
                self.input_batches_word.append([vocab_instance.idx[j] for j in i])
            self.input_word_backward=[]
            for i in input_batches_backward.cpu().numpy().tolist():
                self.input_word_backward.append([vocab_instance.idx[j] for j in i])

        if config["input_layer"]["name"] == "acc_cnn":
            gpu_number = config['other_stuff']["gpu_number"]
            assert(len(self.input_batches_word) % gpu_number == 0)
            examples_per_gpu = int(len(self.input_batches_word) / gpu_number)
            self.input_batches_word = [self.input_batches_word[i: i + examples_per_gpu] for i in range(0, len(self.input_batches_word), examples_per_gpu)]
            self.input_word_backward = [self.input_word_backward[i: i + examples_per_gpu] for i in range(0, len(self.input_word_backward), examples_per_gpu)]
            new_input_forward = []
            new_input_backward = []
            character_level_inputs = []
            lenth_records = []
            with torch.no_grad():
                for index, i in enumerate(self.input_batches_word):
                    forward_and_backward_sentences, character_level_input, tmp_vocab = self.make_charecter_ids((self.input_batches_word[index], self.input_word_backward[index]))
                    new_input_forward.extend(forward_and_backward_sentences[0])
                    new_input_backward.extend(forward_and_backward_sentences[1])
                    character_level_inputs.append(character_level_input)
                    lenth_records.append(character_level_input.size(0))
                max_len = max(lenth_records)
                self.input_batches_word = torch.LongTensor(new_input_forward)
                self.input_word_backward = torch.LongTensor(new_input_backward)
                self.lenth_records= torch.LongTensor(lenth_records)
                for index,i in enumerate(character_level_inputs):
                    character_level_inputs[index] = torch.cat([i, i.new(max_len - i.size(0), *i.size()[1:]).zero_()])
                self.character_level_inputs = torch.stack(character_level_inputs, dim = 0)
                self.input_batches_padded = (self.input_batches_word, self.input_word_backward, self.character_level_inputs, self.lenth_records)
            return

    @staticmethod
    def make_charecter_ids(forward_and_backward_sentences):
        new_vocab = Vocab()
        for index_sentence, sentences in enumerate(forward_and_backward_sentences):
            for index_i, i in enumerate(sentences):
                for index_j, j in enumerate(i):
                    forward_and_backward_sentences[index_sentence][index_i][index_j] = new_vocab.get_and_add_word(j)
        character_level_input = batch_to_ids([new_vocab.idx])[0]
        assert(character_level_input.size(0) == len(new_vocab.idx))
        return forward_and_backward_sentences, character_level_input, new_vocab

class Corpus(object):
    def __init__(self, path, vocab_instance, fasttext_model, config, bptt, input_embedding_model_path, batch_size = 0, tokenizer = None):

        self.dictionary = vocab_instance
        self.current = -1
        self.path = path
        self.epoch_counter = 0
        self.config = config
        self.name = config["dataset"]["name"]
        self.type = config["dataset"]["type"]
        self.bptt = bptt

        if config["rnn_layer"]["name"] == "sru" or config["rnn_layer"]["name"] == "elmo": 
            self.stretched = True
        else:
            self.stretched = False
        
        self.emb_on_the_fly = config["dataset"]["emb_on_the_fly"]

        ############################ Special processing for 1 Billion benchmark
        # we shuffle sentences within a slice file
        # we randomly choose slice file every time
        # hope this implementation is efficient enough
        self.current = -1

        self.all_files = [j for j in os.listdir(path)]

        self.all_files.sort()

        if config["dataset"].get("partial_dataset", 1) < 1:
            self.all_files = self.all_files[:max(1, int(len(self.all_files) * config["dataset"]["partial_dataset"]))]

        self.fasttext_model = fasttext_model
        if "parallel" in self.type:
            random.shuffle(self.all_files)

            dataset_level = config["dataset"].get("level", "sentence") # Including sentence, document, bert

            if dataset_level == "sentence":
                dataset_cls = PreFetchDataset
            elif dataset_level == "document":
                dataset_cls = PreFetchDatasetDocumentLevel
            elif dataset_level == "bert":
                dataset_cls = PreFetchDatasetBERT
            self.dataset = dataset_cls(path = path, 
                    all_files = self.all_files, 
                    dictionary = self.dictionary, 
                    batch_size = batch_size,
                    fasttext_model = fasttext_model,
                    emb_on_the_fly = self.emb_on_the_fly,
                    config = config,
                    tokenizer = tokenizer,
                    bptt = bptt,
                    input_embedding_model_path = input_embedding_model_path)

            self.data_iterator = torch.utils.data.DataLoader(
                dataset = self.dataset,
                batch_size = 1,
                shuffle = False,
                num_workers=config["dataset"]["num_workers"],
                collate_fn = collate_fn_custom)

            self.fetched_where = -1
            self.iter_dataloader_wrapper = iter(self.data_iterator)

    ###################################################################################### General
    @staticmethod
    def create_vocabulary(dictionary, vocabulary_cache_file, vocabulary_original_file):
        if os.path.exists(vocabulary_cache_file):
            dictionary.load(vocabulary_cache_file)
            print('Loaded dictionary from ' + vocabulary_cache_file)
            assert(dictionary.idx[0] == "<pad>")
            assert(dictionary.idx[1] == "<s>")
            assert(dictionary.idx[2] == "</s>")
            assert(dictionary.idx[3] == "<unk>")
        else:
            print("Creating vocabulary from file...")
            try:
                with open(vocabulary_original_file) as f:
                    for index, i in enumerate(f.read().split("\n")):
                        if index % 10000 == 0:
                            print(index)
                        dictionary.add_word(i.split(" ")[0])

                dictionary.save(vocabulary_cache_file)
                print('Saved dictionary to ' + vocabulary_cache_file)
            except:
                print("Create Vocabulary failed! Moving on... Make sure you turned on Embedding On The Fly!")
        
        print("Vocabulary size: " + str(len(dictionary.idx)))
        return dictionary

    def get_batch(self, batch_size):

        if "padded" in self.type:
            return self.get_batch_padded(batch_size)
        elif self.type == "cc_parallel":
            return self.get_batch_parallel(batch_size)
        else:
            assert(0)

    def get_batch_parallel(self, batch_size):
        bptt = self.bptt
        if self.current == -1:
            try:
                if not self.emb_on_the_fly:
                    (self.stretched_data, self.stretched_data_reversed) = next(self.iter_dataloader_wrapper)
                else:
                    (self.stretched_data, self.stretched_data_reversed, self.word_vectors, self.dictionary) = next(self.iter_dataloader_wrapper)
                self.current = 0
            except StopIteration:
                self.iter_dataloader_wrapper = iter(self.data_iterator)
                if not self.emb_on_the_fly:
                    (self.stretched_data, self.stretched_data_reversed) = next(self.iter_dataloader_wrapper)
                else:
                    (self.stretched_data, self.stretched_data_reversed, self.word_vectors, self.dictionary) = next(self.iter_dataloader_wrapper)
                
                random.shuffle(self.dataset.all_files)
                self.current = 0
                self.epoch_counter += 1

            if self.emb_on_the_fly: # Swap the embeddings so we can achieve on the fly
                self.elmo.replace_embedding(self.word_vectors)

            N = len(self.stretched_data)
            if self.config["dataset"]["pad"]:
                L = ((N-1) // (batch_size * bptt)) * batch_size * bptt
            else:
                L = ((N-1) // (batch_size)) * batch_size
            #print("######### Throwing out {} words!".format((N - L)/N))
            self.input_forward = self.stretched_data[:L].view(batch_size,-1)
            self.forward_target = self.stretched_data[1:L+1].view(batch_size,-1)
            self.N = (self.input_forward.size(1)-1)//bptt + 1
            self.input_backward = self.stretched_data_reversed[:L].view(batch_size,-1)
            self.backward_target = self.stretched_data_reversed[1:L+1].view(batch_size,-1)

        x = self.input_forward[:, self.current*bptt : (self.current+1)*bptt]
        y = self.forward_target[:, self.current*bptt : (self.current+1)*bptt] 
        x_2 = self.input_backward[:, self.current*bptt : (self.current+1)*bptt]
        y_2 = self.backward_target[:, self.current*bptt : (self.current+1)*bptt]
        assert(x.size(0) == batch_size)
        assert(x.size(1) == y.size(1))
        self.current += 1
        if self.current >= self.N:
            self.current = -1
        return Batch(vocab_instance = self.dictionary, input_batches_padded = x, target_forward_padded = y, target_backward_padded = y_2, input_batches_backward = x_2, config = self.config)

class PreFetchDataset(Dataset):
    def __init__(self, path, all_files, dictionary, batch_size, fasttext_model, emb_on_the_fly, config, tokenizer, bptt, input_embedding_model_path):
        super(PreFetchDataset, self).__init__()
        self.path = path
        self.all_files = all_files
        self.batch_size = batch_size
        self.fasttext_model = fasttext_model
        self.dictionary = dictionary
        self.emb_on_the_fly = emb_on_the_fly
        self.config = config
        self.type = config["dataset"]["type"]
        self.bptt = bptt
        ############ Just for bert
        self.tokenizer = tokenizer
        self.bert = config["dataset"].get("bert", False)

        ############ Legacy stuff. Do not touch
        self.screen = config["dataset"].get("screen", False)
        self.pad = config["dataset"].get("pad", False)

        self.insert_sentence_boundry = self.config["dataset"].get("insert_sentence_boundry", False)

        self.new_dictionary = set()
        for i in self.dictionary.idx[:50000]:
            self.new_dictionary.add(i)

        self.special_token_list = ["/", "\\", "_", "`", "~", "!", "#", "@", " $", "%", "^", "*", ":", ";", "-", ".", "..."]

        self.input_embedding_model_path = input_embedding_model_path

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, choosed_slice): 
        # Tensor is shared across processes so we can efficiently and load data
        #print("##### Using File: {}".format(self.all_files[choosed_slice]))
        with open(os.path.join(self.path, self.all_files[choosed_slice])) as f:
            document = f.read()
            '''if "<SPECIAL_BOD>" in document:
                cc = 1
                data = [ [k for k in j.split("\n") if k != ""] for j in document.split("<SPECIAL_BOD>")]
            else:'''
            cc = 0
            data = document.split("\n")
        data = [i for i in data if len(i) != 0]
        random.shuffle(data)

        if not self.emb_on_the_fly:
            data = self.idx_data(data)
        else:
            data, new_dictionary = self.idx_data_vocab(data)
            word_vectors = build_word_vectors_cache(
            embedding_model_path = self.input_embedding_model_path, 
            word_list = new_dictionary.idx, 
            embedding_type = self.config["input_layer"]["embedding_type"],
            config = self.config,
            cache_embedding_path = "", 
            weight_file = self.config["input_layer"].get("weight_file", None), # If we want to use a weight file, need to specify
            fasttext_model = self.fasttext_model
            )
            assert(word_vectors.shape[0] == len(new_dictionary.idx))

            word_vectors = torch.from_numpy(word_vectors).float()

        stretched_data = []
        for i in self.pad_data(data, self.dictionary):
            stretched_data.extend(i)
        N = len(stretched_data)
        stretched_data = np.array(stretched_data)

        # Backward data needs padding in the front
        stretched_data_inverse = []
        for i in self.pad_data_inverse(data, self.dictionary):
            stretched_data_inverse.extend(i)
        assert(len(stretched_data) == len(stretched_data_inverse))
        stretched_data_reversed = np.flip(np.array(stretched_data_inverse),0).copy()
        stretched_data = torch.from_numpy(stretched_data)
        stretched_data_reversed = torch.from_numpy(stretched_data_reversed)
        
        #return_dict = {"stretched_data": stretched_data, "stretched_data_reversed": stretched_data_reversed, "word_vectors": word_vectors, "new_dictionary": new_dictionary}

        if not self.emb_on_the_fly:
            return (stretched_data, stretched_data_reversed)
        else:
            return (stretched_data, stretched_data_reversed, word_vectors, new_dictionary)
    '''def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.corpus_lines - self.num_docs - 1
    '''
    def filter(self, data):
        new_data = []
        counter = 0.0
        speical_counter = 0.0
        data = data.split()
        for i in data:
            if i in self.new_dictionary:
                counter += 1
            if i in self.special_token_list:
                speical_counter += 1
        dot_counter = 0
        for i in data:
            if i == "...":
                return 0
        if counter / len(data) > 0.7 and speical_counter / len(data) < 0.3 and len(data) > 8:
            return 1
        else:
            #print(" ".join(data))
            return 0

    def filter_document(self, data):
        data = " ".join(data).split(".")
        average_len = sum([len(i) for i in data]) / len(data) 
        if average_len < 10:
            #print(".".join(data))
            return False
        data = ".".join(data).split()
        speical_counter = 0.0
        counter = 0.0
        for i in data:
            if i in self.new_dictionary:
                counter += 1
            if i in self.special_token_list:
                speical_counter += 1
        if counter / len(data) > 0.93 and speical_counter / len(data) < 0.1:
            return 1
        else:
            return 0

        return True

    def idx_data(self, data):
        # conver the raw data (words) to idex according to self.dictionary
        unk_default = self.dictionary.w2idx["<unk>"]
        start = self.dictionary.w2idx["<s>"]
        end = self.dictionary.w2idx["</s>"]

        new_data = [ [start] + [self.dictionary.get(j, default = unk_default) for j in i.split()] + [end] for i in data]
        return new_data

    def idx_data_vocab(self, data):
        # create vocab and idx data
        # This is to create embeddings on the fly
        new_dictionary = Vocab()
        new_dictionary.add_word("<pad>")
        new_dictionary.add_word("<s>")
        new_dictionary.add_word("</s>")
        new_dictionary.add_word("<unk>") 

        unk_default = new_dictionary.w2idx["<unk>"]
        start = new_dictionary.w2idx["<s>"]
        end = new_dictionary.w2idx["</s>"]
        new_data = []
        for i in data:
            new_data.append([start] + [new_dictionary.get_and_add_word(j) for j in i.split()] + [end])
        return new_data, new_dictionary

    def pad_data(self, data, dictionary):
        if self.pad:
            returned_data = []
            for i in data:
                if len(i) % self.bptt != 0:
                    i = i + [dictionary.w2idx["<pad>"]] * (self.bptt - len(i) % self.bptt)
                returned_data.append(i)
            return returned_data
        else:
            return data

    def pad_data_inverse(self, data, dictionary):
        if self.pad:
            returned_data = []
            for i in data:
                if len(i) % self.bptt != 0:
                    i = [dictionary.w2idx["<pad>"]] * (self.bptt - len(i) % self.bptt) + i
                returned_data.append(i)
            return returned_data
        else:
            return data

class PreFetchDatasetBERT(PreFetchDataset):
    def __init__(self, **kwgs):
        super(PreFetchDatasetBERT, self).__init__(**kwgs)

    def __getitem__(self, choosed_slice): 
        # Tensor is shared across processes so we can efficiently and load data
        #print("##### Using File: {}".format(self.all_files[choosed_slice]))
        with open(os.path.join(self.path, self.all_files[choosed_slice])) as f:
            document = f.read()
            '''if "<SPECIAL_BOD>" in document:
                cc = 1
                data = [ [k for k in j.split("\n") if k != ""] for j in document.split("<SPECIAL_BOD>")]
            else:'''
            cc = 0
            data = document.split("\n")
        data = [i for i in data if len(i) != 0]
        random.shuffle(data)

        if self.emb_on_the_fly:
            self.tokenizer.reset_self_vocabulary()

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = [] # map sample index to doc and line

        with open(os.path.join(self.path, self.all_files[choosed_slice]), encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line == "":
                    self.all_docs.append(doc)
                    doc = []
                    #remove last added sample because there won't be a subsequent line anymore in the doc
                    self.sample_to_doc.pop()
                else:
                    #store as one sample
                    sample = {"doc_id": len(self.all_docs),
                              "line": len(doc)}
                    self.sample_to_doc.append(sample)
                    doc.append(line)
                    self.corpus_lines = self.corpus_lines + 1

        # if last row in file is not empty
        if self.all_docs[-1] != doc:
            self.all_docs.append(doc)
            self.sample_to_doc.pop()
        self.num_docs = len(self.all_docs)

        self.random_file = open(os.path.join(self.path, self.all_files[random.randrange(len(self.all_files))]), "r", encoding=encoding)

        return_list = []
        while item < len(self):
            cur_id = self.sample_counter
            self.sample_counter += 1

            t1, t2, is_next_label = self.random_sent(item)
            item += 1

            # tokenize
            tokens_a = self.tokenizer.tokenize(t1)
            tokens_b = self.tokenizer.tokenize(t2)

            # combine to one sample
            cur_example = InputExample(guid=cur_id, tokens_a=tokens_a, tokens_b=tokens_b, is_next=is_next_label)

            # transform sample to features
            cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)
            return_list.append(cur_features)

        input_ids = torch.stack([torch.tensor(i.input_ids) for  i in return_list], dim = 0)
        input_mask = torch.stack([torch.tensor(i.input_mask) for  i in return_list], dim = 0)
        segment_ids = torch.stack([torch.tensor(i.segment_ids) for  i in return_list], dim = 0)
        lm_label_ids = torch.stack([torch.tensor(i.lm_label_ids) for  i in return_list], dim = 0)
        is_next = torch.stack([torch.tensor(i.is_next) for  i in return_list], dim = 0)
        
        if self.emb_on_the_fly:
            new_dictionary = self.tokenizer.vocab
            word_vectors = build_word_vectors_cache(
                embedding_model_path = self.input_embedding_model_path, 
                word_list = new_dictionary.idx, 
                embedding_type = self.config["input_layer"]["embedding_type"],
                config = self.config,
                cache_embedding_path = "", 
                weight_file = self.config["input_layer"].get("weight_file", None), # If we want to use a weight file, need to specify
                fasttext_model = self.fasttext_model
                )
            assert(word_vectors.shape[0] == len(new_dictionary.idx))
            self.tokenizer.reset_self_vocabulary()



        returned_dict = {"input_ids": input_ids,
                       "input_mask": input_mask,
                       "segment_ids": segment_ids,
                       "lm_label_ids": lm_label_ids,
                       "is_next": is_next}
    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            label = 0
        else:
            t2 = self.get_random_line()
            label = 1

        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, label

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert item < self.corpus_lines
        
        sample = self.sample_to_doc[item]
        t1 = self.all_docs[sample["doc_id"]][sample["line"]]
        t2 = self.all_docs[sample["doc_id"]][sample["line"]+1]
        # used later to avoid random nextSentence from same doc
        self.current_doc = sample["doc_id"]
        return t1, t2

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.
        for _ in range(10):
            if self.on_memory:
                rand_doc_idx = random.randint(0, len(self.all_docs)-1)
                rand_doc = self.all_docs[rand_doc_idx]
                line = rand_doc[random.randrange(len(rand_doc))]
            else:
                rand_index = random.randint(1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                #pick random line
                for _ in range(rand_index):
                    line = self.get_next_line_from_random_file()
            #check if our picked random line is really from another doc like we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def get_next_line_from_random_file(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = self.random_file.__next__().strip()
            #keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = self.random_file.__next__().strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(os.path.join(self.path, self.all_files[random.randrange(len(self.all_files))]), "r", encoding=encoding)
            line = self.random_file.__next__().strip()
        return line

class PreFetchDatasetDocumentLevel(PreFetchDataset):
    def __init__(self, **kwgs):
        super(PreFetchDatasetDocumentLevel, self).__init__(**kwgs)

    def __getitem__(self, choosed_slice): 
        # Tensor is shared across processes so we can efficiently and load data
        #print("##### Using File: {}".format(self.all_files[choosed_slice]))
        data = []
        one_document = ""
        with open(os.path.join(self.path, self.all_files[choosed_slice])) as f:
            for line in f:
                if len(line.strip()) == 0 and len(one_document) > 0:
                    data.append(one_document)
                    one_document = ""
                else:
                    one_document += line
        random.shuffle(data)

        if not self.emb_on_the_fly:
            data = self.idx_data(data)
        else:
            data, new_dictionary = self.idx_data_vocab(data)
            word_vectors = build_word_vectors_cache(
            embedding_model_path = self.input_embedding_model_path, 
            word_list = new_dictionary.idx, 
            embedding_type = self.config["input_layer"]["embedding_type"],
            config = self.config,
            cache_embedding_path = "", 
            weight_file = self.config["input_layer"].get("weight_file", None), # If we want to use a weight file, need to specify
            fasttext_model = self.fasttext_model
            )
            assert(word_vectors.shape[0] == len(new_dictionary.idx))

            word_vectors = torch.from_numpy(word_vectors).float()

        stretched_data = []
        for i in self.pad_data(data, self.dictionary):
            stretched_data.extend(i)
        stretched_data = np.array(stretched_data)
        # Backward data needs padding in the front
        stretched_data_inverse = []
        for i in self.pad_data_inverse(data, self.dictionary):
            stretched_data_inverse.extend(i)
        assert(len(stretched_data) == len(stretched_data_inverse))
        stretched_data_reversed = np.flip(np.array(stretched_data_inverse),0).copy()
        stretched_data = torch.from_numpy(stretched_data)
        stretched_data_reversed = torch.from_numpy(stretched_data_reversed)
        #return_dict = {"stretched_data": stretched_data, "stretched_data_reversed": stretched_data_reversed, "word_vectors": word_vectors, "new_dictionary": new_dictionary}

        if not self.emb_on_the_fly:
            return (stretched_data, stretched_data_reversed)
        else:
            return (stretched_data, stretched_data_reversed, word_vectors, new_dictionary)

def collate_fn_custom(batch):
    batch = batch[0]
    return batch
