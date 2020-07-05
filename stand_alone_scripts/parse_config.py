import json
import os
import torch
import math
import sys
import commentjson

with open("./config.json") as f:
    config = commentjson.load(f)

check_epoch = config["other_stuff"]["check_epoch"]

save_epoch = config["other_stuff"].get("save_epoch", -1)

save_epochs = config["other_stuff"].get("save_epochs", [])


clip = config["other_stuff"]["clip"]

train_iter = config["other_stuff"].get("train_iter", -1)

use_logging = config["other_stuff"].get("use_logging", True)

if "test" in config["other_stuff"]["model_name"]:
    use_logging = False

log_path = config["other_stuff"]["log_path"]

cache_suffix = "_cache"
data_folder = config["other_stuff"]["data_folder"]
models_path = config["other_stuff"]["models_path"]

vocabulary_file = data_folder + config["dataset"]["vocabulary_file"]
cache_vocabulary_file = vocabulary_file + cache_suffix

input_embedding_model_path = data_folder + config["input_layer"]["embedding_model_path"]
output_embedding_model_path = data_folder + config["output_layer"]["embedding_model_path"]

train_file_path = data_folder + config["dataset"]["train_file_path"]
valid_file_path = data_folder + config["dataset"]["valid_file_path"]
try:
    test_file_path = data_folder + config["dataset"]["test_file_path"]
except:
    print("Did not specify test file path!")

try:
    dataset_cache_folder = data_folder + config["dataset"]["dataset_cache_folder"]
except:
    print("Did not specify dataset cache folder!")

if config["input_layer"].get("cache_embedding_path", None):
    input_cache_embedding_path = data_folder + config["input_layer"]["cache_embedding_path"]
else:
    input_cache_embedding_path = data_folder + "{}_{}{}".format(config["input_layer"]["embedding_model_path"], config["dataset"]["name"], cache_suffix)
    print("Did not specify cache embedding path. Automatic resolve: {}".format(input_cache_embedding_path))
   
if config["output_layer"].get("cache_embedding_path", None):
    output_cache_embedding_path = data_folder + config["output_layer"]["cache_embedding_path"]
else:
    output_cache_embedding_path = data_folder + "{}_{}{}".format(config["output_layer"]["embedding_model_path"], config["dataset"]["name"], cache_suffix)
    print("Did not specify cache embedding path. Automatic resolve: {}".format(output_cache_embedding_path))

################################# Some global variable related to training

print_every = config["other_stuff"]["print_every"]

bidirectional = config["rnn_layer"]["bidirectional"]

dataset_name = config["dataset"]["name"]

batch_size = config["dataset"]["batch_size"]

test_size = batch_size

bptt = config["dataset"]["bptt"]

patience = config["other_stuff"].get("patience", math.inf)

if config["optimizer"]["name"] == "scheduled" and "adam" in config["optimizer"]["type"]:

    start_decay = int(400000000 / batch_size * config["optimizer"]["decay_ratio"]) 

    end_decay = int(400000000 / batch_size * config["optimizer"]["end_decay_ratio"])

    base_ratio = config["optimizer"]["base_ratio"]

    lr_scale = base_ratio * batch_size / 128

    base_scale = config["optimizer"]["base_scale"]

    warmup = config["optimizer"]["warmup"]

model_name = config["other_stuff"]["model_name"]#This name only includes the model's original name
model_save_name = model_name + "seq2seq_parameters" + dataset_name + ".torch" # This include dataset name, and sufix

check_scale = config["other_stuff"].get("check_scale", 1)
check_cosine_interval = max(int(2000000 / batch_size / check_scale), 100)
monitor_interval = 49

##################### things that do not need changing
try:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    using_py03 = 0
except:
    print("### Using old version of pytorch!")
    device = None
    using_py03 = 1

########### For speed test
no_update = config["other_stuff"].get("no_update", False)
