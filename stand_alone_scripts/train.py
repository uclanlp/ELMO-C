if __name__ == "__main__":
    from elmo_c.stand_alone_scripts.parse_config import *
    from elmo_c.source.models import *
    from elmo_c.source.helpers import *

    import numpy
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from torch import optim
    import torch.nn.functional as F
    from torch.nn import DataParallel

    from allennlp.modules.lstm_cell_with_projection import LstmCellWithProjection
    from allennlp.modules.elmo import batch_to_ids

    import json
    import sys
    import math
    from timeit import default_timer as timer
    import time
    from datetime import datetime
    ##########################################################
    if use_logging:
        file_log = open(log_path + config["other_stuff"]["model_name"] + "_" + config["dataset"]["name"] + '.log','w')  # File where you need to keep the logs
        file_log.write("")

        class Unbuffered:
           def __init__(self, stream):
               self.stream = stream
           def write(self, data):
               self.stream.write(data)
               self.stream.flush()
               file_log.write(data)    # Write the data of stdout here to a text file as well

        sys.stdout = Unbuffered(sys.stdout)
        #sys.stdout = file_log

    start_time = str(datetime.now())
    print(start_time)

    with open("config.json") as f:
       print(f.read())
    
    print("######################################################################################################")
    ##########################################################
    class LossCalculate():
        def __init__(self, model):
            if config["rnn_layer"]["name"] == "elmo" and config["rnn_layer"].get("custom_elmo", False):
                if config["rnn_layer"]["bidirectional"]:
                    self.hidden = (None, None)
                else:
                    self.hidden = None
            else:
                self.hidden = None

            self.model = model
            if self.hidden is not None or config["rnn_layer"].get("custom_elmo", False):
                self.return_hidden = True
            else:
                self.return_hidden = False

        def __call__(self, batch):
            if config["input_layer"]["name"] == "cnn" and config["rnn_layer"]["name"] != "elmo":
                input = batch_to_ids(batch.input_batches_word)
                if bidirectional:
                    input_backward = batch_to_ids(batch.input_word_backward)
                else:
                    input_backward = None
                input = (input, input_backward)
            elif config["input_layer"]["name"] == "cnn" and config["rnn_layer"]["name"] == "elmo":
                input = batch_to_ids(batch.input_batches_word)
            else:
                input = batch.input_batches_padded

            returned_dict = self.model(
            input = input,
            forward_target = batch.target_forward_padded.contiguous(), 
            mask_forward = batch.mask_forward, 
            backward_target = batch.target_backward_padded.contiguous(),
            mask_backward = batch.mask_backward,
            hidden = self.hidden,
            return_hidden = self.return_hidden)

            self.hidden = returned_dict["hidden"]
            return returned_dict["forward_loss"], returned_dict["backward_loss"]

    class Trainer(): # This is a legacy class. It was used to do profiling. Now it's just an empty class, But we keep it to be clear
        def __init__(self, default_timer_profile):
            return

        #@do_profile(follow=[GeneralRNN.forward, RNNLastLayerWrapper.forward])
        def train(self, batch, model, all_optimizer):
            all_optimizer.zero_grad()
            try:
                elmo_optimizer.zero_grad()
            except:
                pass

            loss_forward, loss_backward = loss_calculate(batch)

            unit_ones = torch.ones(loss_forward.size(0)).to(device)

            loss_forward = loss_forward.sum() / unit_ones.float().sum()

            if config["rnn_layer"]["bidirectional"]:
                loss_backward = loss_backward.sum()/unit_ones.float().sum()
                loss = loss_forward + loss_backward
            else:
                loss = loss_forward
            
            loss.backward()

            if use_adaptive_gradient:
                if adaptive_gradient_clipper(get_parameters_for_optimizer(model)):
                    all_optimizer.step()
                else:
                    pass
            else:
                if clip != -1:
                    #torch.nn.utils.clip_grad_norm_(get_parameters_for_optimizer(model), clip)
                    sparse_clip_norm(get_parameters_for_optimizer(model), clip)
                all_optimizer.step()
                try:
                    elmo_optimizer.step()
                except:
                    pass

            return (loss_forward, loss_backward)

    def eval_model(val_corpus, model, val_batch_size, percentage = 1.0):
        model.eval()
        loss_forward_ = 0.0
        loss_backward_ = 0.0
        iter_number = 0
        
        if config["rnn_layer"]["name"] == "sru": 
            loss_calculate.hidden = torch.zeros(batch_size, config["rnn_layer"]["num_layers"], config["rnn_layer"]["hidden_size"] * 2)
        elif config["rnn_layer"]["name"] == "elmo" and config["rnn_layer"].get("custom_elmo", False):
            if config["rnn_layer"]["bidirectional"]:
                loss_calculate.hidden = (None, None)
            else:
                loss_calculate.hidden = None
        else:
            loss_calculate.hidden = None

        for i in val_corpus.all_files:
            while True:
                one_batch = val_corpus.get_batch(val_batch_size)
                if one_batch is None:
                    break
                with torch.no_grad():

                    loss_forward, loss_backward = loss_calculate_val(one_batch)
                    loss_forward_ += loss_forward.mean().item()
                    if bidirectional:
                        loss_backward_ += loss_backward.mean().item()
                iter_number += 1

                if val_corpus.current == -1:
                    break


        if config["rnn_layer"]["name"] == "sru": 
            loss_calculate.hidden = torch.zeros(batch_size, config["rnn_layer"]["num_layers"], config["rnn_layer"]["hidden_size"] * 2)
        elif config["rnn_layer"]["name"] == "elmo" and config["rnn_layer"].get("custom_elmo", False):
            if config["rnn_layer"]["bidirectional"]:
                loss_calculate.hidden = (None, None)
            else:
                loss_calculate.hidden = None
        else:
            loss_calculate.hidden = None

        loss_forward_ = loss_forward_ / iter_number
        loss_backward_ = loss_backward_ /iter_number
         
        print("Forward loss:  {:.6f}, perplexity {:.6f};\nBackward loss: {:.6f}, perplexity {:.6f}; ".format(loss_forward_, math.exp(loss_forward_), loss_backward_, math.exp(loss_backward_)))

        model.train()
        return loss_forward_, loss_backward_

    ##########################################################

    from elmo_c.stand_alone_scripts.initialize import *

    ##########################################################
    if config["other_stuff"].get("continue_train", False):
        full_state_dict_loaded = torch.load(models_path + config["other_stuff"]["continue_model_name"] )
        load_my_state_dict(elmo, full_state_dict_loaded["model_state"])

    elmo.train()
    loss_calculate = LossCalculate(elmo)
    loss_calculate_val = loss_calculate
    print("There are " + str(parameters_count(elmo)) + " parameters.")


    ##########################################################
    # initilize optimier
    learning_rate = config["optimizer"].get("learning_rate", 0)
    weight_decay = config["optimizer"].get("weight_decay", 0)

    if config["optimizer"]["name"] == "adam":
        all_optimizer = optim.Adam(get_parameters_for_optimizer(elmo), lr=learning_rate, weight_decay = weight_decay)
    elif config["optimizer"]["name"] == "adagrad":
        all_optimizer = optim.Adagrad(get_parameters_for_optimizer(elmo), lr=learning_rate, weight_decay = weight_decay)
    elif config["optimizer"]["name"] == "scheduled":
        if config["optimizer"]["type"] == "adam":
            all_optimizer = NoamOpt_ADAM(
                model_replica = lr_scale,
                start_decay = start_decay, warmup = warmup, optimizer = torch.optim.Adam(get_parameters_for_optimizer(elmo), 
                    lr=0, betas=(0.9, 0.999), eps=1e-6, weight_decay = weight_decay), end_decay = end_decay,
                base_scale = config["optimizer"]["base_scale"]
                )
        elif config["optimizer"]["type"] == "sparse_adam":
            try:
                elmo_optimizer = NoamOpt_ADAM(
                    model_replica = lr_scale,
                    start_decay = start_decay, warmup = warmup, optimizer = torch.optim.SparseAdam(get_parameters_for_optimizer(save_elmo.loss), 
                        lr=0.0001, betas=(0.9, 0.999), eps=1e-6), end_decay = end_decay,
                    base_scale = config["optimizer"]["base_scale"]
                )
            except:
                pass
            all_optimizer = NoamOpt_ADAM(
                model_replica = lr_scale,
                start_decay = start_decay, warmup = warmup, optimizer = torch.optim.Adam(get_parameters_for_optimizer(save_elmo.rnn), 
                    lr=0.0001, betas=(0.9, 0.999), eps=1e-6), end_decay = end_decay,
                base_scale = config["optimizer"]["base_scale"]
            )
        elif config["optimizer"]["type"] == "transformer":
            all_optimizer = NoamOpt(
                config["optimizer"]["size"], 
                config["optimizer"]["transformer_lr_scale"], 
                config["optimizer"]["warmup"],
                torch.optim.Adam(get_parameters_for_optimizer(elmo), lr=0, betas=(0.9, 0.98), 
                eps=1e-9),
                weight_decay = 0.999)
        else:
            assert(0)

    elif config["optimizer"]["name"] == "adagrad":
        all_optimizer = torch.optim.Adagrad(get_parameters_for_optimizer(elmo), lr = learning_rate, lr_decay=0, weight_decay=0, initial_accumulator_value=1.0)

    if config["other_stuff"].get("adaptive_clipper_window", -1) != -1:
        use_adaptive_gradient = True
        adaptive_gradient_clipper = AdaptiveGradientClipper(config["other_stuff"]["adaptive_clipper_window"], get_parameters_for_optimizer(elmo))
    else:
        use_adaptive_gradient = False

    ##########################################################
    trainer = Trainer(False)
    iter_ = 1
    time = 0.0
    loss_record_forward = 0.0
    loss_record_backward = 0.0
    last_loss_forward = 10000000.0
    epoch = 0
    no_update_counter = 0
    s_all = timer()

    global one_batch
    one_batch = corpus_train.get_batch(batch_size)

    iter_tmp = 0
    current_best_loss = 100000
    current_best_acc = -100000
    fail_time = 0
    tmp_time = 0.0
    print_every_counter = 0
    train_time_iter = 0
    previous_epoch_counter = 0

    ##########################################################

    if config["other_stuff"]["continue_train"]:
        if not config["other_stuff"].get("continue_softmax_from_emb", False):
            epoch = full_state_dict_loaded["epoch"]
            try:
                all_optimizer.load_state_dict(full_state_dict_loaded["optimizer_state"])
                print("######### Optimizer Loaded!")
            except:
                print("######### Optimizer Loading Failed!")
            all_optimizer._step = full_state_dict_loaded["optimizer_step"]
            iter_ = all_optimizer._step

            if config["rnn_layer"]["bidirectional"]:
                loss_cri._timer = iter_ * 2
            else:
                loss_cri._timer = iter_

            print("####### Fully resumed training: optimizer_step: {} , epoch: {} , train_iter: {} current learning rate {} ".format(all_optimizer._step, epoch, iter_, all_optimizer.rate()))

    data_reading_time = 0.0

    if config["other_stuff"].get("just_evaluation", False):
        eval_model(corpus_val, elmo, test_size, 1.0)
        assert(0)

    while True:
        s = timer()
        if train_iter > 0 and iter_ >= train_iter:
            break
        if iter_ != 1 and corpus_train.current == -1:
            epoch += 1
        if no_update_counter > patience:
            print("Early stop training!")
            break

        if iter_ != 1 and corpus_train.current == -1:
            if epoch % check_epoch == 0:
                start_eval_time = timer()
                print ('*'*28, ' Epoch : ', str(epoch) , "Pass: ", str(corpus_train.epoch_counter) ,'*'*31)
                print("Bath_size: {}".format(batch_size))
                print("Started since {}".format(start_time))
                print("Time:                       {:.6f}".format(time))
                if print_every == -1:
                    print('%d  %.8f %.8f' % (iter_, loss_record_forward/iter_tmp, loss_record_backward/iter_tmp))
                    loss_record_forward = 0.0
                    loss_record_backward = 0.0
                    print_every_counter = 0
                time = 0
                iter_tmp = 0
                print("Validation set:")

                if config["loss"]["name"] == "vmf" or config["loss"]["name"] == "modified_cosine_l2":
                    loss_forward_eval, loss_backward_eval = eval_model(corpus_val, additional_elmo_model, test_size, 1.0) ##!!
                else:
                    loss_forward_eval, loss_backward_eval = eval_model(corpus_val, elmo, test_size, 1.0)
                    
                if config["dataset"]["emb_on_the_fly"]: 
                    one_batch = corpus_train.get_batch(batch_size) # This is actually because we got one batch from corpus_train, and during evaluation, the embedding was swapped...

                end_eval_time = timer()
                print("Model Name: {}".format(config["other_stuff"]["model_name"]))
                print("Spent {:.6f} on evaluation.".format(float(end_eval_time - start_eval_time)))
                print ('*'*28, ' Check Point Done', '*'*25)

            if (save_epoch != -1 and epoch % save_epoch == 0) or epoch in save_epochs:
                save_elmo.save_full_training_state(epoch = epoch,
                    train_iter = iter_,
                    optimizer_state = all_optimizer.state_dict(), 
                    optimizer_step = all_optimizer._step,
                    save_path = models_path + model_save_name + "_" + str(epoch) + "Files_full_state")

        if previous_epoch_counter < corpus_train.epoch_counter: # This is where we actually passed an Epoch!
            save_elmo.save_full_training_state(epoch = epoch,
                    train_iter = iter_,
                    optimizer_state = all_optimizer.state_dict(), 
                    optimizer_step = all_optimizer._step,
                    save_path = models_path + model_save_name + "_" + str(previous_epoch_counter) + "Epoch_full_state")

            previous_epoch_counter = corpus_train.epoch_counter

        if one_batch is not None:
            iter_ += 1
            iter_tmp += 1
            print_every_counter += 1
            (loss_forward, loss_backward) = trainer.train(one_batch, elmo, all_optimizer)

            s_read_data = timer()
            #######!!!####### get batch here to hide the get batch time in GPU computation time
            one_batch = corpus_train.get_batch(batch_size)
            if iter_ != 2:
                data_reading_time += timer() - s_read_data

            loss_record_forward += loss_forward.sum().item()
            if bidirectional:
                loss_record_backward += loss_backward.sum().item()
            else:
                loss_record_backward = 0.0

            if print_every != - 1 and  iter_ % print_every == 0:
                print('%d  %.8f %.8f' % (iter_, loss_record_forward/print_every_counter, loss_record_backward/print_every_counter))
                loss_record_forward = 0.0
                loss_record_backward = 0.0
                print_every_counter = 0
        else:
            fail_time += 1
            s_read_data = timer()
            one_batch = corpus_train.get_batch(batch_size)
            if iter_ != 2:
                data_reading_time += timer() - s_read_data
        e = timer()
        if iter_ != 2:
            time += float(e - s)