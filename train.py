# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                          GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from conv_ai.utils import get_dataset, make_logdir

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids",
                "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)


def average_distributed_scalar(scalar, local_rank, device):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float,
                            device=device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -1]
                         * (max_l - len(x)) for x in dataset[name]]
    return dataset


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(
        ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(
            new_num_tokens=orig_num_tokens + num_added_tokens)


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + \
        history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) %
                                 2 else speaker1] + s for i, s in enumerate(sequence[1:])]

    # truncate long sequence because the pretrained model has a limit of 512
    sequence = [subseq if len(subseq) <= 512 else subseq[0:512]
                for subseq in sequence]

    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i %
                                  2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s)
                                            for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance


def get_data_loaders(
    dataset_path,
    dataset_cache,
    num_candidates,
    personality_permutations,
    max_history,
    train_batch_size,
    valid_batch_size,
    distributed,
    tokenizer
):
    """ Prepare the dataset for training and evaluation """
    personachat = get_dataset(tokenizer, dataset_path, dataset_cache)

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
    for dataset_name, dataset in personachat.items():
        _num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(num_candidates, _num_candidates)
        for dialog in dataset:
            persona = dialog["personality"].copy()
            for _ in range(personality_permutations):
                for utterance in dialog["utterances"]:
                    history = utterance["history"][-(2*max_history+1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        instance = build_input_from_segments(
                            persona, history, candidate, tokenizer, lm_labels
                        )
                        for input_name, input_array in instance.items():
                            datasets[dataset_name][input_name].append(
                                input_array)
                    datasets[dataset_name]["mc_labels"].append(
                        num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
                # permuted personalities
                persona = [persona[-1]] + persona[:-1]

    logger.info("Pad inputs and convert to Tensor")
    tensor_datasets = {"train": [], "valid": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(
            dataset, padding=tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1]))
        for input_name in MODEL_INPUTS:
            tensor = torch.tensor(dataset[input_name])
            if input_name != "mc_labels":
                tensor = tensor.view(
                    (-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
            tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = TensorDataset(
        *tensor_datasets["train"]), TensorDataset(*tensor_datasets["valid"])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset) if distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset) if distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler,
                              batch_size=train_batch_size, shuffle=(not distributed))
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler,
                              batch_size=valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(
        train_dataset.tensors[0].shape))
    logger.info("Valid dataset (Batch, Candidates, Seq length): {}".format(
        valid_dataset.tensors[0].shape))
    return train_loader, valid_loader, train_sampler, valid_sampler


def train(
    dataset_path,
    dataset_cache='./dataset_cache',
    model_checkpoint='gpt2',
    num_candidates=2,
    max_history=2,
    train_batch_size=4,
    valid_batch_size=4,
    gradient_accumulation_steps=8,
    lr=6.25e-5,
    lm_coef=1.0,
    mc_coef=1.0,
    max_norm=1.0,
    n_epochs=3,
    personality_permutations=1,
    eval_before_start=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    fp16='',
    path_prefix='',
    log_dir='',
    local_rank=-1
):
    args = {**locals()}

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(
        level=logging.INFO if local_rank in [-1, 0] else logging.WARN)
    # This is a logger.warning: it will be printed by all distributed processes
    logger.warning("Running process %d", local_rank)
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    distributed = (local_rank != -1)
    args['distributed'] = distributed

    if distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    # cant use Autotokenizer because checkpoint could be a Path
    tokenizer_class = GPT2Tokenizer if "gpt2" in model_checkpoint else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)

    model_class = GPT2DoubleHeadsModel if "gpt2" in model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(model_checkpoint)
    model.to(device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16)
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(
        dataset_path,
        dataset_cache,
        num_candidates,
        personality_permutations,
        max_history,
        train_batch_size,
        valid_batch_size,
        distributed,
        tokenizer
    )

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels
        )
        loss = (lm_loss * lm_coef + mc_loss * mc_coef) / \
            gradient_accumulation_steps
        if fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(optimizer), max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if engine.state.iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            lm_logits, mc_logits, *_ = model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )
            lm_logits_flat_shifted = lm_logits[..., :-1,
                                               :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              lambda _: evaluator.run(val_loader))
    if n_epochs < 1:
        trainer.add_event_handler(
            Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if eval_before_start:
        trainer.add_event_handler(
            Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if distributed:
        trainer.add_event_handler(
            Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(
            Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(
        optimizer, "lr", [(0, lr), (n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-1), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], local_rank, device),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], local_rank, device)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message(
            "Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = log_dir if log_dir else make_logdir(
            model_checkpoint, path=path_prefix)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(
            tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(
            optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(
            metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(
            log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(
            model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_bin')
        getattr(model, 'module', model).config.to_json_file(
            os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if local_rank in [-1, 0] and n_epochs > 0:
        # TODO: PR in ignite to have better access to saved file paths (cleaner)
        os.rename(checkpoint_handler._saved[-1][1]
                  [-1], os.path.join(log_dir, WEIGHTS_NAME))
        tb_logger.close()
