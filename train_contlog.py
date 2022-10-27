# coding=utf-8
import math
import random
import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import \
    (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Adafactor,
        AdamW
    )


from Datasets import ContlogDataset
from eval_utils import validation_contlog

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='facebook/bart-base', type=str)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to compute the BLEU scores on test split")
    parser.add_argument('--pretrain', default=False, action="store_true", help="whether to train or test the model")

    parser.add_argument('--optimizer', default='Adamw', choices=['Adamw', 'Adafactor'], type=str)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=5 , type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--every', default=50, type=int, help="interval for evaluation")
    parser.add_argument('--interval_type', default='step', type=str, choices=['step', 'epoch'])
    parser.add_argument('--interval_step', default=16000, type=int, help="interval for evaluation when interval_type = step.")
    parser.add_argument('--load_from', default=None, type=str, help="model checkpoint path")
    parser.add_argument('--max_src_len', default=500, type=int, help="whether to train or test the model")
    parser.add_argument('--max_tgt_len', default=200, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=5, help="accumulation steps for gradient")

    parser.add_argument('--data_path', type=str, default='data/contlog')
    parser.add_argument('--log_path',type=str, default='../logs/d2t/outputs')
    parser.add_argument('--ckpt_path', type=str, default='../models/d2t')
    parser.add_argument('--affix', type=str, default=None, required=True, help="The experiment name")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_gpu', type=str, default=0)
    parser.add_argument('--task', type=str, default='text', help='task: text (table2text) or logic (table2logic)')
    parser.add_argument('--add_type', default=False, action="store_true")
    parser.add_argument('--pre_com', default=False, action="store_true", help="whether to do numerical precomputation")
    parser.add_argument('--global_step', default=1, type=int)
    parser.add_argument('--use_cache', default=False, action="store_true")

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()



    set_seed(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    markers = ["{", "}", "<table>", "</table>", "<type>", "</type>", "<cell>", "</cell>", "<col_header>", "</col_header>", "<row_idx>", "</row_idx>"]
    if args.pre_com:
        markers += ["<max_rank>", "</max_rank>", "<min_rank>", "</min_rank>", "<sum_cell>", "</sum_cell>", "<avg_cell>",
                "</avg_cell>"]

    # with open('special_vocab.json') as f:
    #     special_vocabs = json.load(f)
    # vocab = tokenizer.get_vocab()
    # for token in special_vocabs:
    #     if token not in vocab.keys():
    #         markers.append(token)

    tokenizer.add_tokens(markers)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(args.device)
    model.resize_token_embeddings(len(tokenizer))
    if args.load_from is not None:
        model.load_state_dict(torch.load(args.load_from))


    def freeze_params(model: nn.Module):
        """Set requires_grad=False for each of model.parameters()"""
        for par in model.parameters():
            par.requires_grad = False



    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    if not os.path.exists(os.path.join(args.log_path, args.affix)):
        os.makedirs(os.path.join(args.log_path, args.affix))
    if not os.path.exists(os.path.join(args.ckpt_path, args.affix)):
        os.makedirs(os.path.join(args.ckpt_path, args.affix))

    if args.do_train:
        # freeze embedding layers
        if args.model.startswith('t5'):
            for d in [model.encoder]:
                freeze_params(d.embed_tokens)
        else:
            for d in [model.model.encoder, model.model.decoder]:
                # freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)

        # Determin pretraining data.
        if args.task == 'logic':
            train_file = os.path.join(args.data_path, 'all_pretrain_800k.json')
            val_file = os.path.join(args.data_path, 'all_pretrain_valid.json')
            test_file = os.path.join(args.data_path, 'all_pretrain_test.json')
        elif args.task == 'text':
            train_file = os.path.join(args.data_path, 'train.json')
            val_file = os.path.join(args.data_path, 'val.json')
            test_file = os.path.join(args.data_path, 'test.json')
        else:
            raise NotImplementedError

        train_dataset = ContlogDataset(train_file, tokenizer, args.max_src_len, args.max_tgt_len, args.task, args.add_type, args.pre_com)
        train_loader = DataLoader(train_dataset,
                              num_workers=5,
                              batch_size=args.batch_size,
                              shuffle=True)
        model.train()

        if args.optimizer == 'Adamw':
            optimizer = AdamW(model.parameters(), args.learning_rate)
        elif args.optimizer == 'Adafactor':
            optimizer = Adafactor(model.parameters(), args.learning_rate, relative_step=False)
        else:
            raise NotImplementedError

        global_step = 0
        total_loss = []
        # best validation score
        best_val = 0
        best_metric = "bleu4" if args.task == 'text' else "exec_acc"

        for epoch_idx in range(1, args.epoch+1):

            print("[INFO] start training {}th epoch".format(epoch_idx))
            for idx, batch in enumerate(tqdm(train_loader)):
                lm_labels = batch['target_ids'].to(args.device, dtype=torch.long)
                lm_labels[lm_labels == tokenizer.pad_token_id] = -100
                ids = batch['source_ids'].to(args.device, dtype=torch.long)
                mask = batch['source_mask'].to(args.device, dtype=torch.long)

                outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

                loss = loss / args.gradient_accumulation_steps

                total_loss.append(loss.item())
                loss.backward()

                if (idx + 1) % args.gradient_accumulation_steps == 0 or (idx + 1) == len(train_loader):
                    optimizer.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                if idx % args.every == 0 and idx > 0:
                    perplexity = math.exp(np.mean(total_loss))
                    total_loss = []



                if (args.interval_type == 'step' and global_step % args.interval_step == 0 and global_step > 0)\
                        or (args.interval_type == 'epoch' and (idx + 1) == len(train_loader)):
                    if args.interval_type == 'step':
                        torch.save(model.state_dict(), '{}/{}/{}_step{}.pt'.format(args.ckpt_path, args.affix, args.model.split('/')[-1], global_step))
                    else:
                        torch.save(model.state_dict(), '{}/{}/{}_ep{}.pt'.format(args.ckpt_path, args.affix, args.model.split('/')[-1], epoch_idx))

                    val_scores = validation_contlog(val_file, model, tokenizer, 'valid', args)
                    if val_scores[best_metric] > best_val:
                        best_val = val_scores[best_metric]
                        test_scores = validation_contlog(test_file, model, tokenizer, 'test', args)
                global_step += 1

    if args.do_test:
        test_scores = validation_contlog(test_file, model, tokenizer, 'test', args)
        print(test_scores)