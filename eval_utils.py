# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, io, re, subprocess
import json
import logging
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from pyrouge import Rouge155
import nltk
import pandas as pd
from execute.str2form import parse_str
from execute.execute import func_map, execute, execute_logicnlg
from execute.APIs import APIs
from Datasets import ContlogDataset, LogicNLGDataset
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def bleu_score(labels_file, predictions_path):
    bleu_script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'multi-bleu.perl')
    try:
        with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
            bleu_out = subprocess.check_output(
                [bleu_script, labels_file],
                stdin=predictions_file,
                stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            return float(bleu_score)

    except subprocess.CalledProcessError as error:
        return None

def pattern_match(logic, truth):

    def extract_pattern(logic):
        logic = logic.strip()[:-7]
        API_words = [_ for _ in APIs.keys()]
        API_words += ['hop']
        key_words = ['{', '}', 'all_rows', ';']
        temp = []
        for i, x in enumerate(logic.split()):
            if x in key_words or (x in API_words and i < len(logic.split()) - 1 and logic.split()[i + 1] == "{"):
                temp.append(x)
        return temp

    logic = extract_pattern(logic)
    truth = extract_pattern(truth)
    if len(logic) != len(truth):
        return False
    for a, b in zip(logic, truth):
        if a != b:
            return False
    return True

def validation_logicnlg(val_file, val_ids_file, model, tokenizer, split, args, step):
    val_dataset = LogicNLGDataset(split, val_file, val_ids_file,
                                      tokenizer, args)
    val_loader = DataLoader(val_dataset,
                            num_workers=5,
                            batch_size=args.batch_size,
                            shuffle=False)
    model.eval()
    with open(val_file, 'r') as dref:
        reference_data = json.load(dref)

    def get_references(reference, table_id):
        entry = reference[table_id]
        return [_[0].lower().split(' ') for _ in entry]

    k = 0
    pred_by_ids = defaultdict(list)
    ref_by_ids = defaultdict(list)
    
    if args.task == 'text':
        sent_bleus_1, sent_bleus_2, sent_bleus_3 = [], [], []

        with torch.no_grad():
            # original data for tracking table_ids
            val_data = val_dataset.data
            for idx, batch in enumerate(tqdm(val_loader)):
                ids = batch['source_ids'].to(args.device, dtype=torch.long)
                mask = batch['source_mask'].to(args.device, dtype=torch.long)

                samples = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=args.max_tgt_len,
                    num_beams=4,
                    early_stopping=False
                )

                for s in samples:
                    table_id = val_data[k]['table_id']
                    if not ref_by_ids[table_id]:
                        ref_by_ids[table_id] = get_references(reference_data, table_id)
                    k += 1
                    text = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pred_by_ids[table_id].append(text)

                    references = ref_by_ids[table_id]
                    hypothesis = text.lower().split(' ')
                    sent_bleus_1.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(1, 0, 0)))
                    sent_bleus_2.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(0.5, 0.5, 0)))
                    sent_bleus_3.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(0.33, 0.33, 0.33)))

            bleu_1 = format((sum(sent_bleus_1) / len(sent_bleus_1) * 100), '.4f')
            bleu_2 = format((sum(sent_bleus_2) / len(sent_bleus_2) * 100), '.4f')
            bleu_3 = format((sum(sent_bleus_3) / len(sent_bleus_3) * 100), '.4f')
            val_metric_dict = {'bleu_1': float(bleu_1), 'bleu_2': float(bleu_2), 'bleu_3': float(bleu_3)}
    
    elif args.task == 'logic':
        with torch.no_grad():
            # original data for tracking table_ids
            val_data = val_dataset.data
            for idx, batch in enumerate(tqdm(val_loader)):
                ids = batch['source_ids'].to(args.device, dtype=torch.long)
                mask = batch['source_mask'].to(args.device, dtype=torch.long)

                samples = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_length=args.max_tgt_len,
                    num_beams=4,
                    early_stopping=False
                )

                for s in samples:
                    table_id = val_data[k]['table_id']
                    if not ref_by_ids[table_id]:
                        ref_by_ids[table_id] = get_references(reference_data, table_id)
                    k += 1
                    text = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pred_by_ids[table_id].append(text.lower())

            with open(os.path.join(args.log_path, args.affix, f'predictions_{split}_{step}.json'), 'w') as pred:
                json.dump(pred_by_ids, pred, indent=2)

            # Evaluate the correctness of logical forms

            num_samples = 0
            num_p_correct = 0
            num_exact = 0
            num_e_correct = 0
            for table_id in tqdm(pred_by_ids.keys()):
                table = pd.read_csv(os.path.join(args.data_path, 'all_csv', table_id), '#')
                pred_list = pred_by_ids[table_id]
                ref_list = ref_by_ids[table_id]
                for pred in pred_list:
                    num_samples += 1
                    if any([pred in ref and ref in pred for ref in ref_list]):
                        num_exact += 1
                        num_p_correct += 1
                        num_e_correct += 1
                    else:
                        label = any([pattern_match(pred, ref) for ref in ref_list])
                        if label:
                            num_p_correct += 1
                        try:
                            cur_logic = parse_str(pred[:-7], func_map)
                            res = execute_logicnlg(table, cur_logic)
                            if res == True:
                                num_e_correct += 1
                        except:
                            continue

            acc_exact = format(1. * num_exact / num_samples * 100, '.2f')
            acc_p = format(1. * num_p_correct / num_samples * 100, '.2f')
            acc_e = format(1. * num_e_correct / num_samples * 100, '.2f')
            val_metric_dict = {"exec_acc": float(acc_exact),
                               "pat_acc": float(acc_p),
                               "LF_acc": float(acc_e)
                               }
    else:
        raise NotImplementedError

    with open(os.path.join(args.log_path, args.affix, f'predictions_{split}_{step}.json'), 'w') as pred:
            json.dump(pred_by_ids, pred, indent=2)

    model.train()
    return val_metric_dict



def validation_contlog(val_file, model, tokenizer, split, args):
    val_dataset = ContlogDataset(val_file, tokenizer, args.max_src_len, args.max_tgt_len,
                                     args.task, args.add_type, args.pre_com)
    val_loader = DataLoader(val_dataset,
                            num_workers=5,
                            batch_size=args.batch_size,
                            shuffle=False)
    model.eval()
    pred_list = []
    ref_list = []

    # create files for scripts
    gt = open(os.path.join(args.log_path, args.affix, f'references_{split}.txt'), 'w')
    pred = open(os.path.join(args.log_path, args.affix, f'predictions_{split}.txt'), 'w')
    pred_split = os.path.join(args.log_path, args.affix, f'predictions_{split}/')
    ref_split = os.path.join(args.log_path, args.affix, f'references_{split}/')
    if not os.path.exists(pred_split):
        os.makedirs(pred_split)
    if not os.path.exists(ref_split):
        os.makedirs(ref_split)
    k = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            y = batch['target_ids'].to(args.device, dtype=torch.long)
            ids = batch['source_ids'].to(args.device, dtype=torch.long)
            mask = batch['source_mask'].to(args.device, dtype=torch.long)
            samples = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=args.max_tgt_len,
                num_beams=4,
                early_stopping=False
            )


            for reference, s in zip(y, samples):
                with open(ref_split + str(k) + '_reference.txt', 'w') as sr, \
                        open(pred_split + str(k) + '_prediction.txt', 'w') as sw:
                    reference = tokenizer.decode(reference, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
                    gt.write(reference.lower() + '\n')
                    sr.write(reference.lower() + '\n')
                    ref_list.append(reference.lower())
                    text = tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    pred_list.append(text.lower())
                    pred.write(text.lower() + '\n')
                    sw.write(text.lower() + '\n')
                    k += 1
        gt.close()
        pred.close()

        if args.task == 'text':
            gt = os.path.join(args.log_path, args.affix, f'references_{split}.txt')
            pred = os.path.join(args.log_path, args.affix, f'predictions_{split}.txt')
            bleu4 = bleu_score(gt, pred)
            print("[INFO] {} BLEU score = {}".format(split, bleu4))
            # log_file.write("[INFO] {} BLEU score = {}\n".format(split, bleu4))

            # ROUGE scripts
            r = Rouge155()
            r.system_dir = os.path.join(args.log_path, args.affix, f'predictions_{split}/')
            r.model_dir = os.path.join(args.log_path, args.affix, f'references_{split}/')
            # define the patterns
            r.system_filename_pattern = '(\d+)_prediction.txt'
            r.model_filename_pattern = '#ID#_reference.txt'
            logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
            results_dict = r.convert_and_evaluate()
            rouge_result = "\n".join(
                [results_dict.split("\n")[3], results_dict.split("\n")[7], results_dict.split("\n")[15],
                 results_dict.split("\n")[19]])
            print("[INFO] Rouge scores: \n", rouge_result)
            # log_file.write(rouge_result + '\n')
            results_dict = results_dict.split("\n")
            rouge_score_list = []

            for i in [3, 7, 15, 19]:
                results = results_dict[i]
                rouge_score = float(results.split()[3])
                rouge_score_list.append(rouge_score * 100)

        # If the task is table-to-logic generation
        elif args.task == 'logic':
            num_samples = 0  # all samples
            num_p_correct = 0  # pattern-match examples
            num_exact = 0  # exact-match examples
            num_e_correct = 0  # execution-correct examples
            with open(val_file, 'r') as fp:
                data = json.load(fp)
                for d, logic_str in zip(data, pred_list):
                    num_samples += 1
                    # exact match
                    if logic_str == d['logic_str']:
                        num_exact += 1
                        num_p_correct += 1
                        num_e_correct += 1
                    else:
                        # pattern match
                        label = pattern_match(logic_str, d['logic_str'])
                        if label:
                            num_p_correct += 1
                        try:
                            # execution accuracy evaluate
                            cur_logic = parse_str(logic_str[:-7], func_map)
                            cur_execute_batch = {"table_header": d['table_header'],
                                                 "table_cont": d['table_cont'],
                                                 "logic": cur_logic}
                            res = execute(cur_execute_batch)
                            if res == True:
                                num_e_correct += 1
                        except:
                            continue
                acc_exact = 1. * num_exact / num_samples * 100
                acc_p = 1. * num_p_correct / num_samples * 100
                acc_e = 1. * num_e_correct / num_samples * 100
            print("[INFO] Execution accuracy:  ", acc_e)
            print("[INFO] Pattern accuracy: ", acc_p)
            print("[INFO] Exact-match accuracy:  ", acc_exact)


        val_metric_dict = {}
        if args.task == 'text':
            for type, score in zip(['1', '2', '4', 'L'], rouge_score_list):
                val_metric_dict[f'rouge{type}'] = score
            val_metric_dict['bleu4'] = bleu4
        elif args.task == 'logic':
            val_metric_dict = {"exec_acc": acc_exact,
                               "pat_acc": acc_p,
                               "LF_acc": acc_e
                               }
        else:
            raise NotImplementedError
        model.train()
        return val_metric_dict
