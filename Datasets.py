import json
import os
from collections import defaultdict
import torch
import pandas as pd
from tqdm import tqdm
from execute.gen_APIs import process_num_table
from execute.APIs import add_agg_cell



def precompute_cell_information(pd_table):
    cell_ranks = {}
    columns = pd_table.columns
    for i in range(pd_table.shape[1]):
        try:
            sub_table = process_num_table(pd_table, columns[i])
            ranks = sub_table.rank(method='min')
        except:
            for j in range(pd_table.shape[0]):
                cell_ranks[(j, i)] = None, None
            continue

        rank_flag = not sub_table.isnull().values.any()
        for j in range(pd_table.shape[0]):
            if rank_flag:
                max_rank = len(ranks) - int(ranks.iloc[j]) + 1
                min_rank = -int(ranks.iloc[j])
            else:
                max_rank, min_rank = None, None
            cell_ranks[(j, i)] = max_rank, min_rank
    agg_cells = {}
    for i in range(pd_table.shape[1]):
        try:
            sum_val, avg_val = add_agg_cell(pd_table, columns[i])
        except:
            continue
        agg_cells[i] = (sum_val, avg_val)

    return cell_ranks, agg_cells

def process_logicnlg(data_file, ids, serialize_order='column', pre_com=True):
    '''
    Args:
        data_file: path to data file
        ids: table ids
        serialize_order: row-wise or column-wise serialization
        pre_com: whether to do numerical pre-computation

    Returns:
       new data with serialized input
    '''
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        new_data = []
        for table_id in tqdm(ids):
            entry = data[table_id]
            table = pd.read_csv('data/logicnlg/all_csv/' + table_id, '#')
            if pre_com:
                cell_ranks, agg_cells = precompute_cell_information(table)
            columns = table.columns
            for e in entry:
                doc = {}
                # "sent" can be either logic_str or text, depending on the dataset
                doc['sent'] = e[0]
                doc['table_id'] = table_id
                src_text = "<table> " + "<caption> " + e[2] + " </caption> "
                tmp = ""
                if serialize_order == 'column':
                    # column-wise serialization
                    for col in e[1]:
                        for i in range(len(table)):
                            if isinstance(table.iloc[i][columns[col]], str):
                                entity = map(lambda x: x.capitalize(), table.iloc[i][columns[col]].split(' '))
                                entity = ' '.join(entity)
                            else:
                                entity = str(table.iloc[i][columns[col]])
                            if pre_com:
                                max_rank, min_rank = cell_ranks[(i, col)]
                                if max_rank is not None:
                                    tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> <max_rank> {max_rank} </max_rank> <min_rank> {min_rank} </min_rank> </cell> "
                                else:
                                    tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> </cell> "
                            else:
                                tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> </cell> "

                        if pre_com and agg_cells and agg_cells.get(col, None):
                            sum_val, avg_val = agg_cells[col]
                            src_text += f"<sum_cell> {sum_val} <col_header> {columns[col]} </col_header> </sum_cell> "
                            src_text += f"<avg_cell> {avg_val} <col_header> {columns[col]} </col_header> </avg_cell> "

                elif serialize_order == 'row':
                    # Row-wise seralization
                    for i in range(len(table)):
                        for col in e[1]:
                            if isinstance(table.iloc[i][columns[col]], str):
                                entity = map(lambda x: x.capitalize(), table.iloc[i][columns[col]].split(' '))
                                entity = ' '.join(entity)
                            else:
                                entity = str(table.iloc[i][columns[col]])
                            if pre_com:
                                max_rank, min_rank = cell_ranks[(i, col)]
                                if max_rank is not None:
                                    tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> <max_rank> {max_rank} </max_rank> <min_rank> {min_rank} </min_rank> </cell> "
                                else:
                                    tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> </cell> "
                            else:
                                tmp += f"<cell> {entity} <col_header> {columns[col]} </col_header> <row_idx> {i} </row_idx> </cell> "

                    for col in e[1]:
                        if pre_com and agg_cells and agg_cells.get(col, None):
                            sum_val, avg_val = agg_cells[col]
                            src_text += f"<sum_cell> {sum_val} <col_header> {columns[col]} </col_header> </sum_cell> "
                            src_text += f"<avg_cell> {avg_val} <col_header> {columns[col]} </col_header> </avg_cell> "
                src_text += tmp + "</table>"
                doc['src_text'] = src_text
                new_data.append(doc)
        return new_data




def preprocess_contlog(data_file: str, add_type: bool, pre_com: bool):
    '''
    Args:
        data_file: path to the data file
        add_type: whether to add logic type information
        pre_com: whether to add pre-computed information.
    '''
    with open(data_file, 'r') as f:
        data = json.load(f)
        new_data = []
        for d in tqdm(data):
            table_header = d["table_header"]
            table_cont = d["table_cont"]
            h_idx = d['highlight_cells']
            src_text = "<table> " + "<caption> " +  d['topic'] + " </caption> "
            if add_type:
                src_text = "<type> " + d['action'] + "</type> " + src_text
            # Construct pandas table
            pd_in = defaultdict(list)
            for ind, header in enumerate(table_header):
                for inr, row in enumerate(table_cont):
                    if inr == len(table_cont) - 1 \
                            and ("all" in row[0] or "total" in row[0] or "sum" in row[0] or
                                 "a l l" in row[0] or "t o t a l" in row[0] or "s u m" in row[0]):
                        continue
                    pd_in[header].append(row[ind])
            pd_table = pd.DataFrame(pd_in)
            # precomputed ranks
            for row, col, max_rank, min_rank in h_idx:
                val = pd_table[col].iloc[row]
                if pre_com and max_rank is not None:
                    cell_str = f"<cell> {val} <col_header> {col} </col_header> <row_idx> {row} </row_idx> <max_rank> {max_rank} </max_rank> <min_rank> {min_rank} </min_rank></cell> "
                else:
                    cell_str = f"<cell> {val} <col_header> {col} </col_header> <row_idx> {row} </row_idx> </cell> "
                src_text += cell_str
            # precomputed aggregation values
            if pre_com and d['agg_cells']:
                sum_cell, avg_cell = d['agg_cells']
                src_text += f"<sum_cell> {sum_cell[1]} <col_header> {sum_cell[0]} </col_header> </sum_cell> "
                src_text += f"<avg_cell> {avg_cell[1]} <col_header> {avg_cell[0]} </col_header> </avg_cell> "
            src_text += "</table>"
            d['src_text'] = src_text
            new_data.append(d)
        return new_data





class ContlogDataset(torch.utils.data.Dataset):

    def __init__(self, data_file, tokenizer, max_src_len, max_tgt_len, task, add_type=False, pre_com=True):
        self.data = preprocess_contlog(data_file, add_type, pre_com)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        self.task = task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        src_text = d['src_text'].strip()
        src_text = ' '.join(src_text.split())

        if self.task == 'text' and 'sent' in d:
            tgt_text = d['sent'].strip()
        elif self.task == 'logic' and 'logic_str' in d:
            tgt_text = d['logic_str'].strip()
        else:
            raise NotImplementedError

        tgt_text = ' '.join(tgt_text.split())

        source = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_src_len
        )
        target = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tgt_len
        )

        source_input_ids = torch.LongTensor(source.data["input_ids"])
        target_input_ids = torch.LongTensor(target.data["input_ids"])
        source_mask = torch.LongTensor(source.data["attention_mask"])
        return {
            'source_ids': source_input_ids,
            'source_mask': source_mask,
            'target_ids': target_input_ids
        }


class LogicNLGDataset(torch.utils.data.Dataset):

    def __init__(self, option, data_file, id_file, tokenizer, args):
        if args.pre_com:
            cache_path = os.path.join(args.cache_root, f'{option}_{args.task}_{args.order}_agg.cache')
        else:
            cache_path = os.path.join(args.cache_root, f'{option}_{args.task}_{args.order}.cache')
        if os.path.exists(cache_path) and args.use_cache:
            self.data = torch.load(cache_path)
        else:
            with open(id_file) as f1:
                self.ids = json.load(f1)
            self.data = process_logicnlg(data_file, self.ids, args.order, args.pre_com)
            if args.use_cache:
                torch.save(self.data, cache_path)
        self.max_src_len = args.max_src_len
        self.max_tgt_len = args.max_tgt_len
        self.tokenizer = tokenizer
        self.task = args.task

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        src_text = d['src_text'].strip()
        src_text = ' '.join(src_text.split())

        tgt_text = d['sent'].strip()

        tgt_text = ' '.join(tgt_text.split())

        source = self.tokenizer(
            src_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_src_len
        )
        target = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_tgt_len
        )

        source_input_ids = torch.LongTensor(source.data["input_ids"])
        target_input_ids = torch.LongTensor(target.data["input_ids"])
        return {
            'source_ids': source_input_ids,
            'source_mask': torch.LongTensor(source.data["attention_mask"]),
            'target_ids': target_input_ids
        }


