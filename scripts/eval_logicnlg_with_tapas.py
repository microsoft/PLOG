import argparse
from transformers import TapasForSequenceClassification, TapasTokenizer
import torch, json, tqdm, os
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class MyData(Dataset):
    '''
    Dataset for loading table-text data
    '''
    def __init__(self, data_dir, data, tokenizer):
        self.data_dir = data_dir
        self.Data = self.load_data(data)
        self.len = len(self.Data)
        self.tokenizer = tokenizer

    def load_data(self, file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
        new_data = []
        for table_id, entry in data.items():
            for e in entry:
                new_data.append({"table_id": table_id, "sent": e})
        return new_data

    def read_data(self, data : dict):
        '''
        the input is a sample stored as dict
        return: a pandas table and the statement
        '''
        sent = data['sent']
        # table = pd.DataFrame(content, columns=header)
        table =  pd.read_csv(os.path.join(self.data_dir, 'all_csv/' + data['table_id']), '#')
        table = table.astype(str)
        return table, sent

    def encode(self, table, sent):
        return self.tokenizer(table=table, queries=sent,
                  truncation=True,
                  padding='max_length')

    def __getitem__(self, index):
        table, sent = self.read_data(self.Data[index])
        d = self.encode(table, sent)
        for key, value in d.items():
            d[key] = torch.LongTensor(value)
        return d

    def __len__(self):
        return self.len



class TapasTest:

    def __init__(self, model_name):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = TapasForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model.to(self.device)

    def test(self, test_dataloader):
        num_correct = 0
        num_all = 0
        result = {}
        for batch in tqdm.tqdm(test_dataloader):
            # get the inputs
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)

            # forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            model_predictions = outputs.logits.argmax(-1)
            # print(torch.nn.functional.softmax(outputs.logits, dim=1))
            num_correct += model_predictions.sum()
            num_all += model_predictions.size(0)
            result['num_correct'] = int(num_correct)
            result['num_all'] = int(num_all)
            result['acc'] = float(num_correct / num_all)

        return result


def unit_test(args):
    tapas = TapasTest("google/tapas-large-finetuned-tabfact")
    data = MyData(args.data_dir, args.test_file, tapas.tokenizer)
    test_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=1)
    results = tapas.test(test_dataloader)
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', default="", type=str, required=True)
    parser.add_argument('--data_dir', default="data/logicnlg", type=str)
    parser.add_argument('--batch_size', type=int, default=4)
    opt = parser.parse_args()
    unit_test(opt)










