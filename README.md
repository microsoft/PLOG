# PLOG: Table-to-Logic Pretraining for Logical Table-to-Text Generation

Code and data for EMNLP 2022 paper [PLOG: Table-to-Logic Pretraining for Logical Table-to-Text Generation](https://arxiv.org/abs/2205.12697).


### Dependencies
```angular2html
python >= 3.8
transformers >= 4.5
torch >= 1.10.1
```

## Datasets

### [LOGICNLG](https://github.com/wenhuchen/LogicNLG)

Original LOGICNLG dataset consists of three `.json` files for train/dev/test samples and a directory `all_csv/` of `.csv` files for tables.
We provide additional table-to-logic pretraining data for LOGICNLG in `data/logicnlg`.

### CONTLOG
CONTLOG is collected based on [Logic2text](https://github.com/czyssrs/Logic2Text) dataset. We provide the table-to-text data and table-to-logic pretraining data in `data/contlog`.
The pretraining data will be provided later.



## Finetuning on Downstream Tasks

### Training

#### LOGICNLG
```
CUDA_VISIBLE_DEVICES=0 python train_logicnlg.py --do_train 
                          --model [t5-base|t5-large|facebook/bart-large] 
                          --task text 
                          --data_path data/logicnlg 
                          --use_cache 
                          --affix [experiment id] 
                          --interval_type epoch 
                          --pre_com
                          --load_from [pretrained model checkpoint path]
```
#### CONTLOG
```
CUDA_VISIBLE_DEVICES=0 python train_contlog.py --do_train
                          --model [t5-base|t5-large|facebook/bart-large]
                          --task text 
                          --data_path data/contlog 
                          --affix [experiment id] 
                          --interval_type epoch 
                          --pre_com
                          --load_from [pretrained model checkpoint path] 
```

### Evaluation
#### LOGICNLG
```
CUDA_VISIBLE_DEVICES=0 python train_logicnlg.py --do_test 
                          --model [t5-base|t5-large|facebook/bart-large]
                          --task text
                          --data_path data/logicnlg
                          --use_cache
                          --affix [experiment id]
                          --pre_com 
                          --load_from [checkpoint path]

```

#### CONTLOG
```
CUDA_VISIBLE_DEVICES=0 python train_logicnlg.py --do_test 
                          --model [t5-base|t5-large|facebook/bart-large] 
                          --task text 
                          --data_path data/contlog 
                          --affix [experiment id] 
                          --pre_com 
                          --load_from [checkpoint path]
```
## Pretraining with Table-to-Logic Data

#### LOGICNLG
```
CUDA_VISIBLE_DEVICES=0 python train_logicnlg.py --do_train 
                          --model [t5-base|t5-large|facebook/bart-large] 
                          --task logic 
                          --data_path data/logicnlg 
                          --use_cache 
                          --affix [experiment id] 
                          --interval_type step 
                          --pre_com
```
#### CONTLOG
```
CUDA_VISIBLE_DEVICES=0 python train_contlog.py --do_train
                          --model [t5-base|t5-large|facebook/bart-large]
                          --task logic
                          --data_path data/contlog 
                          --affix [experiment id] 
                          --interval_type step
                          --pre_com
```


## Reference

If you find this project useful in your work, please consider citing the paper:

```
@article{liu2022plog,
  title={PLOG: Table-to-Logic Pretraining for Logical Table-to-Text Generation},
  author={Liu, Ao and Dong, Haoyu and Okazaki, Naoaki and Han, Shi and Zhang, Dongmei},
  journal={arXiv preprint arXiv:2205.12697},
  year={2022}
}
```



## License

The CONTLOG dataset follows the MIT License.



