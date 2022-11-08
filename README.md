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




## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.




