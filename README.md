# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
python train_intent.py
```

# 實驗紀錄
| name | val_acc | lr | hidden_size | num_layers | max_len | dropout | batch_size | num_epoch | dropout_layer | shuffle | others |
| ---- | ------- | -- | ----------- | ---------- | ------- | ------- | ---------- | --------- | ------------ | -------- | ------ |
| bset_0.ckpt | 0.83 | 1e-3 | 1024    |   2      |   32    |  0.2   |   256    |   200  | false | false |
| best_1.ckpt | 0.8737 | 1e-3 | 1024 | 2 | 32 | 0.2 | 256 | 50 | false | true |
| best_2.ckpt | 0.8957 | 1e-3 | 1024 | 2 | 32 | 0.2 | 256 | 50 | after fc | true |
| best_3.ckpt | 0.88 | 1e-3 | 1024 | 2 | 32 | 0.1 | 256 | 50 | after fc | true |
| best_3.ckpt | 0.88 | 1e-3 | 512 | 2 | 32 | 0.1 | 128 | 50 | after fc | true |
| best_3.ckpt | 0.8923 | 1e-3 | 1024 | 2 | 32 | 0.2 | 64 | 50 | after fc | true |
| best_4.ckpt | 0.92 | 1e-3 | 64 | 2 | 32 | 0.2 | 512 | 100 | False | True | two layer fc with BN ReLU |
| best_4.ckpt | 0.929 | 1e-3 | 64 | 2 | 32 | 0.2 | 512 | 100 | False | true | dropout BN LR, wd=0.000001 |
| best_4.ckpt | 0.938 | 1e-3 | 128 | 2 | 32 | 0.2 | 512 | 100 | False | True | tow layer fc with BN and LeakyReLU |
| best_5.ckpt | 0.937 | 1e-3 | 128 | 2 | 32 | 0.3 | 512 | 100 | False | true | dropout BN LR(0.4),（會 fit 爛）scheduler(step10, 0.5) |
| best_6.ckpt | 0.943 | 1e-3 | 128 | 2 | 32 | 0.3 | 512 | 100 | False | true | dropout BN LR(0.1), scheduler(step10, 0.5) |
| best_7.ckpt | 0.939 | 1e-3 | 128 | 2 | 32 | 0.2 | 512 | 100 | False | true | dropout BN LR(0.15), scheduler(step10, 0.5) |



# Report
## Q1
Describe how do you use the data for intent_cls.sh, slot_tag.sh: 
How do you tokenize the data.
The pre-trained embedding you used.
If you use the sample code, you will need to explain what it does in your own ways to answer Q1.

## Q2
- Describe
    - your model 
    - performance of your model.
    - (public score on kaggle)
    - the loss function you used.
    - The optimization algorithm (e.g. Adam), learning rate and batch size.

## Q3
- Describe 
    - your model 
    - performance of your model.
    - (public score on kaggle)
    - the loss function you used.
    - The optimization algorithm (e.g. Adam), learning rate and batch size.

## Q4
Please use seqeval to evaluate your model in Q3 on validation set and report classification_report(scheme=IOB2, mode=’strict’).
Explain the differences between the evaluation method in seqeval, token accuracy, and joint accuracy.

## Q5
Please try to improve your baseline method (in Q2 or Q3) with different configuration (includes but not limited to different number of layers, hidden dimension, GRU/LSTM/RNN) and EXPLAIN how does this affects your performance / speed of convergence / ...
Some possible BONUS tricks that you can try: multi-tasking, few-shot learning, zero-shot learning, CRF, CNN-BiLSTM
This question will be grade by the completeness of your experiments and your findings.
