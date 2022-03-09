import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from tqdm import trange
from random import random

from model_bohan import LSTM
from dataset_bohan import SeqClsDataset
from utils import Vocab
from torch.utils.data import DataLoader

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    BATCH_SIZE = 256
    max_len = 32
    # TODO: crecate DataLoader for train / dev datasets
    # 分別讀出train與eval dataset
    for dataset in datasets:
        for idx, all_data in enumerate(datasets[dataset]):  # 存取裡面data
            result = torch.zeros(max_len, 300)  # 串起來的結果
            count = 0  # 判斷長度
            input_list = all_data['text'].split()
            if len(input_list) > 0:
                for word in input_list:
                    if word in vocab.tokens:
                        result[count] = embeddings[vocab.token2idx[word]]
                        count += 1
                        if count >= max_len:
                            break
            all_data['text'] = result
            # all_data['text'] = [embeddings[vocab.encode(word)]
            #                     for word in all_data['text']]
            all_data['intent'] = SeqClsDataset.label2idx(datasets[dataset],
                                                         label=all_data['intent'])  # 將str map成int
    train_loader = DataLoader(
        datasets['train'], batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(
        datasets['eval'], batch_size=BATCH_SIZE, shuffle=False)

    # TODO: init model and move model to target device(cpu / gpu)
    device = get_device()
    print(device)

    num_epoch = 50
    input_dim = 300  # glove的dim
    batch = 256
    class_num = len(intent2idx)
    hidden = 1024
    layer = 2
    model = LSTM(input_dim, hidden, layer, class_num, batch).to(device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_acc = 0.0  # 判斷好壞
    model_path = './ckpt/intent/LSTM_48.ckpt'
    for epoch in range(num_epoch):
        criterion = nn.CrossEntropyLoss()
        train_acc = 0.0
        train_all = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_all = 0.0
        val_loss = 0.0
        # if epoch == 35:
        #     optimizer = torch.optim.SGD(
        #         model.parameters(), lr=0.01, momentum=0.9)
        # training
        model.train()  # train mode
        for i, data in enumerate(train_loader):
            inputs, labels, index = data['text'], data['intent'], data['id']
            inputs, labels = inputs.to(
                device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1)  # 得到最高分類別
            batch_loss.backward()
            optimizer.step()
            train_all += labels.shape[0]
            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                inputs, labels, index = data['text'], data['intent'], data['id']
                inputs, labels = inputs.to(
                    device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)
                val_all += labels.shape[0]
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/train_all, train_loss/len(
                    train_loader), val_acc/val_all, val_loss/len(eval_loader)
            ))

            # 進步則存檔
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(
                    best_acc/len(eval_loader)))
            else:
                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                    epoch + 1, num_epoch, train_acc /
                    len(train_loader), train_loss/len(train_loader)
                ))
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        pass

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=2)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
