import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from model import SlottTagger
from dataset import SlotTagDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SlotTagDataset] = {
        split: SlotTagDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # TODO: create DataLoader for train / dev datasets
    dataloader: Dict[str, DataLoader] = {
        split: (DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True) 
        if split == TRAIN  else 
        DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False))
        for split, dataset in datasets.items()        
    }
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    num_classes = len(tag2idx)
    # TODO: init model and move model to target device(cpu / gpu)
    model = SlottTagger(embeddings=embeddings, hidden_size=args.hidden_size, 
                          num_layers=args.num_layers, dropout=args.dropout,
                          bidirectional=args.bidirectional, num_class=num_classes,
                          sequence_length=args.max_len).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    train_acc_curve = []
    train_loss_curve = []
    eval_acc_curve = []
    eval_loss_curve = []

    # epoch_pbar = trange(args.num_epoch, desc="Epoch")
    # with trange(args.num_epoch, desc="Epoch") as epoch_pbar:
    # for epoch in epoch_pbar:
    for epoch in range(args.num_epoch):
        # TODO: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_correct = 0.0
        train_loss = 0.0
        print(f"\nEpoch: {epoch+1} / {args.num_epoch}")
        for train_data in tqdm(dataloader[TRAIN], desc="Train"):

            features, labels, ids = train_data['tokens'], train_data['tags'], train_data['ids']
            features, labels = torch.tensor(features).to(args.device), torch.tensor(labels).to(args.device)
            
            out = model(features)
            
            optimizer.zero_grad()
            loss = criterion(out, labels)
            
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out,1)
            
            # train_correct += [1 if (pred.cpu() == labels.cpu()).all() else 0]
            train_correct += sum([1 if seq.all() else 0 for seq in (pred.cpu() == labels.cpu())])
            
            # train_correct += (out.argmax(dim=-1) == labels).float().mean()
            train_loss += loss.item()
        
        train_acc = train_correct / len(data[TRAIN])
        # train_acc = train_correct / len(dataloader[TRAIN])
        train_loss /= len(dataloader[TRAIN])
        
        scheduler.step()

        # TODO: Evaluation loop - calcu
        model.eval()
        eval_correct = .0
        eval_loss = .0
        for eval_data in tqdm(dataloader[DEV], desc="Val"):
            with torch.no_grad():
                features, labels, ids = eval_data['tokens'], eval_data['tags'], eval_data['ids']
                features, labels = torch.tensor(features).to(args.device), torch.tensor(labels).to(args.device)
                out = model(features)
                
                loss = criterion(out, labels)
                _, pred = torch.max(out, 1)
                
                # eval_correct += (pred.cpu() == labels.cpu()).sum().item()
                eval_correct += sum([1 if seq.all() else 0 for seq in (pred.cpu() == labels.cpu())])
                
                eval_loss += loss.item()

        # eval_acc = eval_correct / len(dataloader[DEV])
        eval_acc = eval_correct / len(data[DEV])
        eval_loss /= len(dataloader[DEV])

        eval_acc_curve.append(eval_acc)
        eval_loss_curve.append(eval_loss)
        train_loss_curve.append(train_loss)
        train_acc_curve.append(train_acc)

        # tqdm.write("\n")
        print("Train ACC: {:.4f} Loss: {:.4f}".format(train_acc, train_loss))
        print("Val ACC: {:.4f} Loss: {:.4f}".format(eval_acc , eval_loss))
        
        # late accuracy and save model weights
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), args.ckpt_dir / 'best_1.pt')
            print('saving model with acc {:.4f}'.format(best_acc))
            
        # # early stop
        # if len(eval_acc_curve) > 40 and (max(eval_acc_curve[-20:]) - best_acc) < 0.01:
        #     break
    
        plt.plot(eval_acc_curve)
        plt.plot(train_acc_curve)
        plt.legend(["eval", "train"])
        plt.title("acc")
        plt.xlabel("Epoch")
        plt.xticks([epoch for epoch in range(args.num_epoch) if epoch % 10 == 0])
        plt.savefig("curve_acc")
        plt.clf()

        plt.plot(eval_loss_curve)
        plt.plot(train_loss_curve)
        plt.legend(["eval", "train"])
        plt.title("loss")
        plt.xlabel("Epoch")
        plt.xticks([epoch for epoch in range(args.num_epoch) if epoch % 10 == 0])
        plt.savefig("curve_loss")
        plt.clf()
    print("best acc:", best_acc)
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=16)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
