import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import SeqClassifier
from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    
    # TODO: create DataLoader for train / dev datasets
    dataloader: Dict[str, DataLoader] = {
        split: DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)
        for split, dataset in datasets.items()
    }
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    num_classes = len(intent2idx)
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(embeddings=embeddings, hidden_size=args.hidden_size, 
                          num_layers=args.num_layers, dropout=args.dropout,
                          bidirectional=args.bidirectional, num_class=num_classes,
                          sequence_length=args.max_len).to(args.device)

    # TODO: init optimizer
    # optimizer = torch.optim.Adam(model.parameters(),lr=args.lr, weight_decay=0.01)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.8)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = .0
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
        train_correct = .0
        train_loss = .0
        print(f"\nEpoch: {epoch} / {args.num_epoch}")
        for train_data in tqdm(dataloader[TRAIN], desc="Train"):
            
            features, labels, ids = train_data['features'], train_data['labels'], train_data['ids']
            features, labels = torch.tensor(features).to(args.device), torch.tensor(labels).to(args.device)
            
            out = model(features).to(args.device)
            loss = criterion(out, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out,1)
            # print(pred)
            # print(labels)
            train_correct += (pred.cpu() == labels.cpu()).sum().item()
            train_loss += loss.item()
        
        train_acc = train_correct / len(data[TRAIN])
        train_loss /= len(dataloader[TRAIN])
        
        # scheduler.step()

        # TODO: Evaluation loop - calcu
        model.eval()
        eval_correct = .0
        eval_loss = .0
        val_all = 0
        for eval_data in tqdm(dataloader[DEV], desc="Val"):
            with torch.no_grad():
                features, labels, ids = eval_data['features'], eval_data['labels'], eval_data['ids']
                features, labels = torch.tensor(features).to(args.device), torch.tensor(labels).to(args.device)
                out = model(features).to(args.device)
                loss = criterion(out, labels)
                _, pred = torch.max(out,1)
                val_all += labels.shape[0]
                eval_correct += (pred.cpu() == labels.cpu()).sum().item()
                eval_loss += loss.item()

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
            torch.save(model.state_dict(), args.ckpt_dir / 'best_3.ckpt')
            print('saving model with acc {:.4f}'.format(best_acc))
    
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
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=64)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=200)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
