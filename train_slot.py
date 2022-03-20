import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from collections import Counter

import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from seqeval.metrics import f1_score, accuracy_score, classification_report
from seqeval.scheme import IOB2

from model import SlottTagger
from dataset import SlotTagDataset
from utils import Vocab
# from focal_loss import focal_loss

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    datasets: Dict[str, SlotTagDataset] = {
        split: SlotTagDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    # create DataLoader for train / dev datasets
    dataloader: Dict[str, DataLoader] = {
        split: (DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=True)
                if split == TRAIN else
                DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False))
        for split, dataset in datasets.items()
    }

    # imbalance class
    cnt = Counter(())
    pad_count = 0
    for train_data in data[TRAIN]:
        cnt.update(train_data["tags"])
        pad_count += max(args.max_len, args.max_len - len(train_data["tags"]))
    cnt["PAD"] = pad_count

    # class_weight = [0] * (datasets[TRAIN].num_classes + 1)
    class_weight = torch.zeros(datasets[TRAIN].num_classes + 1).to(args.device)
    class_weight += len(data[TRAIN]) + pad_count

    for tag, count in cnt.items():
        try:
            idx = datasets[TRAIN].label2idx(tag)
            class_weight[idx] /= (count * len(cnt))
        except:
            class_weight[-1] /= (cnt["PAD"] * len(cnt))

    print(cnt)
    print(class_weight)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    num_classes = len(tag2idx)
    # init model and move model to target device(cpu / gpu)
    model = SlottTagger(embeddings=embeddings, hidden_size=args.hidden_size,
                        num_layers=args.num_layers, dropout=args.dropout,
                        bidirectional=args.bidirectional, num_class=num_classes).to(args.device)

    # init optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-8)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=200, gamma=0.5)

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weight, label_smoothing=0.01)
    # criterion = focal_loss(alpha=class_weight, gamma=2, reduction='mean', device=args.device)

    best_acc = .0
    train_acc_curve = []
    train_loss_curve = []
    eval_acc_curve = []
    eval_loss_curve = []

    for epoch in range(args.num_epoch):
        # Training loop - iterate over train dataloader and update model weights
        model.train()
        train_correct = 0.0
        train_loss = 0.0
        print(f"\nEpoch: {epoch+1} / {args.num_epoch}")
        for train_data in tqdm(dataloader[TRAIN], desc="Train"):
            features, labels, ids = train_data['tokens'], train_data['tags'], train_data['ids']
            features, labels = torch.tensor(features).to(
                args.device), torch.tensor(labels).to(args.device)
            out = model(features)
            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out, 1)
            
            train_correct += sum([1 if seq.all()
                                 else 0 for seq in (pred.cpu() == labels.cpu())])
            
            train_loss += loss.item()

        train_acc = train_correct / len(data[TRAIN])
        train_loss /= len(dataloader[TRAIN])

        # Evaluation loop
        model.eval()
        eval_correct = .0
        eval_loss = .0
        token_acc = .0
        token_num = .0
        y_true = []
        y_pred = []
        for eval_data in tqdm(dataloader[DEV], desc="Val"):
            with torch.no_grad():
                features, labels, ids, length = eval_data['tokens'], eval_data['tags'], eval_data['ids'], eval_data['length']
                features, labels = torch.tensor(features).to(
                    args.device), torch.tensor(labels).to(args.device)
                out = model(features)

                loss = criterion(out, labels)
                out = out.permute(0,2,1)
                _, preds = torch.max(out, 2)
                
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
                # seqeval
                for b, pred, lbl in zip(length, preds, labels):
                    pred = pred[:b]
                    lbl = lbl[:b]
                    batch_pred = [datasets[DEV].idx2label(idx=pred_id) for pred_id in pred]
                    batch_true = [datasets[DEV].idx2label(idx=lbl_id) for lbl_id in lbl]
                    y_pred.append(batch_pred)
                    y_true.append(batch_true)

                    # token acc
                    token_num += len(batch_true)
                    token_acc += sum(np.array(batch_true) == np.array(batch_pred))

                # joint acc
                eval_correct += sum([1 if seq.all()
                                    else 0 for seq in (preds == labels)])
                eval_loss += loss.item()

        eval_acc = eval_correct / len(data[DEV])
        eval_loss /= len(dataloader[DEV])

        print("Train ACC: {:.4f} Loss: {:.4f}".format(train_acc, train_loss))
        print("Val Acc: {:.4f} Loss: {:.4f}".format(eval_acc, eval_loss))

        # late accuracy and save model weights
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(), args.ckpt_dir / '_best.pt')
            print('saving model with acc {:.4f}'.format(best_acc))
            
            # show seqeval
            print("-"*50)
            cr = classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
            print(cr)
            print("Joint ACC: {:.4f}".format(eval_acc))
            print("Token ACC: {:.4f}".format(token_acc / token_num))
        
        # plot curve
        eval_acc_curve.append(eval_acc)
        eval_loss_curve.append(eval_loss)
        train_loss_curve.append(train_loss)
        train_acc_curve.append(train_acc)
        
        plt.plot(eval_acc_curve)
        plt.plot(train_acc_curve)
        plt.legend(["eval", "train"])
        plt.title("acc")
        plt.xlabel("Epoch")
        plt.xticks([epoch for epoch in range(
            args.num_epoch) if epoch % 10 == 0])
        plt.savefig("curve_acc")
        plt.clf()

        plt.plot(eval_loss_curve)
        plt.plot(train_loss_curve)
        plt.legend(["eval", "train"])
        plt.title("loss")
        plt.xlabel("Epoch")
        plt.xticks([epoch for epoch in range(
            args.num_epoch) if epoch % 10 == 0])
        plt.savefig("curve_loss")
        plt.clf()

    print("best acc:", best_acc)


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
    parser.add_argument("--max_len", type=int, default=27)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)

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
