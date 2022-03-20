import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from  tqdm import tqdm

from dataset import SlotTagDataset
from model import SlottTagger
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SlotTagDataset(data, vocab, tag2idx, args.max_len)
    # create DataLoader for test dataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SlottTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)

    # predict dataset
    rst_dict = {}
    model.eval()
    for test_data in tqdm(dataloader, desc="Test"):
        with torch.no_grad():
            features, length, ids = test_data['tokens'], test_data['length'], test_data['ids']
            features = torch.tensor(features).to(args.device)
            
            out = model(features).to(args.device)
            
            out = torch.permute(out, (0,2,1))
            _, preds = torch.max(out, 2)
            
            preds = preds.cpu().numpy()

            for i, pred in enumerate(preds):
                pred = pred[:length[i]]
                rst_dict[ids[i]] = [dataset.idx2label(idx=pred_id) for pred_id in pred]
    
    # write prediction to file (args.pred_file)
    with open(args.pred_file, "w") as f:
        f.write('id,tags\n')
        for id, tag_list in rst_dict.items():
            tag = ''.join(tag + " " for tag in tag_list).strip()
            f.write('{},{}\n'.format(id, tag))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=27)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
