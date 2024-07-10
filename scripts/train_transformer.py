import argparse
import os
import sys

from time import time

import torch

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.data import WikipediaTokenizedDataset
from modules.transformer import Transformer, TransformerConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_base_dir", type=str, required=True)
    parser.add_argument("--ckpts_dir", type=str, required=True)
    parser.add_argument("--logs_dir", type=str, required=True)

    # model config
    parser.add_argument("--vocab_size", type=int, default=15256)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--p_dropout", type=float, default=0.1)

    # training config
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save_every", type=int, default=100)
    return parser.parse_args()


def save_ckpt(steps, model, optimizer, scheduler, path_to_save):
    torch.save(
        {
            "steps": steps,
            "config": model.config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        path_to_save,
    )

def main(args):
    os.makedirs(args.ckpts_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    device = torch.device("mps")

    model = Transformer(
        TransformerConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            context_length=args.context_length,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            p_dropout=args.p_dropout,
        )
    ).to(device)

    train_dataset = WikipediaTokenizedDataset(os.path.join(args.dataset_base_dir, "train"))
    test_dataset = WikipediaTokenizedDataset(os.path.join(args.dataset_base_dir, "test"))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)  # type: ignore
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)  # type: ignore

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.steps)

    sw = SummaryWriter(args.logs_dir)

    for step, batch in enumerate(train_dataloader, start=1):
        t0 = time()

        # prepare
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        attn_mask = batch["pad_mask"].to(device)

        # train
        optimizer.zero_grad()
        loss = model.compute_loss(x, y, attn_mask)
        loss.backward()
        optimizer.step()
        scheduler.step()

        t1 = time()
        tokens_per_second = (args.batch_size * args.context_length) / (t1 - t0)

        # log
        print(f"{step}| {round(loss.item(), 2)}| tps: {tokens_per_second}")

        sw.add_scalar("train/loss", loss.item(), step)
        sw.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

        # save
        if step % args.save_every == 0:
            save_ckpt(step, model, optimizer, scheduler, os.path.join(args.ckpts_dir, f"ckpt_{step}.pt"))

        if step == args.steps:
            break

    model.save(os.path.join(args.ckpts_dir, f"model_{step}.pt"))


if __name__ == "__main__":
    main(parse_args())
