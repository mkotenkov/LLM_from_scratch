import argparse
import os
import shutil
import sys

from time import time

import torch

from torch.utils.tensorboard.writer import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.data import WikipediaTokenizedDataset
from modules.tokenizer import Tokenizer
from modules.transformer import SamplingStrategy, Transformer, TransformerConfig


def parse_args():
    parser = argparse.ArgumentParser()

    # directories
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
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)

    return parser.parse_args()


def manage_dirs(ckpts_dir, logs_dir):
    if os.path.exists(ckpts_dir):
        shutil.rmtree(ckpts_dir)
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)

    os.makedirs(ckpts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)


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


@torch.no_grad()
def validate(model, dataloader):
    total_loss = 0
    for batch in dataloader:
        x = batch["x"].to(model.device)
        y = batch["y"].to(model.device)
        attn_mask = batch["pad_mask"].to(model.device)
        loss = model.compute_loss(x, y, attn_mask)
        total_loss += loss.item() / len(dataloader)
    return total_loss


def log_gradients(model, sw, step):
    for name, param in model.named_parameters():
        if param.requires_grad:
            sw.add_histogram(f"{name}.grad", param.grad, step)


def log_text_completions(model, sw, step):
    gen = model.generate(ids, 10, SamplingStrategy(do_sample=True), attn_mask=mask)
    sw.add_text("text_completions", "\n".join(tokenizer.decode_batch(gen)), step)


def main(args):
    manage_dirs(args.ckpts_dir, args.logs_dir)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=args.steps // 10),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps),
        ],
        milestones=[args.steps // 10],
    )


    sw = SummaryWriter(args.logs_dir)

    for step, batch in enumerate(train_dataloader, start=1):
        t0 = time()

        # prepare
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        attn_mask = batch["pad_mask"].to(device)

        # train
        model.train()
        optimizer.zero_grad()
        loss = model.compute_loss(x, y, attn_mask)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        t1 = time()
        tokens_per_second = (args.batch_size * args.context_length) / (t1 - t0)

        # validate
        if step % args.eval_every == 0:
            model.eval()
            validation_loss = validate(model, test_dataloader)
            sw.add_scalar("val/loss", validation_loss, step)

        # log
        print(f"{step}| {loss.item():2f}| tps: {tokens_per_second}| norm: {norm:2f}")
        sw.add_scalar("train/loss", loss.item(), step)
        sw.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

        if step % args.log_every == 0:
            log_gradients(model, sw, step)
            log_text_completions(model, sw, step)

        # save
        if step % args.save_every == 0:
            save_ckpt(step, model, optimizer, scheduler, os.path.join(args.ckpts_dir, f"ckpt_{step}.pt"))

        if step == args.steps:
            break

    model.save(os.path.join(args.ckpts_dir, f"model_{step}.pt"))

    sw.close()


if __name__ == "__main__":
    tokenizer = Tokenizer.init_and_load("/Users/maksimkoltugin/Dev/huawei_LLM_test_task/checkpoints/tokenizer/tokenizer_15k_10k_uncased.pkl")

    list_of_texts = [
        "What is a piece of text?",
        "A text is a passage of words that conveys a set of meanings.",
        "To put it as simply as possible, it is a group of words.",
    ]
    ids, mask = tokenizer(list_of_texts)

    main(parse_args())
