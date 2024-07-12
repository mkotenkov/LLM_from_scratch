import argparse
import os
import shutil
import sys

from time import time

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.data import WikipediaTokenizedDataset
from modules.tokenizer import Tokenizer
from modules.transformer import SamplingStrategy, Transformer, TransformerConfig


def parse_args():
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--dataset_base_dir", type=str, required=True)
    parser.add_argument("--ckpts_dir", type=str, required=True)
    parser.add_argument("--logs_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)

    # model config
    parser.add_argument("--vocab_size", type=int, default=15256)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--p_dropout", type=float, default=0.1)

    # training config
    parser.add_argument("--steps", type=int, default=2_000)  # 5000
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--save_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--grad_accum_start", type=int, default=2)
    parser.add_argument("--grad_accum_end", type=int, default=16)

    return parser.parse_args()


def manage_args(args):
    if os.path.exists(args.ckpts_dir):
        shutil.rmtree(args.ckpts_dir)
    if os.path.exists(args.logs_dir):
        shutil.rmtree(args.logs_dir)

    os.makedirs(args.ckpts_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    return args


def save_ckpt(steps, model, optimizer, scheduler, path_to_save):
    print(f"Saving ckpt to {path_to_save}")
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


def log_text_completions(model, tokenizer, ids, mask, sw, step):
    gen = model.generate(ids, 10, SamplingStrategy(do_sample=True), attn_mask=mask)
    sw.add_text("text_completions", "\n".join(tokenizer.decode_batch(gen)), step)
    print("\n".join(tokenizer.decode_batch(gen)))


def get_grad_accum(step, total_steps, grad_accum_start, grad_accum_end):
    return int(step / total_steps * (grad_accum_end - grad_accum_start) + grad_accum_start)


def main(args):
    # for validation:
    tokenizer = Tokenizer.init_and_load(args.tokenizer_path)
    list_of_texts = [
        "What is a piece of text?",
        "A text is a passage of words that conveys a set of meanings.",
        "To put it as simply as possible, it is a group of words.",
    ]
    ids, mask = tokenizer(list_of_texts)

    model = Transformer(
        TransformerConfig(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            context_length=args.context_length,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            p_dropout=args.p_dropout,
        )
    )
    device = "cuda:1"
    model = torch.compile(model).to(device)

    train_dataset = WikipediaTokenizedDataset(os.path.join(args.dataset_base_dir, "train"))
    test_dataset = WikipediaTokenizedDataset(os.path.join(args.dataset_base_dir, "test"))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)  # type: ignore
    train_dataloader = iter(train_dataloader)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)  # type: ignore

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=args.steps // 10),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps),
        ],
        milestones=[args.steps // 10],
    )

    sw = SummaryWriter(args.logs_dir)

    scaler = torch.cuda.amp.GradScaler()

    for step in range(1, args.steps + 1):
        t0 = time()

        # one optimization step
        model.train()
        loss_accum = 0
        grad_accum = get_grad_accum(step, args.steps, args.grad_accum_start, args.grad_accum_end)
        for _ in range(grad_accum):
            batch = next(train_dataloader)

            # prepare
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            attn_mask = batch["pad_mask"].to(device)

            # calc loss
            with torch.cuda.amp.autocast():
                loss = model.compute_loss(x, y, attn_mask) / grad_accum
                loss_accum += loss.detach()
                scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        t1 = time()
        dt = t1 - t0
        tokens_processed = args.batch_size * args.context_length * grad_accum
        tokens_per_second = tokens_processed / dt

        # log
        print(f"{step}| {loss_accum.item():2f}| tps: {tokens_per_second}| grad_accum: {grad_accum}| time_elapsed: {dt:2f}| tokens_processed: {tokens_processed}")
        sw.add_scalar("train/loss", loss.item() * grad_accum, step)
        sw.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

        # validate
        if step % args.eval_every == 0:
            model.eval()
            validation_loss = validate(model, test_dataloader)
            sw.add_scalar("val/loss", validation_loss, step)
            print(f"{step}| valid loss: {validation_loss:2f}")

        if step % args.log_every == 0:
            log_text_completions(model, tokenizer, ids, mask, sw, step)

        # save
        if step % args.save_every == 0:
            save_ckpt(step, model, optimizer, scheduler, os.path.join(args.ckpts_dir, f"ckpt_{step}.pt"))

    model.save(os.path.join(args.ckpts_dir, f"model_{step}.pt"))

    sw.close()


if __name__ == "__main__":
    main(manage_args(parse_args()))
