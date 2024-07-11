import argparse
import os
import shutil
import sys

from time import time

import torch
import torch.multiprocessing as mp

from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
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
    parser.add_argument("--steps", type=int, default=1_000)  # 5000
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--grad_accum_start", type=int, default=2)
    parser.add_argument("--grad_accum_end", type=int, default=32)
    parser.add_argument("--gpus", type=str, required=True)

    return parser.parse_args()


def manage_args(args):
    if os.path.exists(args.ckpts_dir):
        shutil.rmtree(args.ckpts_dir)
    if os.path.exists(args.logs_dir):
        shutil.rmtree(args.logs_dir)

    os.makedirs(args.ckpts_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    args.gpus = [int(gpu) for gpu in args.gpus.split(",")]

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


def log_gradients(model, sw, step):
    for name, param in model.named_parameters():
        if param.requires_grad:
            sw.add_histogram(f"{name}.grad", param.grad, step)


def log_text_completions(model, sw, step):
    gen = model.generate(ids, 10, SamplingStrategy(do_sample=True), attn_mask=mask)
    sw.add_text("text_completions", "\n".join(tokenizer.decode_batch(gen)), step)
    print("\n".join(tokenizer.decode_batch(gen)))


def get_grad_accum(step, total_steps, grad_accum_start, grad_accum_end):
    return int(step / total_steps * (grad_accum_end - grad_accum_start) + grad_accum_start)


def train(rank, args):
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1234)

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

    if len(args.gpus) > 1:
        init_process_group(
            backend="nccl",
            init_method="tcp://localhost:54321",
            world_size=len(args.gpus),
            rank=rank,
        )
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[rank])
        raw_model = model.module

    else:
        device = torch.device(f"cuda:{args.gpus[0]}")
        model = model.to(device)
        raw_model = model

    is_master_process = rank == 0
    model = torch.compile(model)

    train_dataset = WikipediaTokenizedDataset(os.path.join(args.dataset_base_dir, "train"))
    test_dataset = WikipediaTokenizedDataset(os.path.join(args.dataset_base_dir, "test"))

    train_sampler = DistributedSampler(train_dataset) if len(args.gpus) > 1 else None
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, shuffle=False, pin_memory=True)  # type: ignore
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
        for i in range(grad_accum):
            batch = next(train_dataloader)

            # prepare
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            attn_mask = batch["pad_mask"].to(device)

            # calc loss
            with torch.cuda.amp.autocast():
                loss = raw_model.compute_loss(x, y, attn_mask) / grad_accum
                loss_accum += loss.detach()
                
                if len(args.gpus) > 1:
                    model.require_backward_grad_sync = (i == grad_accum - 1)
                    
                scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        torch.cuda.synchronize()
        t1 = time()
        dt = t1 - t0
        tokens_processed = args.batch_size * args.context_length * grad_accum
        tokens_per_second = tokens_processed / dt

        if is_master_process:
            # validate
            if step % args.eval_every == 0:
                model.eval()
                validation_loss = validate(raw_model, test_dataloader)
                sw.add_scalar("val/loss", validation_loss, step)

            # log
            print(f"{step}| {loss_accum.item():2f}| tps: {tokens_per_second}| grad_accum: {grad_accum}| time_elapsed: {dt:2f}| tokens_processed: {tokens_processed}")
            sw.add_scalar("train/loss", loss.item() * grad_accum, step)
            sw.add_scalar("lr", optimizer.param_groups[0]["lr"], step)

            if step % args.log_every == 0:
                # log_gradients(model, sw, step)
                log_text_completions(raw_model, sw, step)

            # save
            if step % args.save_every == 0:
                save_ckpt(step, raw_model, optimizer, scheduler, os.path.join(args.ckpts_dir, f"ckpt_{step}.pt"))

    if is_master_process:
        model.save(os.path.join(args.ckpts_dir, f"model_{step}.pt"))

    sw.close()
    destroy_process_group()


if __name__ == "__main__":
    tokenizer = Tokenizer.init_and_load("/data/d2/m.koltyugin/TEST_TASK_LLM/checkpoints/tokenizer/tokenizer_15k_10k_uncased.pkl")

    list_of_texts = [
        "What is a piece of text?",
        "A text is a passage of words that conveys a set of meanings.",
        "To put it as simply as possible, it is a group of words.",
    ]
    ids, mask = tokenizer(list_of_texts)

    args = manage_args(parse_args())

    if len(args.gpus) > 1:
        mp.spawn(
            train,
            nprocs=len(args.gpus),
            args=(args,),
        )
    else:
        train(0, args)
