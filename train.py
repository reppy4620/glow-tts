import os
from tqdm.auto import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import commons
import models
import utils
from data_utils import TextMelLoader, TextMelCollate
from text_jp.tokenizer import Tokenizer


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '80000'

    hps = utils.get_hparams()
    mp.spawn(train_and_eval, nprocs=n_gpus, args=(n_gpus, hps,))


def train_and_eval(rank, n_gpus, hps):
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextMelLoader(hps.data.training_files, hps.data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    collate_fn = TextMelCollate(1)
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
                              batch_size=hps.train.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
    if rank == 0:
        val_dataset = TextMelLoader(hps.data.validation_files, hps.data)
        val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
                                batch_size=hps.train.batch_size, pin_memory=True,
                                drop_last=True, collate_fn=collate_fn)

    cleaner = Tokenizer()

    generator = models.FlowGenerator(
        n_vocab=len(cleaner),
        out_channels=hps.data.n_mel_channels,
        **hps.model).cuda(rank)
    optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler,
                               dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps,
                               lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    if hps.train.fp16_run:
        generator, optimizer_g._optim = amp.initialize(generator, optimizer_g._optim, opt_level="O1")
    generator = DDP(generator)
    epoch_str = 1
    global_step = 0
    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator,
                                                   optimizer_g)
        epoch_str += 1
        optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
        optimizer_g._update_learning_rate()
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
            _ = utils.load_checkpoint(os.path.join(hps.model_dir, "ddi_G.pth"), generator, optimizer_g)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train(rank, epoch, hps, generator, optimizer_g, train_loader, logger)
            evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger)
            if epoch % 50 == 0:
                utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
        else:
            train(rank, epoch, hps, generator, optimizer_g, train_loader, None, None)


def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger):
    train_loader.sampler.set_epoch(epoch)
    tracker = utils.Tracker()

    generator.train()
    bar = tqdm(desc=f'Epoch: {epoch}')
    for x, x_lengths, y, y_lengths, a1, f2 in train_loader:
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        a1, f2 = a1.cuda(rank, non_blocking=True), f2.cuda(rank, non_blocking=True)

        # Train Generator
        optimizer_g.zero_grad()

        (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, a1, f2,
                                                                                                 y, y_lengths, gen=False)
        l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = commons.duration_loss(logw, logw_, x_lengths)

        loss_gs = [l_mle, l_length]
        loss_g = sum(loss_gs)

        bar.update()
        bar.set_postfix_str(f'Loss: {loss_g.item():.4f}, MLE: {l_mle.item():.4f}, Duration: {l_length.item():.4f}')
        tracker.update(mle=l_mle.item(), dur=l_length.item(), all=loss_g.item())

        if hps.train.fp16_run:
            with amp.scale_loss(loss_g, optimizer_g._optim) as scaled_loss:
                scaled_loss.backward()
            grad_norm = commons.clip_grad_value_(amp.master_params(optimizer_g._optim), 5)
        else:
            loss_g.backward()
            grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
        optimizer_g.step()

        tracker.update(mle=l_mle.item(), dur=l_length.item(), all=loss_g.item())

    if rank == 0:
        logger.info(f'Train Epoch: {epoch}, '
                    f'Loss: {tracker.all.mean():.6f}, '
                    f'MLE Loss: {tracker.mle.mean():.6f}, '
                    f'Duration Loss: {tracker.dur.mean():.6f}')


def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger):
    if rank == 0:
        generator.eval()
        tracker = utils.Tracker()
        with torch.no_grad():
            for batch_idx, (x, x_lengths, y, y_lengths, a1, f2) in enumerate(val_loader):
                x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
                y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
                a1, f2 = a1.cuda(rank, non_blocking=True), f2.cuda(rank, non_blocking=True)

                (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, a1, f2,
                                                                                                         y, y_lengths,
                                                                                                         gen=False)
                l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
                l_length = commons.duration_loss(logw, logw_, x_lengths)

                loss_gs = [l_mle, l_length]
                loss_g = sum(loss_gs)

                tracker.update(mle=l_mle.item(), dur=l_length.item(), all=loss_g.item())
        logger.info(f'Eval Epoch: {epoch}, '
                    f'Loss: {tracker.all.mean():.6f}, '
                    f'MLE Loss: {tracker.mle.mean():.6f}, '
                    f'Duration Loss: {tracker.dur.mean():.6f}')


if __name__ == "__main__":
    main()
