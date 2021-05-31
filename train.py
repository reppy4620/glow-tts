import os

import torch

from accelerate import Accelerator
from torch.utils.data import DataLoader

import utils
import models
import commons
from data_utils import TextMelLoader, TextMelCollate
from text_jp.tokenizer import Tokenizer

global_step = 0
accelerator = Accelerator()


def main():

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '80000'

    hps = utils.get_hparams()
    train_and_eval(hps)


def train_and_eval(hps):
    global global_step

    if accelerator.is_main_process:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)

    torch.manual_seed(hps.train.seed)

    train_dataset = TextMelLoader(hps.data.training_files, hps.data)
    collate_fn = TextMelCollate(1)
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
                              batch_size=hps.train.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)
    if accelerator.is_main_process:
        val_dataset = TextMelLoader(hps.data.validation_files, hps.data)
        val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
                                batch_size=hps.train.batch_size, pin_memory=True,
                                drop_last=True, collate_fn=collate_fn)
        val_loader = accelerator.prepare_data_loader(val_loader)

    cleaner = Tokenizer()

    generator = models.FlowGenerator(
        n_vocab=len(cleaner) + getattr(hps.data, "add_blank", False),
        n_accent=hps.data.n_accent,
        out_channels=hps.data.n_mel_channels,
        **hps.model)
    optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler,
                               dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps,
                               lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    generator, optimizer_g, train_loader = accelerator.prepare(generator, optimizer_g, train_loader)
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
        if accelerator.is_main_process:
            train(epoch, hps, generator, optimizer_g, train_loader, logger)
            evaluate(epoch, hps, generator, optimizer_g, val_loader, logger)
            if epoch % 10 == 0:
                utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, "G_{}.pth".format(epoch)))
        else:
            train(epoch, hps, generator, optimizer_g, train_loader, None, None)


def train(epoch, hps, generator, optimizer_g, train_loader, logger):
    global global_step

    generator.train()
    for batch_idx, (x, x_lengths, y, y_lengths, a1, f2) in enumerate(train_loader):

        # Train Generator
        optimizer_g.zero_grad()

        (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, a1, f2,
                                                                                                 y, y_lengths, gen=False)
        l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = commons.duration_loss(logw, logw_, x_lengths)

        loss_gs = [l_mle, l_length]
        loss_g = sum(loss_gs)

        accelerator.backward(loss_g)
        grad_norm = commons.clip_grad_value_(generator.parameters(), 5)
        optimizer_g.step()

        if accelerator.is_main_process:
            if batch_idx % hps.train.log_interval == 0:
                (y_gen, *_), *_ = generator.module(x[:1], x_lengths[:1], a1[:1], f2[:1], gen=True)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(x), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss_g.item()))
                logger.info([x.item() for x in loss_gs] + [global_step, optimizer_g.get_lr()])
        global_step += 1

    if accelerator.is_main_process:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(epoch, hps, generator, optimizer_g, val_loader, logger):
    if accelerator.is_main_process:
        global global_step
        generator.eval()
        losses_tot = []
        with torch.no_grad():
            for batch_idx, (x, x_lengths, y, y_lengths, a1, f2) in enumerate(val_loader):

                (z, z_m, z_logs, logdet, z_mask), (x_m, x_logs, x_mask), (attn, logw, logw_) = generator(x, x_lengths, a1, f2,
                                                                                                         y, y_lengths,
                                                                                                         gen=False)
                l_mle = commons.mle_loss(z, z_m, z_logs, logdet, z_mask)
                l_length = commons.duration_loss(logw, logw_, x_lengths)

                loss_gs = [l_mle, l_length]
                loss_g = sum(loss_gs)

                if batch_idx == 0:
                    losses_tot = loss_gs
                else:
                    losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

                if batch_idx % hps.train.log_interval == 0:
                    logger.info('Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(x), len(val_loader.dataset),
                               100. * batch_idx / len(val_loader),
                        loss_g.item()))
                    logger.info([x.item() for x in loss_gs])
        logger.info('====> Epoch: {}'.format(epoch))


if __name__ == "__main__":
    main()
