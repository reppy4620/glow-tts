import os

import torch
from torch.utils.data import DataLoader

import commons
import models
import utils
from data_utils import TextMelLoader, TextMelCollate
from text_jp.tokenizer import Tokenizer


class FlowGenerator_DDI(models.FlowGenerator):
    """A helper for Data-dependent Initialization"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for f in self.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(True)


def main():
    hps = utils.get_hparams()
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)

    torch.manual_seed(hps.train.seed)

    train_dataset = TextMelLoader(hps.data.training_files, hps.data)
    collate_fn = TextMelCollate(1)
    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=True,
                              batch_size=hps.train.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)
    tokenizer = Tokenizer(state_size=hps.model.state_size)

    generator = FlowGenerator_DDI(
        n_vocab=len(tokenizer),
        out_channels=hps.data.n_mel_channels,
        **hps.model).cuda()
    optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler,
                               dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps,
                               lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    generator.train()
    for batch_idx, (x, x_lengths, y, y_lengths, a1, f2) in enumerate(train_loader):
        x, x_lengths = x.cuda(), x_lengths.cuda()
        y, y_lengths = y.cuda(), y_lengths.cuda()
        a1, f2 = a1.cuda(), f2.cuda()

        _ = generator(x, x_lengths, a1, f2, y, y_lengths, gen=False)
        break

    utils.save_checkpoint(generator, optimizer_g, hps.train.learning_rate, 0, os.path.join(hps.model_dir, "ddi_G.pth"))


if __name__ == "__main__":
    main()
