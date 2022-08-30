import argparse
import os
import time

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset

from evaluate import evaluate
# from apex import amp

use_amp = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if use_amp else torch.float32
np_dtype = np.float16 if dtype==torch.float16 else np.float32

def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    # if use_amp:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print(f"Number of FastSpeech2 Parameters: {num_param/1.0e6:.2f}M")

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    if use_amp:
        vocoder.half()
        # model.half()

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    avg_dur = 0.0
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                start = time.time()
                batch = to_device(batch, device, dtype=dtype)
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
                    # Forward
                    output = model(*(batch[2:]))

                    # Cal Loss
                    losses = Loss(batch, output)
                    total_loss = losses[0]

                    # Backward
                    total_loss = total_loss / grad_acc_step
                    scaler.scale(total_loss).backward()
                    if step % grad_acc_step == 0:

                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                        # optimizer.step_and_update_lr()
                        optimizer._update_learning_rate()
                        scaler.step(optimizer._optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    avg_dur += (time.time() - start)
                if step % log_step == 0:
                    total_bs = batch[3].shape[0] * log_step
                    tput = total_bs / avg_dur
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, LR: {:.3e} ".format(step, total_step, optimizer.lr)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *losses
                    )
                    message3 = f"[{total_bs}, {batch[3].dtype}]Throughput: {tput} samples/s, Duration: {avg_dur:.2f}s"

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + message3 + "\n")

                    outer_bar.write(message1 + message2 + message3)

                    log(train_logger, step, losses=losses)
                    avg_dur = 0.0

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
