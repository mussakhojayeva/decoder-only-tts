import torch 
from tqdm import tqdm 
from utils import get_mask_from_lengths
import pdb
import utils

def loss_fn(mel_pred, mel_target, end_pred, end_target, mel_lengths, l1_loss, bce_loss):
    mask = ~get_mask_from_lengths(mel_lengths)
    
    mel_target = mel_target.masked_select(mask.unsqueeze(2))
    mel_pred = mel_pred.masked_select(mask.unsqueeze(2))
    
    end_target = end_target.masked_select(mask)
    end_pred = end_pred.squeeze(-1)
    end_pred = end_pred.masked_select(mask)
    mel_loss = l1_loss(mel_pred, mel_target)
    end_loss = bce_loss(end_pred, end_target)
    
    return mel_loss, end_loss


def train_fn(model, dataloader, optimizer, l1_loss, bce_loss, device):
    running_loss = [0, 0]
    model.train()
    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        text_padded, text_lengths, mel_padded, mel_lengths, end_logits_padded = [
            x.to(device) for x in batch
        ]
        mel_out, mel_out_post, stop_tokens, att_ws = model(text_padded, mel_padded, text_lengths, mel_lengths)
        mel_loss, end_loss = loss_fn(mel_out_post, mel_padded, stop_tokens, end_logits_padded, mel_lengths, l1_loss, bce_loss)
        running_loss[0] += mel_loss.item()
        running_loss[1] += end_loss.item()
        (mel_loss + end_loss).backward()
        optimizer.step()
    
    epoch_loss = [loss/len(dataloader) for loss in running_loss]
    return epoch_loss


def eval_fn(model, dataloader, l1_loss, bce_loss, device):
    running_loss = [0, 0]
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            text_padded, text_lengths, mel_padded, mel_lengths, end_logits_padded = [
                x.to(device) for x in batch
            ]
            mel_out, mel_out_post, stop_tokens, att_ws = model(text_padded, mel_padded, text_lengths, mel_lengths)
            mel_loss, end_loss = loss_fn(mel_out_post, mel_padded, stop_tokens, end_logits_padded, mel_lengths, l1_loss, bce_loss)
            running_loss[0] += mel_loss.item()
            running_loss[1] += end_loss.item()

        spec_fig = utils.plot_melspec(mel_padded, mel_out, mel_out_post, mel_lengths)
        gate_fig = utils.plot_gate(stop_tokens)
        alignment_fig = utils.plot_alignments(att_ws, mel_lengths)
        
    epoch_loss = [loss/len(dataloader) for loss in running_loss]
    
    return epoch_loss, spec_fig, gate_fig, alignment_fig

