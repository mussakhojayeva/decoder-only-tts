import torch 
import torch.nn as nn 
from tqdm import tqdm 
from utils import get_mask_from_lengths
import pdb

from IPython.core.debugger import set_trace


def loss_fn(mel_pred, mel_target, end_pred, end_target, mel_lengths):
    mask = ~get_mask_from_lengths(mel_lengths)
    
    mel_target = mel_target.masked_select(mask.unsqueeze(2))
    mel_pred = mel_pred.masked_select(mask.unsqueeze(2))
    
    end_target = end_target.masked_select(mask)
    end_pred = end_pred.squeeze(-1)
    end_pred = end_pred.masked_select(mask)
    mel_loss = nn.L1Loss()(mel_pred, mel_target)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(7))(end_pred, end_target)
    return mel_loss + bce_loss


def train_fn(model, dataloader, optimizer):
    running_loss = 0
    model.train()
    for num, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        text_padded, text_lengths, mel_padded, mel_lengths, end_logits_padded = [
            x.cuda() for x in batch
        ]        
        mel_pred, end_logits_pred = model(text_padded, mel_padded, text_lengths, mel_lengths)
        loss = loss_fn(mel_pred, mel_padded, end_logits_pred, end_logits_padded, mel_lengths)
        running_loss += loss.sum().item()
        loss.sum().backward()
        optimizer.step()
    
    epoch_loss = running_loss/len(dataloader)
    return epoch_loss


def eval_fn(model, dataloader):
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for num, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            text_padded, text_lengths, mel_padded, mel_lengths, end_logits_padded = [
                x.cuda() for x in batch
            ]        
            mel_pred, end_logits_pred = model(text_padded, mel_padded, text_lengths, mel_lengths)
            loss = loss_fn(mel_pred, mel_padded, end_logits_pred, end_logits_padded, mel_lengths)
            running_loss += loss.item()
        
    epoch_loss = running_loss/len(dataloader)
    return epoch_loss

