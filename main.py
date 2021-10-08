# #!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from tqdm import tqdm
import wandb
import numpy as np
import random, os

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

from model import DecoderTTS
from load_dataset import PrepareDataset, collate_fn
import trainer
import hparams as hp
import writer
SEED=44

def init_process(rank, size, backend='nccl'):
    """ Initialize the distributed environment. """
    
    dist.init_process_group(                                   
        backend=backend,                                         
        init_method='env://',                                   
        world_size=size,                              
        rank=rank                                               
    ) 

def get_model():
    return DecoderTTS(idim=hp.hidden_dim)
    
def get_dataloader(rank, world_size):
   
    dataset = PrepareDataset(hp.meta_file, hp.dumpdir)
    
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    sampler = DistributedSampler(train_set, rank=rank, num_replicas=world_size)
    
    batch_size = hp.batch_size // world_size 
    
    train_loader = DataLoader(train_set, 
                            batch_size=batch_size, 
                            collate_fn=collate_fn,
                            num_workers=8,
                            sampler=sampler)
    val_loader = DataLoader(val_set, 
                            batch_size=batch_size, 
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=8)
    
    return train_loader, val_loader

    
def train(rank, world_size):
    init_process(rank, world_size)
    print(f"Rank {rank}/{world_size} training process initialized.\n")
    # master process gets data
    if rank == 0:
        get_dataloader(rank, world_size)
        get_model()
    
    dist.barrier()
    
    m = get_model()
    m.cuda(rank)
    m = DDP(m, device_ids=[rank])
    train_loader, val_loader = get_dataloader(rank, world_size)
    
    optimizer = torch.optim.Adam(m.parameters(), lr=0.0001  * world_size)
    l1_loss = torch.nn.L1Loss()
    bce_loss = torch.nn.BCEWithLogitsLoss()
    
    n_epochs = hp.n_epochs
    
    if rank == 0: writer.init_wandb(m)
        
    dist.barrier()  
    
    best_loss = 1e10 
    for epoch in range(n_epochs):
        train_loss = trainer.train_fn(m, train_loader, optimizer, l1_loss, bce_loss, rank)
        val_loss, spec_fig, gate_fig, alignment_fig = trainer.eval_fn(m, val_loader, l1_loss, bce_loss, rank)
        print(f'TRAIN LOSS = {sum(train_loss)} | VAL LOSS = {sum(val_loss)} | \n')

        # Log the loss and accuracy values at the end of each epoch
        if rank == 0: writer.log_wandb(epoch, train_loss, val_loss, spec_fig, gate_fig, alignment_fig)
            
        if best_loss > sum(val_loss):
            best_loss = sum(val_loss)
            torch.save(m.state_dict(), 'checkpoints/model_'+str(epoch)+'.pth')
    dist.destroy_process_group()
    
WORLD_SIZE = torch.cuda.device_count()            
def main():
    mp.spawn(train, args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE, join=True)
    
if __name__ == "__main__":
    main()
