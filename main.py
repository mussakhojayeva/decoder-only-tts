from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import torch
from model import DecoderTTS
from load_dataset import PrepareDataset, collate_fn
from sampler import SortedBatchSampler
import trainer
import hparams as hp


def train_val_dataset(dataset, val_split=0.01):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


dataset = PrepareDataset('data/metadata_RUSLAN_22200.csv', 'data/RUSLAN_dump')

datasets = train_val_dataset(dataset)

dataloaders = {x:DataLoader(datasets[x], hp.batch_size, shuffle=True, 
                            num_workers=16, drop_last=True, collate_fn=collate_fn) for x in ['train','val']}

## TODO
'''
sampler = SortedBatchSampler(
32,
shape_file='utt2shape',
sort_in_batch='descending',
sort_batch='descending',
drop_last=True,
)
'''

m = torch.nn.DataParallel(DecoderTTS(idim=hp.hidden_dim, token2id=dataset.token2id).cuda())
optimizer = torch.optim.Adam(m.parameters(), lr=0.0001)
n_epochs = hp.n_epochs

wandb.init(project='transformer-decoder', entity='dhcppc0')
wandb.watch(m)

best_loss = 1e10 
for epoch in range(0, n_epochs):
    train_loss = trainer.train_fn(m, dataloaders['train'], optimizer)
    val_loss = trainer.eval_fn(m, dataloaders['val'])
    print(f'EPOCH -> {epoch+1}/{n_epochs} | TRAIN LOSS = {train_loss} | VAL LOSS = {val_loss} | \n')

    # Log the loss and accuracy values at the end of each epoch
    wandb.log({
        "Epoch": epoch,
        "Train Loss": train_loss,
        "Valid Loss": val_loss
    })
    
    if best_loss > val_loss:
        best_loss = val_loss
        best_model = m.state_dict()
        torch.save(best_model, 'checkpoints/model.pth')


if __name__ == "__main__":
    #main()
    pass
