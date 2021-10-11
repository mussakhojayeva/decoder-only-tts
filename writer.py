import wandb
import matplotlib.pyplot as plt

def init_wandb(m):
    wandb.init(project='transformer-decoder', entity='dhcppc0')
    wandb.watch(m)

def log_wandb(epoch, train_loss, val_loss, spec_fig, gate_fig, alignment_fig):
    wandb.log({
                "Epoch": epoch,
                "train_l1_loss": train_loss[0],
                "train_bce_loss": train_loss[1],
                "val_l1_loss": val_loss[0],
                "val_bce_loss": val_loss[1],
                "val_spec": wandb.Image(spec_fig),
                "val_gate": wandb.Image(gate_fig),
                "val_alignment": wandb.Image(alignment_fig)
        
            })
    plt.close(spec_fig)
    plt.close(gate_fig)
    plt.close(alignment_fig)

    

        
