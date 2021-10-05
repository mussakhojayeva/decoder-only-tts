################################
# Input Parameters             #
# ###############################
num_mels=80
sr=24000
hop_size=300
fft_size=2048
win_length=1200
preemphasis=0 ## check
ref_db=0
max_db=0
################################
# Model Parameters             #
# ###############################
batch_size=32
n_epochs=100
hidden_dim=256
n_heads=4
n_layers=6
ff_dim=1024
