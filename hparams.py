meta_file="data/metadata_RUSLAN_22200.csv"
dumpdir="data/RUSLAN_dump"
################################
# Input Parameters             #
# ###############################
num_mels=80
sr=24000
hop_size=300
fft_size=2048
win_length=1200
preemphasis=0 ## check
max_seq_len=2048
ref_db=0 ## check
max_db=0 ## check
################################
# Model Parameters             #
# ###############################
bce_weights=7
batch_size=32
n_epochs=200
hidden_dim=512
n_heads=4
n_layers=6
att_num_buckets=32
