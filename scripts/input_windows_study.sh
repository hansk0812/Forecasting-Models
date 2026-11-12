source beam_search_indices.sh

######################################
#         DIRECTORY NAME
######################################

dname="lags_mining/1"

if [[ ! -d $dname ]]; then
    mkdir $dname
fi

######################################
#        HYPERPARAMETERS
######################################

# 4 Datasets
dirs=(dataset/ETDataset/ETT-small dataset/weather/ dataset/Delhi_NASA_POWER_AgroClimatology/ dataset/ASTGNN/data/PEMS03/)
paths=(ETTm2.csv 1 delhi.csv PEMS03.npz)
dsets=(ETTm2 JenaWeather AGDelhiWeather CATraffic)
nvars=(7 8 12 358)

# Ablation Studies
# Ignoring nlayers for 3 layer NN
# nlayers=("1 0" "1 1" "2 1" "3 2")
H=(720)
L_factors=("0.25" "0.5" "0.75" "1" "2" "4" "8" "16")

######################################
#           ARRAY LENGTHS
######################################
Lengths=(2 1 8)

######################################
#   SAVE ARRAYS' KEYS FOR REFERENCE
######################################

if [[ ! -e $dname/key.txt ]]; then

    echo "Index 1: datasets: ${dsets[@]}" >> $dname/key.txt
    echo "Index 2: H: ${H[@]}" >> $dname/key.txt
    echo "Index 3: L_factors: ${L_factors[@]}" >> $dname/key.txt
    echo "Index 4: nlayers: $(IFS='|'; echo "${nlayers[*]}")" >> $dname/key.txt
    
fi

######################################
#       CHECKPOINTING INDICES
######################################

# Define start indices for each hyperparameter
ckpt=(1 0 3)

# FOR USE INSIDE EXPT SCRIPT
# Optionally choose to save start indices after every experiment
ckpt_fname="$dname/ckpt.array"
# Save start indices to file if it doesn't exist
if [[ ! -e $ckpt_fname ]]; then
    set | grep ^ckpt= > $ckpt_fname
else
    source $ckpt_fname
fi

######################################
#    NUMBER OF PYTHON PROCESSES
######################################

# Define number of hyperparameter experiments per run based on CPU-GPU sizes
nprocs=2


######################################
#   OUTPUT ARRAY FOR INDICES SCRIPT
######################################

# Get results in an output array
output=()

# Run the function
multi_process_expt_indices Lengths $nprocs $ckpt_fname output

# Handle KeyboardInterrupt
indices_bkp=("${indices[@]}")
reset_ckpt() {
    ckpt=("${indices_bkp[@]}")
    set | grep ^ckpt= > $ckpt_fname
    exit
}
# Register the trap for SIGINT (or INT)
trap reset_ckpt SIGINT

######################################
# LOOP THROUGH $nprocs ORDERED INDICES
######################################

if [[ ! -d $dname/logs/ ]]; then
    mkdir $dname/logs/
fi

# Get $nprocs processes one at a time using IFS
for idx in `seq 0 $((${#output[@]}-1))`; do
    echo "Running iteration $(($idx+1)) / ${#output[@]} with $nprocs training calls"
    IFS="," read -r -a processes <<< "${output[idx]}"
    
    for p in `seq 0 $((${#processes[@]}-1))`; do

        read -r -a indices <<< "${processes[p]}"
        indices_bkp=("${indices[@]}")

        seq_len=$(bc <<< "scale=0; ${H[${indices[1]}]}/${L_factors[${indices[2]}]}")
        
        if [[ ${indices[0]} -eq 1 && ${indices[2]} -lt 2 ]]; then
            continue
        fi

        CUDA_VISIBLE_DEVICES=0 python run.py --root_path ${dirs[${indices[0]}]} --data_path ${paths[${indices[0]}]} --model CycleNet --data ${dsets[${indices[0]}]} --features M --is_training 1 --pred_len ${H[${indices[1]}]}  --seq_len $seq_len --label_len 0 --enc_in ${nvars[${indices[0]}]} --dec_in ${nvars[${indices[0]}]} --c_out ${nvars[${indices[0]}]} --itr 1 --factor 64 --e_layers 2 --d_layers 1 --d_model 384 --batch_size 100 --train_epochs 20 --patience 5 --inspect_backward_pass forward --load_from checkpoints/test_CycleNet_random_modes64_${dsets[${indices[0]}]}_ftM_sl\"$seq_len\"_ll0_pl${H[${indices[1]}]}_dm384_nh8_el2_dl1_df2048_fc64_ebtimeF_dtTrue_test_0/checkpoint.pth >> $dname/logs/${dsets[${indices[0]}]}CycleNet"$seq_len"_${H[${indices[1]}]}_forward_gradnorms.txt &
        CUDA_VISIBLE_DEVICES=0 python run.py --root_path ${dirs[${indices[0]}]} --data_path ${paths[${indices[0]}]} --model CycleNet --data ${dsets[${indices[0]}]} --features M --is_training 1 --pred_len ${H[${indices[1]}]}  --seq_len $seq_len --label_len 0 --enc_in ${nvars[${indices[0]}]} --dec_in ${nvars[${indices[0]}]} --c_out ${nvars[${indices[0]}]} --itr 1 --factor 64 --e_layers 2 --d_layers 1 --d_model 384 --batch_size 100 --train_epochs 20 --patience 5 --inspect_backward_pass backward --load_from checkpoints/test_CycleNet_random_modes64_${dsets[${indices[0]}]}_ftM_sl\"$seq_len\"_ll0_pl${H[${indices[1]}]}_dm384_nh8_el2_dl1_df2048_fc64_ebtimeF_dtTrue_test_0/checkpoint.pth >> $dname/logs/${dsets[${indices[0]}]}CycleNet"$seq_len"_${H[${indices[1]}]}_backward_gradnorms.txt &
        
    done
    
    wait $(jobs -p)

######################################
#  NEXT INDEX FOR KEYBOARDINTERRUPTS
######################################

    # Update ckpt in the file
    indices_idx=$((${#indices[@]}-1))
    while [[ ${indices[$indices_idx]} -eq $((${Lengths[$indices_idx]}-1)) && $indices_idx -gt -1 ]]; do
        indices[$indices_idx]=0
        indices_idx=$(($indices_idx-1))
    done
    if [[ $indices_idx -gt -1 ]]; then
        indices[$indices_idx]=$((${indices[$indices_idx]}+1))
    else
        # Use all array lengths for completion
        indices=(${Lengths[@]})
    fi
    ckpt=(${indices[@]})
    set | grep ^ckpt= > $ckpt_fname

done
