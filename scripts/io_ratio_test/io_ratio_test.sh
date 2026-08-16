source ./beam_search_indices.sh

seq_lens=(10 200 500 1000 1500 2000)
pred_lens=(10 200 500 1000 1500 2000)
n_variates=(`seq 1 2 8`)
d_hidden=(256 512)
n_layers=("1 0" "1 1" "2 1" "2 3")

if [[ ! -e io_ratio_test/key.txt ]]; then

    echo "Index 1: seq_lens: ${seq_lens[@]}" >> io_ratio_test/key.txt
    echo "Index 2: pred_lens: ${pred_lens[@]}" >> io_ratio_test/key.txt
    echo "Index 3: n_variates: ${n_variates[@]}" >> io_ratio_test/key.txt
    echo "Index 4: d_hidden: ${d_hidden[@]}" >> io_ratio_test/key.txt
    echo "Index 5: n_layers: $(IFS='|'; echo "${n_layers[*]}")" >> io_ratio_test/key.txt
    
fi

# Define lengths and number of hyperparameter arrays
Lengths=(6 6 4 2 4)

# Define start indices for each hyperparameter
ckpt=(0 0 0 0 0)

# FOR USE INSIDE EXPT SCRIPT
# Optionally choose to save start indices after every experiment
ckpt_fname="ckpt.array"
# Save start indices to file
set | grep ckpt^= > $ckpt_fname

# Define number of hyperparameter experiments per run based on CPU-GPU sizes
nprocs=8

# Get results in an output array
output=()

# Run the function
multi_process_expt_indices Lengths $nprocs $ckpt_fname output

# Create pipes for saving text alongwith background processes
Models=(CycleNet NHITS FiLM NLinear)
for index in `seq 0 $((${#Models[@]}-1))`; do
    for idx in `seq 0 $(($nprocs-1))`; do
        fname="${Models[index]}-$idx"
        mkfifo "io_ratio_test/$fname"
    done
done

# Get $nprocs processes one at a time using IFS
for idx in `seq 0 $((${#output[@]}-1))`; do
    echo "Running iteration $idx / $((${#output[@]}-1)) with $nprocs training calls"
    IFS="," read -r -a processes <<< "${output[idx]}"
    for p in `seq 0 $((${#processes[@]}-1))`; do

        read -r -a indices <<< "${processes[p]}"
        read -r -a layers <<< "${n_layers[${indices[4]}]}"
        
        exec 3<> io_ratio_test/${Models[0]}-$p

        (CUDA_VISIBLE_DEVICES=0 python -u run.py --root_path dataset/weather/ --data_path 1 --model CycleNet --data JenaWeather --features M --is_training 1 --pred_len ${pred_lens[${indices[1]}]} --seq_len ${seq_lens[${indices[0]}]} --enc_in 8 --dec_in 8 --c_out 8 --itr 1 --factor 64 --e_layers ${layers[0]} --d_layers ${layers[1]} --d_model ${d_hidden[${indices[3]}]} --label_len 0 --batch_size 100 --select_variates ${n_variates[${indices[2]}]} >> "io_ratio_test/${Models[0]}-$p" 2>&1) & 
    done
    
    wait $(jobs -p)
done
