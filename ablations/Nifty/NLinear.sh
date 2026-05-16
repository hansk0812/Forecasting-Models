layers=("1 1" "2 2" "3 3" "4 3") #("0 1" "1 1" "2 1" "2 2" "3 2" "3 3" "4 3" "4 4" "4 5" "5 5")
ma_window=(5 13)
d_model=(256 384 512)
drop=(0.05 0.2 0.25)
model_name="NLinear"

if [[ -e ablations/Nifty/"$model_name"_ckpt.array ]]; then
  . ablations/Nifty/"$model_name"_ckpt.array
else
  ckpt=(0 0 0 0)
fi


for l in "${layers[@]: ${ckpt[0]}}"; do
  el=${l% *}
  dl=${l#* }

  for m in ${ma_window[@]: ${ckpt[1]}}; do
    for dm in ${d_model[@]: ${ckpt[2]}}; do
      for d in ${drop[@]: ${ckpt[3]}}; do
        
        if [[ -e checkpoints/Exp"$d"_NLinear_random_modes64_NIFTYStocks_ftM_sl120_ll48_pl120_dm"$dm"_nh8_el"$el"_dl"$dl"_df2048_fc"$m"_ebtimeF_dtTrue_Exp"$d"_0/checkpoint.pth ]]; then
          continue
        fi

        python -u run.py   --root_path dataset/NIFTYStocks/   --data_path nifty_v2_.csv   --model NLinear   --data NIFTYStocks   --features M   --seq_len 120 --pred_len 120   --enc_in 15   --dec_in 15   --c_out 15   --itr 1  --batch_size 100 --des "Exp$d" --task_id "Exp$d" --train_epochs 20 --patience 10 --is_training 1 --factor $m --e_layers $el --d_layers $dl --d_model $dm --dropout $d >> ablations/NLinear.txt
       
        echo "mse: Layers $el $dl ; MA Window (factor) $m ; d_model $dm dropout $d" >> ablations/NLinear.txt
        
        ckpt[3]=$(((${ckpt[3]}+1) % ${#drop[@]}))
        set | grep ^ckpt= > ablations/Nifty/NLinear_ckpt.array

      done

      ckpt[2]=$(((${ckpt[2]}+1) % ${#d_model[@]}))
    done

    ckpt[1]=$(((${ckpt[1]}+1) % ${#ma_window[@]}))
  done

  ckpt[0]=$(((${ckpt[0]}+1) % ${#layers[@]}))
done    
