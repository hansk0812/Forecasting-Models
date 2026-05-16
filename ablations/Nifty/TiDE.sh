layers=("1 1" "2 2" "3 3" "4 3")
decoder_temporal_dim=(128 64 32 256)
d_model=(256 384 512)
drop=(0.05 0.2)

if [[ -e ablations/Nifty/TiDE_ckpt.array ]]; then
  . ablations/Nifty/TiDE_ckpt.array
else
  ckpt=(0 0 0 0)
fi

for l in "${layers[@]: ${ckpt[0]}}"; do
  el=${l% *}
  dl=${l#* }

  for t in ${decoder_temporal_dim[@]: ${ckpt[1]}}; do
    for dm in ${d_model[@]: ${ckpt[2]}}; do
      for d in ${drop[@]: ${ckpt[3]}}; do
        
        if [[ -e checkpoints/Exp"$d"_TiDE_random_modes64_NIFTYStocks_ftM_sl120_ll48_pl120_dm"$dm"_nh8_el"$el"_dl"$dl"_df2048_fc"$m"_ebtimeF_dtTrue_Exp"$d"_0/checkpoint.pth ]]; then
          continue
        fi

        python -u run.py   --root_path dataset/NIFTYStocks/   --data_path nifty_v2_.csv   --model TiDE   --data NIFTYStocks   --features M   --seq_len 120 --pred_len 120   --enc_in 15   --dec_in 15   --c_out 15   --itr 1  --batch_size 100 --des "Exp$d" --task_id "Exp$d" --train_epochs 20 --patience 10 --is_training 1 --factor $t --e_layers $el --d_layers $dl --d_model $dm --dropout $d >> ablations/TiDE.txt
        echo "python -u run.py   --root_path dataset/NIFTYStocks/   --data_path nifty_v2_.csv   --model TiDE   --data NIFTYStocks   --features M   --seq_len 120 --pred_len 120   --enc_in 15   --dec_in 15   --c_out 15   --itr 1  --batch_size 100 --des "Exp$d" --task_id "Exp$d" --train_epochs 20 --patience 10 --is_training 1 --factor $t --e_layers $el --d_layers $dl --d_model $dm --dropout $d"
      
        echo "mse: Layers $el $dl ; Decoder temporal dim (factor) $t ; d_model $dm dropout $d" >> ablations/TiDE.txt
        
        ckpt[3]=$(((${ckpt[3]}+1) % ${#drop[@]}))
        set | grep ^ckpt= > ablations/Nifty/TiDE_ckpt.array

      done

      ckpt[2]=$(((${ckpt[2]}+1) % ${#d_model[@]}))
    done

    ckpt[1]=$(((${ckpt[1]}+1) % ${#decoder_temporal_dim[@]}))
  done

  ckpt[0]=$(((${ckpt[0]}+1) % ${#layers[@]}))
done
