layers=("1 1" "2 2" "3 3" "4 3") #("0 1" "1 1" "2 1" "2 2" "3 2" "3 3" "4 3" "4 4" "4 5" "5 5")
d_ff=(32 64 128)
factor=(16 32 64)
embed=(0.1 0.5 1)
d_model=(256 384 512)
detail_freq=("(1,2,4)" "(1,4,8)" "(1,8,16)")
w_size=("(32)" "(64)" "(96)")
model_name="FiLM"

if [[ -e ablations/Nifty/"$model_name"_ckpt.array ]]; then
  . ablations/Nifty/"$model_name"_ckpt.array
else
  ckpt=(0 0 0 0 0 0 0)
fi


for l in "${layers[@]: ${ckpt[0]}}"; do
  el=${l% *}
  dl=${l#* }

  for ff in ${d_ff[@]: ${ckpt[1]}}; do
    for fc in ${factor[@]: ${ckpt[2]}}; do
      for e in ${embed[@]: ${ckpt[3]}}; do
        for dm in ${d_model[@]: ${ckpt[4]}}; do
          for df in ${detail_freq[@]: ${ckpt[5]}}; do
            for ws in ${w_size[@]: ${ckpt[6]}}; do
        
              python -u run.py   --root_path dataset/NIFTYStocks/   --data_path nifty_v2_.csv   --model "$model_name"   --data NIFTYStocks   --features M   --seq_len 120 --pred_len 120   --enc_in 15   --dec_in 15   --c_out 15   --itr 1  --batch_size 100 --des "Exp$df" --task_id "Exp$ws" --train_epochs 20 --patience 10 --is_training 1 --factor $fc --e_layers $el --d_layers $dl --d_model $dm --d_ff $ff --embed $e --detail_freq $df --moving_avg $ws  >> ablations/"$model_name".txt
       
              echo "mse: Layers $el $dl ; d_ff $ff ; factor $fc embed $e d_model $dm detail_freq $df moving_avg $ws" >> ablations/"$model_name".txt
        
              ckpt[6]=$(((${ckpt[6]}+1) % ${#w_size[@]}))
              set | grep ^ckpt= > ablations/Nifty/"$model_name"_ckpt.array
            done

            ckpt[5]=$(((${ckpt[5]}+1) % ${#detail_freq[@]}))
          done

          ckpt[4]=$(((${ckpt[4]}+1) % ${#d_model[@]}))
        done

        ckpt[3]=$(((${ckpt[3]}+1) % ${#embed[@]}))
      done

      ckpt[2]=$(((${ckpt[2]}+1) % ${#factor[@]}))
    done

    ckpt[1]=$(((${ckpt[1]}+1) % ${#d_ff[@]}))
  done

  ckpt[0]=$(((${ckpt[0]}+1) % ${#layers[@]}))
done    
