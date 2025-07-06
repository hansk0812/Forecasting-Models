models=('FEDformer' 'Autoformer' 'Informer' 'Triformer' 'Pyraformer' 'xLSTM_TS' 'NBEATS' 'NHITS' 'DLinear' 'NLinear' 'FiLM' 'TiDE')
features=M
ln=(96 192 336 720)

if [ "$features" = "S" ]; then
  ft=(S S S S S S S S S S S S)
  b=()
  e=1
else
  ft=(M M M M M SM SM M M M M M)
  b=(
    "800;430;260;120"
	"430;240;210;190"
	"1150;530;340;220"
	""
	"90;150;100;100"
	"2600;1500;750;300"
	"100;100;100;100"
	"100;100;100;100"
	"1000;800;600;400"
	"1000;800;600;400"
	"2500;1600;800;170"
	"2000;1700;1500;1200"
  )
  bP=(
	"800;430;260;120"
	"430;240;210;190"
	"1150;530;340;220"
	"60;30;10;6"
	"90;150;100;100"
	"900;450;400;90"
	"100;100;100;100"
	"100;100;100;100"
	"1000;800;600;400"
	"1000;800;600;400"
	"2500;1600;75;170"
	"2000;1700;1500;1200"
  )
  e=7
fi

for m in `seq 0 11`; do
for l in `seq 0 3`; do

echo "Model: ${models[m]}, H=${ln[l]}"

arr1=${b[m]}
arr2=${bP[m]}

IFS=";" read -r -a arr1 <<< "${arr1}"
IFS=";" read -r -a arr2 <<< "${arr2}"

python -u run.py   --is_training 1   --root_path ../ETDataset/ETT-small/   --data_path ETTm2.csv   --model ${models[m]}   --data ETTm2   --features ${ft[m]}   --pred_len ${ln[l]}   --enc_in $e   --dec_in $e   --c_out $e   --itr 1   --model_params_json trained_models.json --batch_size ${arr1[l]} --train_epochs 300 --patience 100 #>> logs/${models[m]}_${ft[m]}_orig_$l.txt

python -u run.py   --is_training 1   --root_path ../ETDataset/ETT-small/   --data_path ETTm2.csv   --model LHF/${models[m]}   --data ETTm2   --features ${ft[m]}   --pred_len ${ln[l]}   --enc_in $e   --dec_in $e   --c_out $e   --itr 1   --model_params_json trained_models.json --batch_size ${arr2[l]} --patches_size $((ln[l]/4)) --train_epochs 300 --patience 100 #>> logs/${models[m]}_${ft[m]}_lhf4_$l.txt

done
done
