models=('FEDformer' 'Autoformer' 'Informer' 'Triformer' 'Pyraformer' 'xLSTM_TS' 'NBEATS' 'NHITS' 'DLinear' 'NLinear' 'FiLM' 'TiDE' 'CycleNet')
features=M

#models_available=(1 2 4 6 7 9 12)
#models_str=""
#for m in ${models_available[@]}; do

#echo "Model: ${models[m]}"

#python -m utils.visualize gradnorms logs/gradnorms1/ --m ${models[m]} --start_color_idx $m

#models_str="$models_str ${models[m]}"

#done

#models_str=${models_str[@]:1}

#python -m utils.visualize gradnorms logs/gradnorms1/ --m $models_str --start_color_idx ${models_available[@]}

models_str=""
start=""
for idx in 2 1 0 3 4; do
  models_str="$models_str ${models[idx]}"
  start="$start $idx"
done
models_str=${models_str[@]:1}
start=${start[@]:1}
python -m utils.visualize autocorr=1 logs/autocorr/ --m $models_str --start_color_idx $start
python -m utils.visualize gradnorms logs/gradnorms1/ --m $models_str --start_color_idx $start

models_str=""
start=""
for idx in 6 7 9 12; do
  models_str="$models_str ${models[idx]}"
  start="$start $idx"
done
models_str=${models_str[@]:1}
start=${start[@]:1}
python -m utils.visualize autocorr=1 logs/autocorr/ --m $models_str --start_color_idx $start
python -m utils.visualize gradnorms logs/gradnorms1/ --m $models_str --start_color_idx $start
