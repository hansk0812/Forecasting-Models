#!/bin/bash

folders=(1 2)
layers=(cycleQueue.data model.0.weight model.0.bias model.2.weight model.2.bias model.4.weight model.4.bias)
layer_names=(CycleQueue W0 b0 W1 b1 W2 b2)
names=(cycleQueue ReLU_MLP0.W ReLU_MLP0.b ReLU_MLP1.W ReLU_MLP1.b MLP2.W MLP2.b)

models=()
permodel_colours=(6 2 7 3 0 4 1 5)
for idx in `seq 0 $((${#folders[@]}-1))`; do
    models[$idx]=$(ls ${folders[$idx]}/logs/*forward_gradnorms.txt)

    if [[ ! -d ${folders[$idx]}/logs/plots ]]; then
        mkdir ${folders[$idx]}/logs/plots
    fi
done

cd ../
for n_idx in `seq 0 $((${#layers[@]}-1))`; do

    for idx in `seq 0 $((${#models[@]}-1))`; do
        lags=(${models[$idx]})
        for m_idx in `seq 0 $((${#lags[@]}-1))`; do
            m=${lags[$m_idx]}
            model=${m##*/}   # Remove the left part.
            model=${model%%_720_forward_*}                          # Remove the right part.    
            
            if [[ $m_idx -ge ${#permodel_colours[@]} ]]; then
                c_idx=$(($m_idx - ${#permodel_colours[@]}))
            else
                c_idx=$m_idx
            fi
            
            #echo "python -m utils.visualize \"gradnorms=(${layers[$n_idx]})\" lags_mining/${folders[$idx]}/logs/ --models $model --plots_dir lags_mining/${folders[$idx]}/logs/plots --name \"(${names[$n_idx]})\" --start_co ${permodel_colours[$c_idx]} --interpolated"
            
            #python -m utils.visualize "gradnorms=(${layers[$n_idx]})" lags_mining/${folders[$idx]}/logs/ --models $model --plots_dir lags_mining/${folders[$idx]}/logs/plots --name "(${names[$n_idx]})" --start_co ${permodel_colours[$c_idx]} --interpolated
        done
    done

done

lags=(45 90 180 360 720 960 1440 2880)
all_models=(ETTm2CycleNet JenaWeatherCycleNet AGDelhiWeatherCycleNet CATrafficCycleNet)

c=1
for m in ${all_models[@]}; do

    model=""
    for l in ${lags[@]}; do
        model+="$m$l "
    done    
    
    if [[ $c -lt 3 ]]; then
        F=1
    else
        F=2
    fi

    #python -m utils.visualize gradnorms lags_mining/$F/logs/ --models $model --start_co ${permodel_colours[@]} --plots_dir lags_mining
 
    c=$(($c+1))
done

for l_idx in `seq 0 $((${#layers[@]}-1))`; do

    c=1
    for m in ${all_models[@]}; do

        model=""
        model_name=""
        for l in ${lags[@]}; do
            model+="$m$l "
            model_name+="$m$l"L="$l"_"${layer_names[$l_idx]},"
        done    
        model_name=${model_name%?}
        
        if [[ $c -lt 3 ]]; then
            F=1
        else
            F=2
        fi
        
        if [[ ! -d lags_mining/${layers[$l_idx]} ]]; then
            mkdir lags_mining/${layers[$l_idx]}
        fi

        echo "python -m utils.visualize \"gradnorms=(${layers[$l_idx]})\" lags_mining/$F/logs/ --models $model --start_co ${permodel_colours[@]} --plots_dir lags_mining/${layers[$l_idx]} --name     \"(${model_name[@]})\" --interpolated"
        continue

        python -m utils.visualize "gradnorms=(${layers[$l_idx]})" lags_mining/$F/logs/ --models $model --start_co ${permodel_colours[@]} --plots_dir lags_mining/${layers[$l_idx]} --name "(${model_name[@]})" --interpolated
     
        c=$(($c+1))
    done

done
