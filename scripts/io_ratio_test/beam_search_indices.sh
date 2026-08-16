multi_process_expt_indices() {

local -n lengths=$1
local nprocs=$2
local ckpt_fname=$3
local -n runs=$4

if [[ -e "$ckpt_fname" ]]; then
  . "$ckpt_fname"
else
  ckpt=($(printf '0 ' {1..${#lengths[@]}}))
fi
local -n arr=ckpt

local combos=0 
local prod=1
for idx in `seq $((${#lengths[@]}-1)) -1 0`; do
    if [[ $idx -eq $((${#lengths[@]}-1)) ]]; then
        combos=$(($combos+${lengths[$idx]}-${ckpt[$idx]}))
    else
        prod=1
        for jdx in `seq $idx $((${#lengths[@]}-1))`; do
            if [[ $idx -eq $jdx ]]; then
                prod=$(($prod*(${lengths[$jdx]}-${ckpt[$jdx]}-1)))
            else
                prod=$(($prod*${lengths[$jdx]}))
            fi
        done
        combos=$(($combos+$prod))
    fi
done

num_runs=$((($combos/$nprocs)+1))
for last_important in `seq 1 $num_runs`; do
  final=()
  for _ in `seq 1 $nprocs`; do

  st="${arr[@]}"
  final+=("$st")
  
  for idx in `seq $((${#arr[@]}-1)) -1 0`; do
    
    if [[ $((${arr[$idx]}+1)) -eq ${lengths[$idx]} ]]; then
      continue
    fi
  
    arr[$idx]=$((${arr[$idx]}+1))

    if [[ $((${idx}+1)) -eq ${#arr[@]} ]]; then
      break
    fi

    if [[ ! $idx -eq $((${#arr[@]}-1)) ]]; then
      for jdx in `seq $(($idx+1)) $((${#arr[@]}-1))`; do
        arr[$jdx]=0
      done
    fi

    break

  done

  done

  if [[ $last_important -eq $num_runs ]]; then
    r=""
    for p_idx in `seq 0 $((($combos%$nprocs)-1))`; do
      r="$r,${final[$p_idx]}"
    done
    runs+=("${r[@]:1}") # First character is comma
  else
    r=""
    for p_idx in `seq 0 $(($nprocs-1))`; do
      r="$r,${final[$p_idx]}"
    done
    runs+=("${r[@]:1}")
  fi

done

}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # This line ONLY runs when executing script.sh directly.

    # Define lengths and number of hyperparameter arrays
    Lengths=(4 2 3 5 10)

    # Define start indices for each hyperparameter
    ckpt=(3 0 1 3 7)

    # FOR USE INSIDE EXPT SCRIPT
    # Optionally choose to save start indices after every experiment
    ckpt_fname="ckpt.array"
    # Save start indices to file
    set | grep ckpt^= > $ckpt_fname

    # Define number of hyperparameter experiments per run based on CPU-GPU sizes
    nprocs=11

    # Get results in an output array
    output=()

    # Run the function
    multi_process_expt_indices Lengths $nprocs $ckpt_fname output

    # Get $nprocs processes one at a time using IFS
    for idx in `seq 0 $((${#output[@]}-1))`; do
        IFS="," read -r -a processes <<< "${output[idx]}"
        for p in `seq 0 $((${#processes[@]}-1))`; do
            array_str="${processes[p]}"
            prefix_str="${array_str// /}"
            echo "${processes[p]}"
            echo "$prefix_str"
        done
        echo ""
    done
    echo "There are a total of ${#output[@]} iterations of $nprocs experiments at a time"

fi
