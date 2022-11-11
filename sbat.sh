for i in 0.00003 0.00005 0.00008; do
    for j in 0.5 1; do
        for g in 10 25; do
            jobs='lr'+_${i}+'tau'+_${j}+'group'+_${g}
            echo ${jobs}
            sbatch --job-name=${jobs} run_train.sh ${i} ${j} ${g}
        done
    done
done