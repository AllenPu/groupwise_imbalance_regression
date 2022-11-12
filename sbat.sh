for i in 0.00001. 0.0000005; do
    for j in 0.5 1; do
        for g in 10 20 25 30 50; do
            jobs='lr'_${i}_+'tau'_${j}_'group'_${g}
            echo ${jobs}
            sbatch --job-name=${jobs} run_train.sh ${i} ${j} ${g}
        done
    done
done