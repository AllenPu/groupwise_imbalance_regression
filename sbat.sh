for i in 0.0001 0.0005 0.00005; do
    for j in 0.5; do
        for g in 10 25 50; do
            for d in 18 50; do
                jobs='lr'_${i}_'tau'_${j}_'group'_${g}_'model_depth'_${d}
                echo ${jobs}
                sbatch --job-name=${jobs} run_train.sh ${i} ${j} ${g} ${d}
            done
        done
    done
done