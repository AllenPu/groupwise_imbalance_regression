for i in 0.001; do
    for j in 0.5; do
        for g in 10 25; do
            for d in 18 50; do
                for e in 80 90 100 150 200; do
                    jobs='lr'_${i}_'tau'_${j}_'group'_${g}_'model_depth'_${d}_'epoch'_${e}
                    echo ${jobs}
                    sbatch --job-name=${jobs} run_train.sh ${i} ${j} ${g} ${d} ${e}
                done
            done
        done
    done
done