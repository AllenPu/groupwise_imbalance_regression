for i in 0.00003 0.00005, 0.00008; do
    for j in 0.5, 1; do
        for g in 10, 25; do
            sbatch --job-name= run.sh ${i} ${j} ${g}
        done
    done
done