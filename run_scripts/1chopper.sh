#!/usr/bin/env bash
# YOU SHOULD CHANGE HalfCheetah to Swimmer, Hopper to test on other two envs
# YOU SHOULD CHANGE exp_name for each subproblem. Otherwise, your result will be wiped.
for i in `seq 0 2`;
do
    echo $i
<<<<<<< HEAD:run_scripts/1chopper.sh
    python ppo_run_sweep.py --env_name Hopper \
                            --exp_name ppo2_unclipped \
=======
    python ppo_run_sweep.py --env_name HalfCheetah \
                            --exp_name ppo_unclipped \
>>>>>>> parent of e4645d3... bash files:run_scripts/1c.sh
                            --exp_num $i \
                            --discount 0.99 \
                            --use_baseline 1 \
                            --use_ppo_obj 1 \
                            --use_clipper 0 \
                            --use_entropy 0 \
                            --use_gae 0 &
done
