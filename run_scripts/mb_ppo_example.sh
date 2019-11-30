#!/usr/bin/env bash
# YOU SHOULD CHANGE HalfCheetah to Swimmer, Hopper to test on other two envs
# YOU SHOULD CHANGE exp_name for each subproblem. Otherwise, your result will be wiped.
for i in `seq 0 2`;
do
    echo $i
    python mb_ppo_run_sweep.py --env_name HalfCheetah \
                               --exp_name mbppo_new \
                               --exp_num $i \
                               --ensemble 1
done