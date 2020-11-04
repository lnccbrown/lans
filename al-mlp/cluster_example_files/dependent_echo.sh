#!/bin/bash

my_fancy_var='hello'
sbatch --export my_fancy_var=$my_fancy_var /users/afengler/git_repos/nn_likelihoods/cluster_example_files/sbatch_echo_env.sh