#!/bin/bash

# DDM -----------------------------------------------

# model="ddm"
# machine="home"
# traindatanalytic=0
# ngraphs=9
# trainfileidx=0
# networkidx=8
# mlekdereps=100
# manifoldlayers=100

# python -u visualization_mlp_performance.py --model $model --machine $machine --traindatanalytic $traindatanalytic --ngraphs $ngraphs --trainfileidx $trainfileidx --networkidx $networkidx --mlekdereps $mlekdereps --manifoldlayers $manifoldlayers


# model="ddm"
# machine="home"
# traindatanalytic=0
# modelidentifier='_100k'
# ngraphs=9
# trainfileidx=-1
# networkidx=2
# mlekdereps=50
# manifoldlayers=10

# python -u visualization_mlp_performance.py --model $model \
#                                            --machine $machine \
#                                            --traindatanalytic $traindatanalytic \
#                                            --ngraphs $ngraphs \
#                                            --trainfileidx $trainfileidx \
#                                            --networkidx $networkidx \
#                                            --mlekdereps $mlekdereps \
#                                            --manifoldlayers $manifoldlayers \
#                                            --modelidentifier $modelidentifier \

# -----------------------------------------------------


# # # DDM SDV ---------------------------------------------

# model="ddm_sdv"
# machine="home"
# traindatanalytic=0
# ngraphs=9
# trainfileidx=0
# networkidx=-2
# mlekdereps=100
# manifoldlayers=100

# python -u visualization_mlp_performance.py --model $model --machine $machine --traindatanalytic $traindatanalytic --ngraphs $ngraphs --trainfileidx $trainfileidx --networkidx $networkidx --mlekdereps $mlekdereps --manifoldlayers $manifoldlayers


model="ddm_sdv"
machine="home"
traindatanalytic=0
modelidentifier='_100k'
ngraphs=9
trainfileidx=0
networkidx=-1
mlekdereps=50
manifoldlayers=10

python -u visualization_mlp_performance.py --model $model \
                                           --machine $machine \
                                           --traindatanalytic $traindatanalytic \
                                           --ngraphs $ngraphs \
                                           --trainfileidx $trainfileidx \
                                           --networkidx $networkidx \
                                           --mlekdereps $mlekdereps \
                                           --manifoldlayers $manifoldlayers \
                                           --modelidentifier $modelidentifier \

# # # -----------------------------------------------------


# ANGLE 2 ---------------------------------------------

# model="angle2"
# machine="home"
# traindatanalytic=0
# ngraphs=9
# trainfileidx=0
# networkidx=-1
# mlekdereps=100
# manifoldlayers=100

# python -u visualization_mlp_performance.py --model $model --machine $machine --traindatanalytic $traindatanalytic --ngraphs $ngraphs --trainfileidx $trainfileidx --networkidx $networkidx --mlekdereps $mlekdereps --manifoldlayers $manifoldlayers


# model="angle2"
# machine="home"
# traindatanalytic=0
# modelidentifier='_100k'
# ngraphs=9
# trainfileidx=0
# networkidx=-1
# mlekdereps=50
# manifoldlayers=10

# python -u visualization_mlp_performance.py --model $model \
#                                            --machine $machine \
#                                            --traindatanalytic $traindatanalytic \
#                                            --ngraphs $ngraphs \
#                                            --trainfileidx $trainfileidx \
#                                            --networkidx $networkidx \
#                                            --mlekdereps $mlekdereps \
#                                            --manifoldlayers $manifoldlayers \
#                                            --modelidentifier $modelidentifier \
# -----------------------------------------------------


# FULL_DDM2 ---------------------------------------------------

# model="full_ddm2"
# machine="home"
# traindatanalytic=0
# ngraphs=9
# trainfileidx=0
# networkidx=-1
# mlekdereps=100
# manifoldlayers=100

# python -u visualization_mlp_performance.py --model $model --machine $machine --traindatanalytic $traindatanalytic --ngraphs $ngraphs --trainfileidx $trainfileidx --networkidx $networkidx --mlekdereps $mlekdereps --manifoldlayers $manifoldlayers


model="full_ddm2"
machine="home"
traindatanalytic=0
modelidentifier='_100k'
ngraphs=9
trainfileidx=0
networkidx=-1
mlekdereps=50
manifoldlayers=10

python -u visualization_mlp_performance.py --model $model \
                                           --machine $machine \
                                           --traindatanalytic $traindatanalytic \
                                           --ngraphs $ngraphs \
                                           --trainfileidx $trainfileidx \
                                           --networkidx $networkidx \
                                           --mlekdereps $mlekdereps \
                                           --manifoldlayers $manifoldlayers \
                                           --modelidentifier $modelidentifier \

# -----------------------------------------------------------

# ORNSTEIN ---------------------------------------------------

# model="ornstein"
# machine="home"
# traindatanalytic=0
# ngraphs=9
# trainfileidx=0
# networkidx=-1
# mlekdereps=100
# manifoldlayers=100

# python -u visualization_mlp_performance.py --model $model --machine $machine --traindatanalytic $traindatanalytic --ngraphs $ngraphs --trainfileidx $trainfileidx --networkidx $networkidx --mlekdereps $mlekdereps --manifoldlayers $manifoldlayers

# model="ornstein"
# machine="home"
# traindatanalytic=0
# modelidentifier='_100k'
# ngraphs=9
# trainfileidx=0
# networkidx=-1
# mlekdereps=50
# manifoldlayers=10

# python -u visualization_mlp_performance.py --model $model \
#                                            --machine $machine \
#                                            --traindatanalytic $traindatanalytic \
#                                            --ngraphs $ngraphs \
#                                            --trainfileidx $trainfileidx \
#                                            --networkidx $networkidx \
#                                            --mlekdereps $mlekdereps \
#                                            --manifoldlayers $manifoldlayers \
#                                            --modelidentifier $modelidentifier \

# -----------------------------------------------------------

# ORNSTEIN ---------------------------------------------------

# model="weibull_cdf"
# machine="home"
# traindatanalytic=0
# ngraphs=9
# trainfileidx=0
# networkidx=-1
# mlekdereps=100
# manifoldlayers=100

# python -u visualization_mlp_performance.py --model $model --machine $machine --traindatanalytic $traindatanalytic --ngraphs $ngraphs --trainfileidx $trainfileidx --networkidx $networkidx --mlekdereps $mlekdereps --manifoldlayers $manifoldlayers


model="weibull_cdf2"
machine="home"
traindatanalytic=0
modelidentifier='_100k'
ngraphs=9
trainfileidx=0
networkidx=-1
mlekdereps=50
manifoldlayers=10

python -u visualization_mlp_performance.py --model $model \
                                           --machine $machine \
                                           --traindatanalytic $traindatanalytic \
                                           --ngraphs $ngraphs \
                                           --trainfileidx $trainfileidx \
                                           --networkidx $networkidx \
                                           --mlekdereps $mlekdereps \
                                           --manifoldlayers $manifoldlayers \
                                           --modelidentifier $modelidentifier \

# -----------------------------------------------------------

# ORNSTEIN ---------------------------------------------------

# model="levy"
# machine="home"
# traindatanalytic=0
# ngraphs=9
# trainfileidx=0
# networkidx=-1
# mlekdereps=100
# manifoldlayers=100

# python -u visualization_mlp_performance.py --model $model --machine $machine --traindatanalytic $traindatanalytic --ngraphs $ngraphs --trainfileidx $trainfileidx --networkidx $networkidx --mlekdereps $mlekdereps --manifoldlayers $manifoldlayers

# model="levy"
# machine="home"
# traindatanalytic=0
# modelidentifier='_100k'
# ngraphs=9
# trainfileidx=0
# networkidx=-1
# mlekdereps=50
# manifoldlayers=10

# python -u visualization_mlp_performance.py --model $model \
#                                            --machine $machine \
#                                            --traindatanalytic $traindatanalytic \
#                                            --ngraphs $ngraphs \
#                                            --trainfileidx $trainfileidx \
#                                            --networkidx $networkidx \
#                                            --mlekdereps $mlekdereps \
#                                            --manifoldlayers $manifoldlayers \
#                                            --modelidentifier $modelidentifier \

# -----------------------------------------------------------

# # # DDM SDV ---------------------------------------------

# model="ddm_sdv"
# machine="home"
# traindatanalytic=0
# ngraphs=9
# trainfileidx=0
# networkidx=2
# mlekdereps=100
# manifoldlayers=100

# python -u visualization_mlp_performance.py --model $model --machine $machine --traindatanalytic $traindatanalytic --ngraphs $ngraphs --trainfileidx $trainfileidx --networkidx $networkidx --mlekdereps $mlekdereps --manifoldlayers $manifoldlayers

# model="ddm_sdv"
# machine="home"
# traindatanalytic=1
# ngraphs=9
# trainfileidx=0
# networkidx=2
# mlekdereps=100
# manifoldlayers=100

# python -u visualization_mlp_performance.py --model $model --machine $machine --traindatanalytic $traindatanalytic --ngraphs $ngraphs --trainfileidx $trainfileidx --networkidx $networkidx --mlekdereps $mlekdereps --manifoldlayers $manifoldlayers

# # # -----------------------------------------------------

