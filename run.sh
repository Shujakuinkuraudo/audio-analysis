#! /bin/bash
CUDA_VISIBLE_DEVICES=0 nohup python dl.py --model CNN >> log.txt &
CUDA_VISIBLE_DEVICES=0 nohup python dl.py --model AE >> log.txt &
CUDA_VISIBLE_DEVICES=0 python dl.py --model AE