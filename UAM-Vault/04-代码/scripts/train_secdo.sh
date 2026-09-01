#!/usr/bin/env bash
# SECDO v2 training (cuda:0)
set -e
PY="${PY:-python}"
$PY -m secdo.train mode=pretrain --config secdo/configs/train/pretrain.yaml
$PY -m secdo.train mode=joint --config secdo/configs/train/joint.yaml
$PY -m secdo.train mode=online --config secdo/configs/train/online.yaml
