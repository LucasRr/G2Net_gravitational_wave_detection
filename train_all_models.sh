#! /bin/bash

echo "Training all models (CNN, CNN_2, EfficientNet)"

python train_model.py --model_name CNN --device mps --num_epochs 15 --batch_size 32
python train_model.py --model_name CNN_2 --device mps --num_epochs 15 --batch_size 32
python train_model.py --model_name EfficientNet  --device mps --num_epochs 15 --batch_size 32