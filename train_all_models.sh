#! /bin/bash

echo "Training all models (CNN, CNN_2, EfficientNet)"

python train_model.py --model_name CNN --data_folder data/ --device mps --num_epochs 15 --batch_size 32
python train_model.py --model_name CNN_2 --data_folder data/ --device mps --num_epochs 15 --batch_size 32
python train_model.py --model_name EfficientNet --data_folder data/ --device mps --num_epochs 15 --batch_size 32