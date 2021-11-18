Word Transcription with audiovisual lip-reading
Project includes audio-only, visula-only and audiovisual models

## Dataset

The results obtained with the proposed model on the [LRW dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html). The coordinates for cropping mouth ROI are suggested as (x1, y1, x2, y2) = (80, 116, 175, 211) in Matlab. Please note that the fixed cropping mouth ROI (FxHxW) = [:, 115:211, 79:175] in python.

## Training

This is the suggested order to train models including video-only model, audio-only model and audiovisual models:

i) Start by training with temporal convolutional backend, you can run the script:

```
CUDA_VISIBLE_DEVICES='' python main.py --path '' --dataset <dataset_path> \
                                       --mode 'temporalConv' \
                                       --batch_size 36 --lr 3e-4 \
                                       --epochs 30
```

ii)Throw away the temporal convolutional backend, freeze the parameters of the frontend and the ResNet and train the LSTM backend, then run the script:

```
CUDA_VISIBLE_DEVICES='' python main.py --path './temporalConv/temporalConv_x.pt' --dataset <dataset_path> \
                                       --mode 'backendGRU' --every-frame \
                                       --batch_size 36 --lr 3e-4 \
                                       --epochs 5
```

iii)Train the whole network end-to-end. You can run the script:

```
CUDA_VISIBLE_DEVICES='' python main.py --path './backendGRU/backendGRU_x.pt' --dataset <dataset_path> \
                                       --mode 'finetuneGRU' --every-frame \
                                       --batch_size 36 --lr 3e-4 \
                                       --epochs 30
```

**Notes**

`every-frame` is activated when the backend module is recurrent neural network.

`dataset` need be correctly specified before running. Code has assumptions on the dataset organisation.

`temporalConv_x.pt` or `backendGRU_x.pt` are the models with best validation performance on step ii) or step iii).

## Models&Accuracy

|Stream        |Accuracy    |
|--------------|------------|
|video-only    |83.39       |
|audio-only    |90.72       |
|audiovisual   |98.38       |
