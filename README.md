# RNN-MNIST
Try to predict handwritten digit with RNN

## Config
* LEARNING_RATE = 0.001
* NUM_EPOCH = 10
* KEPP_PROB = 0.8
* BATCH_SIZE = 32
* TIME_STEPS = 28  
* INPUT_SIZE = 28 
* NUM_UNITS = [128, 64, 32]
* NUM_LAYERS = 3

## Result

```
Epoch 1 : Step 1718 => Train Loss: 0.1850 | Train ACC: 0.9375: 100%|██████████| 1718/1718 [00:24<00:00, 69.04it/s]
Epoch 1 : Step 311 => Val Loss: 0.2677 | Val ACC: 0.9259 
Epoch 2 : Step 1718 => Train Loss: 0.3416 | Train ACC: 0.9375: 100%|██████████| 1718/1718 [00:24<00:00, 70.01it/s]
Epoch 2 : Step 311 => Val Loss: 0.2209 | Val ACC: 0.9383 
Epoch 3 : Step 1718 => Train Loss: 0.3437 | Train ACC: 0.8438: 100%|██████████| 1718/1718 [00:24<00:00, 71.46it/s]
Epoch 3 : Step 311 => Val Loss: 0.1923 | Val ACC: 0.9452 
Epoch 4 : Step 1718 => Train Loss: 0.1422 | Train ACC: 0.9062: 100%|██████████| 1718/1718 [00:24<00:00, 71.37it/s]
Epoch 4 : Step 311 => Val Loss: 0.1639 | Val ACC: 0.9535 
Epoch 5 : Step 1718 => Train Loss: 0.2446 | Train ACC: 0.9062: 100%|██████████| 1718/1718 [00:24<00:00, 69.43it/s]
Epoch 5 : Step 311 => Val Loss: 0.1604 | Val ACC: 0.9571 
```

## TODO
