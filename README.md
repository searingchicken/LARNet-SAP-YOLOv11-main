

https://github.com/user-attachments/assets/6077188f-2d50-4cca-8ca2-7a250f7ce0b5

## LARNet-SAP-YOLOv11

## Preparation

We test the code on PyTorch 1.13.0 + CUDA 11.6 + cuDNN 8.2.0.

## The final file path should be the same as the following:
```
┬─ saved_models
│   └─ indoor
│       ├─ LARNet.pth
│       └─ ... (model name)
│   
├─ data
│   └─ RESIDE-IN
│       ├─ train
│       │   ├─ GT
│       │   │   └─ images1_GT.jpg
│       │   └─ hazy
│       │       └─ images1_hazy.jpg (corresponds to the former)
│       └─ test
│           ├─ GT
│           │   └─ images2_GT.jpg
│           └─ hazy
│               └─ images2_hazy.jpg (corresponds to the former)
│
└─ ressults
    ├─ LARNet-SAP-YOLOv11
    │      └─ images3_GT.jpg (Final result of joint reasoning)
    │
    └─RESIDE-IN
           └─ images4_GT.jpg (The images after clearing)
```

            
```
### Test
Run the following script to test the trained model:
python test.py --model (model name) --dataset (dataset name) --exp (exp name)
```

## For example, we test the LARNet on the test set:
```
python demo.py --model LARNet --dataset RESIDE-IN --exp indoor
```

