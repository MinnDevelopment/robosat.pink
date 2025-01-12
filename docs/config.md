## config.toml
```
# RoboSat.pink Configuration


# Input channels configuration
# You can, add several channels blocks to compose your input Tensor. Order is meaningful.
#
# name:		dataset subdirectory name
# bands:	bands to keep from sub source. Order is meaningful
# mean:		bands mean value [default, if model pretrained: [0.485, 0.456, 0.406] ]
# std:		bands std value  [default, if model pretrained: [0.229, 0.224, 0.225] ]

[[channels]]
  name   = "images"
  bands = [1, 2, 3]


# Output Classes configuration
# Nota: available colors are either CSS3 colors names or #RRGGBB hexadecimal representation.
# Nota: only support binary classification for now.

[[classes]]
  title = "Building"
  color = "deeppink"



[model]
  # Neurals Network name
  nn = "Albunet"

  # Pretrained on ImageNet
  #pretrained = true

  # Loss function name
  loss = "Lovasz"
  
  # Batch size for training
  #bs = 4

  # Learning rate for the optimizer
  #lr = 0.000025

  # Model internal input tile size [W=ts, H=ts]
  #ts = 512

  # Dataset loader name
  loader = "SemSegTiles"

  # Kind of data augmentation to apply while training
  da = "Strong"

  # Data Augmentation probability
  #dap = 1.0
  
  # Metrics
  metrics = ["iou", "mcc"]
```
