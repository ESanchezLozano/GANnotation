# GANnotation training scripts

This code is the re-polished version of the training pipeline of GANnotation I developed whilst holding a post-doc position at the University of Nottingham. Due to the clean-up, it might be the case that the models trained under this script are not suitable to be tested under the previously released testing code. A new testing pipeline will follow up.

# Data

Example of training:
```
python train.py --bSize 64 --db 300VW --path '' --use_edge
```

This code uses the 300VW. We have first processed it by extracting its frames and gathering the landmarks in the files pts.pt. In order to run the code as is you need to extract the frames and gather the corresponding points into the pts.pt file. A script to process it might follow-up.

Alternatively, you can use your own data. It is suggested to use the skeleton provided in databases.py as it enables the use of multiple databases in a much easier way:

```
ConcatDataset([SuperDB(path=args.path[i], db=db, size=128, hm_size=128, flip=True, angle=15, tight=int(args.tight), edge=args.use_edge) for i,db in enumerate(args.db)])
```

The skeleton only needs an init function to gather the needed information, and a collect function, that yields a list with images and corresponding points. The rest will be processed within the main SuperDB class.

```
if db == 'Skeleton': # this is an example of what to prepare in a db
  def init(self):
    # - here there's the stuff needed to collect points and images and labels or whatever
    # - they are then set to db as 
    keys = ['frames','images']
    for k in keys:
      setattr(self, k, eval(k)) # if the variables are named after the keys
    setattr(self,'len', lenval )  # set value of len  
  def collect(idx):
    # - function to collect a sample to be processed in getitem
    return [image], [points]
  init(self) # - do the initialisation
  setattr(self,'collect', collect) # - add collect function to the class
```


# Notes

- Use of heatmaps is not supported in this code yet. You should train your network with the flag --use_edge

- This README.md file will be uploaded properly in due time


# LICENSE

Copyright © 2019. The University of Nottingham. All Rights Reserved.

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

All the material, including source code, is made freely available for non-commercial use under the Creative Commons CC BY-NC 4.0 license. Feel free to use any of the material in your own work, as long as you give us appropriate credit by mentioning the title and author list of our paper.

All the third-party libraries (Python, PyTorch, Torchvision, Numpy, OpenCV, Visdom, Torchnet, GPUtil and SciPy) are owned by their respective authors, and must be used according to their respective licenses. The same applies to the models defined in model_GAN.py, mainly adopted from StarGAN.
