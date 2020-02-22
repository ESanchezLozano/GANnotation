# GANnotation
## Face to face synthesis using Generative Adversarial Networks ([model](https://drive.google.com/open?id=1YhpFXME3pnwy_WhgBtyhGhI1Y9WYYEOz)) ([paper](https://arxiv.org/pdf/1811.03492.pdf))

<a href="https://www.youtube.com/watch?v=-8r7zexg4yg
" target="_blank"><img src="https://esanchezlozano.github.io/files/test_gannotation.gif" 
alt="GANnotation example" width="240" height="180" border="10" /></a>

This is the PyTorch repository for the GANnotation implementation. GANnotation is a landmark-guided face to face synthesis network that incorporates a triple consistency loss to bridge the gap between the input and target distributions

Release v1 (Nov. 2018): Demo showing the performance of our GANnotation

Release v2 (Nov. 2019): Training script is already available

Paper has been accepted to FG'2020 as an Oral presentation, see citation below.

NOTE: The re-training of the StarGAN using the triple consistency loss can be found [here](https://github.com/ESanchezLozano/stargan)


## License

Copyright © 2019. All Rights Reserved.

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.

All the material, including source code, is made freely available for non-commercial use under the Creative Commons CC BY-NC 4.0 license. Feel free to use any of the material in your own work, as long as you give us appropriate credit by mentioning the title and author list of our paper.

All the third-party libraries (Python, PyTorch, Torchvision, Numpy, OpenCV, Visdom, Torchnet, GPUtil and SciPy) are owned by their respective authors, and must be used according to their respective licenses. 

## Requirements
* Linux
* Python 3.6 or further 
* PyTorch 1.X with torchvision
* Numpy
* OpenCV (cv2)

For visualization, the code above uses [Visdom](https://github.com/facebookresearch/visdom), through its wrapper [TorchNet](https://github.com/pytorch/tnt). You can alternatively disable the visualization by setting --visdom False when running the code. No source code would be needed in that case. 

This code assigns a CUDA device directly based on availability, with [GPUtil](https://github.com/anderskm/gputil). This option can be disabled by setting --cuda to the target GPU device. Again, no need to have GPUtil installed in such case.

## Test

The model can be downloaded from [https://drive.google.com/open?id=1YhpFXME3pnwy_WhgBtyhGhI1Y9WYYEOz]

Please, see the demo_gannotation.py file for usage


## Contributions

All contributions are welcome

## Citation

Should you use this code or the ideas from the paper, please cite the following paper:

```
@inproceedings{Sanchez2020Gannotation,
  title={A recurrent cycle consistency loss for progressive face-to-face synthesis},
  author={Enrique Sanchez and Michel Valstar},
  booktitle={IEEE Int'l Conf. on Automatic Face and Gesture Recognition (FG)},
  year={2020}
}
```
