# GANnotation
## Face to face synthesis using Generative Adversarial Networks ([model](https://drive.google.com/open?id=1YhpFXME3pnwy_WhgBtyhGhI1Y9WYYEOz))

<a href="https://www.youtube.com/watch?v=-8r7zexg4yg
" target="_blank"><img src="https://esanchezlozano.github.io/files/test_gannotation.gif" 
alt="GANnotation example" width="240" height="180" border="10" /></a>

This is the PyTorch repository for the GANnotation implementation. GANnotation is a landmark-guided face to face synthesis network that incorporates a triple consistency loss to bridge the gap between the input and target distributions

Release v1 (Nov. 2018): Demo showing the performance of our GANnotation

Release v2 will follow soon with the training code

## Requirements

OpenCV --> pip install cv2 [Link](http://opencv-python-tutroals.readthedocs.io/en/latest/)

PyTorch --> follow the steps in [https://pytorch.org/](https://pytorch.org/)

It also requires scipy and matplotlib, and the Python version to be 3.X 


## Use

The model can be downloaded from [https://drive.google.com/open?id=1YhpFXME3pnwy_WhgBtyhGhI1Y9WYYEOz]

Please, see the demo_gannotation.py file for usage


## Contributions

All contributions are welcome

## Citation

Should you use this code or the ideas from the paper, please cite the following paper:

```
@article{Sanchez2018Gannotation,
  title={Triple consistency loss for pairing distributions in GAN-based face synthesis},
  author={Enrique Sanchez and Michel Valstar},
  journal={arXiv preprint arXiv:1811.03492},
  year={2018}
}
```