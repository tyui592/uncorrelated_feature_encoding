Uncorrelated feature encoding for faster style transfer
==

**Unofficial Pytorch Implementation of ['Uncorrelated feature encoding for faster style transfer'](https://doi.org/10.1016/j.neunet.2021.03.007)**

```Text
This repository implements the above paper based on vgg19. Please refer to scripts.sh for differences in details.
```

# Usage
* Requirements
  * torch (version: 1.13.0)
  * torchvision (version: 0.14.0)
  * wandb

* Dataset
  * Content Image: [MSCOCO 2014](https://cocodataset.org/#download) (train2014 for train, test2014 for test)
  * Style Image: [Painter By Numbers](https://www.kaggle.com/competitions/painter-by-numbers/data)

# Result
### Training Loss
![training_loss](./imgs/training_losses.png)
*(From the top left to the bottom right.) style loss, content loss, uncorrelation loss and number of nonzero channels from the feature map.  More details: [wandb link](https://api.wandb.ai/links/minssi/bzxc3jqs).*

### Correlation Matrix
Correlation matrix of the feature map extracted from the vgg encoder calculated by the test data. The feature map has 512 channels, and the matrix is normalized to the total number of images.

![correlation_matrix](./imgs/correlation_matrix.png)


### Stylization with channel pruning

The value of accumulating and sorting the absolute values of the channel vectors.
![channel_magnitude](./imgs/sorted_channel_magnitude.png)

The stylization result of the network pruning the channels in the above order of magnitude.
![pruning_stylization](./imgs/stylization_per_prune_channels.png)

**The above results were calculated through the jupyter notebook.**

### Stylization Result
| Content | Style | w/o Uncorrealtion Loss | w/ Uncorrealtion Loss |
| --- | --- | --- | --- |
| ![image1](./imgs/content/18.jpg) | ![image2](./imgs/style/4.jpg) | ![image3](./imgs/without_uncorr/18_4.jpg) | ![imag4](./imgs/with_uncorr/18_4.jpg) |
| ![imag5](./imgs/content/0.jpg) | ![image6](./imgs/style/1.jpg) | ![image7](./imgs/without_uncorr/0_1.jpg) | ![image8](./imgs/with_uncorr/0_1.jpg) |
| ![image9](./imgs/content/4.jpg) | ![image10](./imgs/style/2.jpg) | ![image11](./imgs/without_uncorr/4_2.jpg) | ![image12](./imgs/with_uncorr/4_2.jpg) |
