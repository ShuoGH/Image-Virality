# Modelling the Image Virality

To record the details of my summer project, to explore the relationship between the content of images and their virality.


## Files:

The jupyter notebooks are basically used for testing demos before building scripts.

- `Load_OriginalDataSet.ipynb`: Load the original data set from the previous paper
- `EDA & Build_dataset.ipynb`: do some EDA and build the data sets: image pair for the siamese net, and the image data sets for the classification
- `Net_Construction.ipynb`: build the Siamese network
- `Preliminary Training.ipynb`: doing some training
- `Classification_task.ipynb`: predicting the subreddit of the images

Scripts:

- `classifier_net.py`: the network used for the classification, including the Alexnet, VGG, resnet and densenet.
- `data_set.py`: define the data set of images. There are two datasets defined here:
  - `Reddit_Img_Pair`: data used for the Siamese Net, return the image pairs
  - `Reddit_Images_for_Classification`: data for the classification task, use the `torchvision.datasets.MNIST` as the reference
- `download_images.py`: scripts used for downloading the images from server
- `feature_extractor.py`: features layers from the CNN models, used for the feature extraction.
- `losses.py`: define the loss for training the siamese network
- `siamese_net.py`: define the siamese net
- `train_classifier.py`
- `train_siamese.py`
- `transforms.py`: define some transforms used on the images
- `utils.py`: some other functions
- `visualisation.py`: some functions related to image visualisation.
- `script.sh`: some command for training and testing the models

## Pipeline

**Classification:**

1. download all the images to form the data set.
2. Get the images from 5 most popular subreddit and make the data set
3. Training the classifier to predict the image subreddit

According to the performance, choose the suitable model for the siamese network. Alexnet is chosen due to its small size and low computation resource demand. Its performance is also not bad.

**Siamese Network for predicting the virality:**

1. build image pairs (500 image pairs for the beginning)
2. build siamese network, combining the Alexnet and Spatial Transformer Network
3. train and test the performance 



## Reference 

1.  H. Lakkaraju, J. McAuley, and J. Leskovec, “What’s in a name? Understanding the Interplay between Titles, Content, and Communities in Social Media,” p. 10.

2. A. Deza and D. Parikh, “Understanding Image Virality,” presented at the Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp. 1818–1826.

3. K. K. Singh and Y. J. Lee, “End-to-End Localization and Ranking for Relative Attributes,” arXiv:1608.02676 [cs], Aug. 2016.

