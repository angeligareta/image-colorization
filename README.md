<h1 align="center">Image Colorization</h1>
<h4 align="center">Final project of the Scalable Machine Learning course of the EIT Digital data science master at <a href="https://www.kth.se/en">KTH</a></h4>

<p align="center">
  <img alt="KTH" src="https://img.shields.io/badge/EIT%20Digital-KTH-%231954a6?style=flat-square" />  
  <img alt="License" src="https://img.shields.io/github/license/angeligareta/image-colorization?style=flat-square" />
  <img alt="GitHub contributors" src="https://img.shields.io/github/contributors/angeligareta/image-colorization?style=flat-square" />
</p>

## Problem statement

Before Artificial Intelligence, image colorization was reserved to artists that aimed to give the original colors to a picture. In fact, professional image colorization is currently done by hand in Photoshop. Despite it is a challenging problem due to the multiple image conditions that need to be considered, nowadays, deep learning techniques have achieved promising results in this field.

This project aims to study the Image Colorization problem of the Computer Vision branch and implement a Deep Neural Network that is able to colorize black and white images.

## Implementation

First, two image datasets will be collected and pre-processed to be provided as an input for a deep learning model. Next, several Deep Neural Network (DNN) architectures including Convolutional Neural Networks (CNNs) and Autoencoders will be implemented using a machine learning framework. Finally, the evaluation metric Mean Squared Error (MSE) will be selected to compare the implemented models on the selected datasets using hyper-tuning.

The project includes a [documented implementation](notebooks/image_colorization.ipynb) and a [final report](docs/report.pdf).

### Data

Regarding the data, two public datasets were used:

- Flickr 30k dataset: a public dataset which contains 30 thousand images in 200x200 resolution. It has become a standard benchmark for image captioning, but because of the variety of images that it contains, this dataset was used for final training and validation of the models [2].
- Flickr 8k dataset: a subset of the previous dataset containing 8 thousand images in 200x200 resolution. It was used mostly for initial training of the models to understand the performance of the implemented models [3].

### Tools

The tools utilized for this project include:

- Python 3.7.
- Tensorflow 2.4 with Keras: open-source framework for machine learning and deep learning applications. It was used to preprocess input data, create data pipelines and train models with different Keras layers.
- Multiple python libraries, such as NumPy and matplotlib.

## Usage

### Training the model

In order to train the model, the following steps should be followed:

1. Download [Flickr 30k dataset](https://www.kaggle.com/adityajn105/flickr30k) and export the images in to a new folder called _ImageLarge_ in the current [data](data) folder.
2. Select training parameters found in the _Train Stage_ section on the notebook.
3. Fit the model with the new data and configuration. Optionally, the resulting model can be saved to use in the test stage.

### Running demo

On the notebook developed, the last section was created to use as a demonstration and colorize custom pictures. In order to use it, the following steps need to be followed:

- Download the best weights [cnn_model_last.h5](https://drive.google.com/uc?export=download&id=1KI9fCihX3c2DpU_s6XCJGCngpxZeEcOM) of the model trained and store it in a new folder called _model/cnn_model_last.h5_.
- Run all cells before the _Train Stage_ section and the ones in _Convert custom pictures_ section.
- After that, run the helper function _"predict_and_show"_ can be used to convert the picture contained in the path passed as argument.

Below are presented some examples of the test images [test.jpg](data/test.jpg) and [test2.jpg](data/test2.jpg) found in data folder.

<p align="center">
<img alt="example-1" src="docs/example-1.jpg" width="65%" />
</p>

<p align="center">
<img alt="example-2" src="docs/example-2.jpg" width="65%" />
</p>

## References

[1] CIELAB color space.Wikipedia Article: Advantages paragraph.

[2] Flickr 30k dataset.https://www.kaggle.com/adityajn105/flickr30k.

[3] Flickr 8k dataset with captions.https://www.kaggle.com/kunalgupta2616/flickr-8k-images-with-captions.

[4] tf.data.dataset api.https://www.tensorflow.org/apidocs/python/tf /data/Dataset.

[5] Anwar, S., Tahir, M., Li, C., Mian, A., Khan, F. S., and Muzaffar, A. W.Imagecolorization: A survey and dataset, 2020.

[6] Chollet, F.Xception: Deep learning with depthwise separable convolutions. InProceedings ofthe IEEE conference on computer vision and pattern recognition(2017), pp. 1251–1258.

[7] Lewinson,E.Imagecolorizationusingconvolutionalautoencoders.https://towardsdatascience.com/image-colorization-using-convolutional-autoencoders-fdabc1cb1dbe.

[8] Zhang, R., Isola, P., and Efros, A. A.Colorful image colorization. InEuropean conferenceon computer vision(2016), Springer, pp. 649–666.

## Authors

- Angel Igareta ([angel@igareta.com](mailto:angel@igareta.com))
- Serghei Socolovschi ([serghei@kth.se](mailto:serghei@kth.se))