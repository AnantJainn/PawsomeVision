# PawsomeVision
<h1 align="center">PawsomeVision ~ Dog Breed Identification using TensorFlow Deep Learning</h1>

<!-- <p align="center">
  <a href="https://user-images.githubusercontent.com/60153018/109568977-54aa2a00-7aee-11eb-8b1d-67016ccac551.png">
    <img width="600" src="https://user-images.githubusercontent.com/60153018/109568977-54aa2a00-7aee-11eb-8b1d-67016ccac551.png">
  </a>
</p> -->

<p align="center">
  <a href="#prerequisites">Prerequisites</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#training-the-model">Training the Model</a> •
  <a href="#evaluating-the-model">Evaluating the Model</a> •
  <a href="#predicting-dog-breeds">Predicting Dog Breeds</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

This project aims to classify dog breeds using deep learning with TensorFlow and a convolutional neural network (CNN). We will use the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) which contains 20,580 images of 120 breeds of dogs.

## Prerequisites

Before running the code, make sure you have the following installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- OpenCV (cv2)
- Matplotlib

You can install the dependencies using `pip`:
pip install tensorflow numpy opencv-python matplotlib


## Dataset

We will be using the Stanford Dogs Dataset, which contains images of 120 dog breeds. You can download the dataset from [here](http://vision.stanford.edu/aditya86/ImageNetDogs/) and extract it to a folder named `dataset`.

## Training the Model

To train the model, run the following command:
python train.py --dataset dataset --model model/dog_breed.model --plot model/plot.png


This will train the model on the dataset and save the trained model to `model/dog_breed.model`. It will also save a plot of the accuracy and loss over time to `model/plot.png`.

## Evaluating the Model

To evaluate the model, run the following command:
python evaluate.py --dataset dataset --model model/dog_breed.model


This will evaluate the trained model on the test set and print out the classification report.

## Predicting Dog Breeds

To predict the breed of a dog using the trained model, run the following command:
python predict.py --model model/dog_breed.model --image examples/dog.jpg


This will load the trained model from `model/dog_breed.model` and predict the breed of the dog in the `examples/dog.jpg` image.

## Credits

This project was inspired by the [Deep Learning for Computer Vision with Python](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/) book by Adrian Rosebrock.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
<p align="center">
  Made with :heart: by [Anant Jain]
</p>



