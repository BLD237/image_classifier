# CIFAR-10 Image Classification Using PyTorch

This project implements a convolutional neural network (CNN) for classifying images from the CIFAR-10 dataset using PyTorch.

## Dataset

The model is trained on the **CIFAR-10** dataset, which consists of 60,000 32x32 color images in 10 different classes:
- Plane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Dependencies

Ensure you have the following libraries installed before running the code:
### pip install numpy pillow torch torchvision


## Model Architecture

The CNN consists of:
- **Two convolutional layers** with ReLU activation and max pooling.
- **Three fully connected layers** to process image features.
- **Cross-entropy loss** for classification.
- **Stochastic Gradient Descent (SGD)** optimizer.

## Training the Model

Run the following notebook to train the model:
## main.ipynb

The model is trained for 30 epochs, and the training loss is printed after each epoch.

## Saving & Loading the Model

After training, the model's weights are saved to `net.pth`: torch.save(net.state_dict(), 'net.pth')

To load the trained model: net.load_state_dict(torch.load('net.pth'))

## Evaluating Accuracy

After loading the model, run the evaluation script to compute accuracy on the test dataset: which gave an accuracy of 67.1%

## Predicting Images

To classify new images:

1. Place test images inside the `test/` directory.
2. Run the test cell on the notebook

   
## License

This project is licensed under the MIT License. Feel free to use and modify as needed.

---

Happy coding! 🚀








