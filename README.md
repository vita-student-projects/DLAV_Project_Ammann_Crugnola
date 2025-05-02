# DLAV Project

## Milestone 1 hand-in:

### How to run the code
The notebook works from the top to de Bottom Run all the cells senquentialy. You will go trough data download, data manipulation and augmentation, model definition, training and validation, with visualisations of the worst and best case.

In order to just do a prediction with the model we trained you omit the cell that does the training and just run the last cells that load the model named `best_model.pth` and do the prediction and creating the csv file.

### Model Architecture

#### Image encoding:
We use a fully convolutional version of ResNet18 pre-trained on ImageNet. Making the model fully convolutional allows us to use images of any size creating a useful feature extractor.

After the resnet we add an adaptive average pooling layer to reduce the output size to 1x1, and a fully connected layer to reduce the output size to 128. This creates a number of image features that is not predominant in comparison with the motion features with the goal being that the model does not focus too much on image features.

Other explored architectures were:
- EfficientNetB0: slightly worse performance than ResNet18, ADE ~1.9
- ResNet50 + FPN: too large and prone to overfitting, ADE ~2.5

#### Motion encoding:
We use a 2 layer LSTM with 64 hidden units. This proved to be a good trade-off between performance and speed. The LSTM is fed with the motion features extracted from the data.

Other explored architectures were:
- TCN: worse performance, ADE ~2 with ResNet18 backbone

#### Fusion decoder:
To fuse and decode the image and motion features into a predicted trajectory, we use a 2 Layer FCN with 128 hidden units, ReLU activation and dropout. The output is a 2D vector with the predicted x and y coordinated of the trajectory.

### Data inspection and Augmentation
A Heatmap of all trajectories of training and validation data was created to inspect the data. The heatmap shows that the data is not very well balanced, with a lot of trajectories in the center of the image and very few on the edges. This is a problem because the model will learn to predict trajectories in the center of the image and generalization can be an issue.

From past trajectories we extract more motion feautures such as speed and delta x and y coordinates and heading. This is done to give the model more information about the motion of the vehicle.

Further the image data is augmented with random crops, random gaussian blur and ColorJitter. This is done to increase the amount of training data and to make the model more robust to different scenarios. We also explored horizontal flipping with inversing the coordinates but this kept making the model perform worse so we decided to not use it.

### Training

We train the model with a fairly standard setting using Adam optimizer with weight decay and a learning rate of 0.001. The model is trained for 30-40 epochs with a batch size of 64. The loss function used is the mean squared error between the predicted and ground truth trajectories. Additionally a learning rate scheduler is used to continuously reduce the learning rate during training. Early model extraction is used to save the model with the best validation ADE. We also use dropout in addition to the weight decay to regularize the model and prevent overfitting.

In a first phase the training is done with a frozen image backbone to allow the model to learn the motion features. After 10 epochs the image backbone is unfrozen and the model is trained for another 20-30 epochs with a steadily decreasing learning rate. This is done to allow the model to learn the motion features first and then fine-tune the image features.

A lot of different hyperparameters were tried, such as different learning rates, batch sizes, number of epochs and dropout rates. The best performing model was the one given in the notebook.

### Results and Analysis
We analized the results of the model by inspecting the worst and best case results in terms of ADE. The model performs reasonably well in most cases, but it struggles with scenarios where the vehicle is turning or accelerating quickly. The model also struggles with scenarios where red lights are present or the vehicle is stopped or even stopps and accelerates again. This is natural since such a situation seems super hard to predict also for a human given that only one single image is given and not the whole video feed. 

As post-processing we added smoothing to the predicted trajectory using a savitzky-golay filter. This helps to reduce noise in the predicted trajectory and makes it more realistic. The smoothing is done with a window size of 9 and a polynomial order of 3.