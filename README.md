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

## Milestone 2 hand-in:

### How to run the code
See first milestone, we continued to work in the same file

### Model Architecture

#### Image encoding:
The image encoding part did not change from last time, we spent time trying to fuse the depth and semantic data into the model that we already have, the two main approaches we followed were:
- **Early Fusion**: Try to fuse the depth data and semantic data directly into the image inputs as additional channels since they are very closely related to the image inputs. A lot of modifications of the resnet were necessary and we ran into problems using the pretrained weights fo the resnet for the rgb part and random weights for the additional channels. The best we got was with a ConvNeXt from timm that workt decently well but ended up taking tremenduous training times (around 5' per epoch) with ADE stagnating around 1.8-1.9. The whole training process got a lot less stable when adding in the additional data as well.

- **Late Fusion**: Try to fuse the depth and semantic data in the final fusion decoder. We tried different models to separately and jointly encode depth and semantic data but none of the methods were able to significantly improve the validation ADE and they all took more time to train so the tradeoff seems rather bad. Models we tried are: simple custom made CNN for separate single-channel encoding, fused non pre-trained MobileNet V2 to jointly extract from [Depth, Semantic, Zero] channels.

#### Motion encoding:
We slightly enhanced our 2 layer LSTM with 64 hidden units to accept also an input for x, y acceleration and acceleration norm. This slightly improved the best reached validation ADE (around 0.05 lower).

#### Fusion decoder:
In this milestone, we haded a multimode trajectory prediction. We fuse and decode the image and motion feature into 3 different trajectories with a confidence score for each of them. The fusion is done by a 256 to 256 fully connected layer with ReLU activation and dropout and then 2 other 256 fully connected layer are in charge, one for the multimode trajectory prediction and one for the confidence score calculation. The output is 3 2D vectors with the x and y coordinate of each mode's trajectory, and a confidence score for each trajectory.


### Data inspection and Augmentation
The main changes effected in this milestone fall into this category. First we implemented a clean image-flip that also flips the trajectories, depth and semantic data in the same go. Second we noticed that unpickling and augmenting data in the dataloader was the main bottleneck for training speed, hence we built a data preprocessing pipeline that takes all input data, does motion feature extraction, flipping, normalizing and conversion to tensors for all the training data points. It then stores a [safetensors](https://huggingface.co/docs/safetensors/index) file (new more efficient format used by hugging face so store and load tensor data) for both the flipped and the normal instance of the data. These files are then loaded in the dataloader and directly fed into the model. This virtually doubled our training data and resulted in an improvement in ADE of around 0.15.

We also tried to use the same image augmentation techniques that we used in the last hand-in but they continuously yielded worse results, maybe this is related to the relatively strong normalization that we are using. Also the augmentation slowed down the training again since we could no longer normalize images in the preprocessing but had to do it afterwards in the dataloader.

An other already mentionned change is the addition of augmented acceleration data to the motion features by differentiating the trajectory a second time, as mentionned this resulted in another small gain in ADE.

The last big modification is the addition of a more balanced sampling. Two approaches were tested here and the simpler one proved to yield better results. As mentionned in the last milestones explainations, the data is rather unbalanced between curves and more or less straight lines so we tried to oversample curves a little bit because our model generalizes poorly for a lot of curve scenarios. This again improved ADE by around 0.08-0.1. (for the first approaches we divided into more classes, check [this file](./img/class_heatmap_five_mode.png) with the according [label distribution](./img/five_mode.png)).

### Training

The training process changed a bit now that we have mulitple modes. The loss is calculated with a soft max on the modes weighted with their confidence score. and for the validation we take the best confidence score trajectory.

For the rest, we briefly explored slightly different hyperparameter combinations but with no success.

### Results and Analysis

The  model now manages turns a lot better and the cases with which it still struggles are mostly edge cases i.e. with a red light present that will turn green in the future or with a congested zone ahead that will eventually clear up. Those scenarios are vera hard to guess with the provided information - even for a human. We are a bit sad that we could not do more with the depth and semantic data and would have liked to explore more possibilities.

## Milestone 3 hand-in:

For this milestone we took the model from milestone 2 and used it as-is with the new data. This worked very well, on the mixed data the model performs well and generalizes even better to the public test data. Therefore we decide not to change anything and leave the model like this. The only small change we added was balancing the data a little bit between real and synthetic samples rather than straight lines and curves.

### Results and Analysis
See poster
