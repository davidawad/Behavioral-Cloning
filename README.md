# behavioral_cloning

This project is very close to working properly.



## Model Architecture Design
This network uses a modified version of the popular comma ai architecture published and shared between members of the december cohort.  


## Architecture Characteristics


## Data Preprocessing
This network consumes a generator that performs some careful preprocessing. 

We grab training examples of size `batch_size`. 

For each training example we choose one of the three cameras mounted on the car for training. 

With this image we perform a couple of operation

- translate the image randomly.
- add random shadows
- randomly augment the brightness. 
- crop the top 1/3 and the bottom 1/5 containing the car
- flip half of the images and angles

Here is the code for this generator. (the generator for the validation data is similar but of course does no preprocessing) 

```python
    # grab a random training example
    row = data_points[np.random.randint(len(data_points))]

    # select a random camera image and set path to one of our 3 camera images
    camera_selection = np.random.randint(3)
    impath = row[camera_selection]  # set image path
    angle = float(row[3])           # read steering angle

    # TODO make threshold a constant
    # ignore low angles
    min_ang_threshold = 0.4
    if abs(angle) < .1:
        # for every small angle, flip a coin to see if we use it.
        rand = np.random.uniform()
        if rand > min_ang_threshold: continue

    # center cam
    if (camera_selection == 0):
        shift_ang = 0.

    # left cam
    if (camera_selection == 1):
        shift_ang = .30

    # right cam
    if (camera_selection == 2):
        shift_ang = -.30

    # read our image from the camera of choice
    # impath = os.path.normpath(os.getcwd() + "/data/" + impath).replace(" ", "")
    impath = os.path.normpath(impath).replace(" ", "")
    image = cv2.imread(impath)
    angle = angle + shift_ang

    # translate the image randomly to better simulate road conditions
    image, angle = trans_image(image, angle, 100)

    # add random shadow
    image = add_random_shadows(image)

    # augment brightness
    image = augment_brightness_camera_images(image)

    # do the actual image preprocessing and cropping
    image = preprocess_image(image)

    # flip half the images
    flip_prob = np.random.randint(2)
    if flip_prob > 0:
        image = cv2.flip(image, 1)
        angle = -angle

    # fill batch of data
    batch_images[batch_filled] = image
    batch_steering[batch_filled] = angle
    batch_filled += 1
yield batch_images, batch_steering
```


## Model Training (Include hyperparameter tuning.)

I've read a few articles online that mention to use lower batch sizes and that will generalize better with more epochs. 

I chose an aggressive dropout of .25 and saw moderate success with that after seeing some other articles shared by classmates. 

Using lower learning rates has been much more useful in the long run and has helped both in troubleshooting and testing. 

I've had the best mileage (heh) with using a `samples_per_epoch` that was high and a multiple of the `batch_size`

The performance of my models tends to plateau around 15 `epochs` and doesn't get much better beyond that.

This is by far the most frustrating program I've ever developed.

Starting with the stock udacity data, I began recording additional data using an xbox one controller. 

![](path to xbox controller image)

## Usage 

To generate the model based on the training dataset simply run `python network.py`. 

The `drive.py` file remains relatively unchanged from the provided one. 

After network.py has been run; simply run the following command.

```shell
python drive.py model.json
```

## Notes

When running `network.py` it currently expects that it is in the same directory as the `data` folder that the simulator normally outputs.

This depends on what operating system that you're running this on, for example on windows.
```
C:\Users\david\Desktop\sel_driving>
├───data
│   └───IMG
├───network.py
```
Depending on whether the data was recorded on windows or not, the paths will not read correctly when training.

Lines `116` and `117` contain the different imread functions for the different path formats.


