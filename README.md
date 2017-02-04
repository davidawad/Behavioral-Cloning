# behavioral_cloning

This project is very close to working properly. 

## Generating model

To generate the model based on the training dataset simply run `python network.py`. 


## Driving the car

The `drive.py` file remains relatively unchanged from the provided one. 

After network.py has been run; simply run the following command.

```
python drive.py model.json
```

## Notes

When running `network.py` it currently expects that it is in the same directory as the `data` folder that the simulator normally outputs.  
```
C:\Users\david\Desktop\sel_driving>
├───data
│   └───IMG
├───network.py
```
Depending on whether the data was recorded on windows or not, the paths will not read correctly when training.

Lines `116` and `117` contain the different imread functions for the different path formats.
