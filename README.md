# PredictingHorseRaces
We create a model to predict whether or not a horse will show , ie finish first, second or third, based on data from Hong Kong horse races.  We use features based on the track and conditions, the horse itself, as well as more advanced features such as an indicator variable that describes whether the horse is the race's favorite and a variable that measures if the horse is racing in a longer horse than it has recently been running in.

To run, first run ModifyData.py to add the user-created features to the dataset, as well as shuffle the values in the dataset.
Then, run PredictHorses.py to create and run the neural network, as well as get the winnings from gambling using this model.


Work.py was made when we changed our problem to predicting whether a horse would show or not before completely changing our codebase.  TrainModel.py and PredictHorses.py are the same code.
