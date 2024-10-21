## Files
- `analysis`: notebooks for analyzing data or trained models
- `checkpoints`: checkpoints of trained models for predicting mechanical properties
- `data`: data for training models to predict mechanical properties
- `images`: images visualizing data processing and effectiveness of ML models
- `result_features`: lists of features for each mechanical parameter that were found to be most useful
- `scripts`: code to run data processing and train the model to predict mechanical properties
- `run_model_creation.ipynb`: notebook to run data processing and model training for predicting mechanical properties

## Run from the command line
To start data processing and create a model that predicts a mechanical parameter 'Unobstructed' run:
```commandline
   python scripts/model_creation.py -target_property Unobstructed 
```