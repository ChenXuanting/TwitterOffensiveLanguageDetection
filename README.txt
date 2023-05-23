Read the report for project information.

Package required:
1. scikit-learn
2. tqdm
3. regex
4. pytorch
5. cuda for gpu
6. flair

Our data folder: datasets

Our code includes two files:
1. FinalProject.ipynb 
2. utility.py

The jupyter file FinalProject.ipynb is our major file which contains all training and evaluating process.
The python file utility.py contains helper functions for data loading and processing.

The default setting was set for subtask A. Follow the comments to set for subtask B and C.

Each time a model was trained and a prediction was made, they will be saved to folder cached-results.
The FinalProject file runs a single model each time, but results will be saved for ensemble.
When doing ensemble, modify the for loop starting with "for f in files" in the last cell to select desired models.
