<img src = 'sudoku_solve.png'
style = "width:800px; height:480px;"/>

First, download the data from [link](https://drive.google.com/drive/folders/1JGJ-G9VsW5L2wqIdB95KB17mFAw2X8xm?usp=sharing)

Put the files `train_data`, `train_labels`, `validation_data`, `validation_labels` into the `data` directory.

Put the files `sudoku_weights.h5` and `inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5` into the models folder.

**To solve a Sudoku board**
```python 
python3 solve.py --model models / sudoku_weights.h5 --image data / tests / 1.png --one
```
**To solve all Sudoku tables in the tests folder**
```python
python3 solve.py --model models / sudoku_weights.h5 --folder data / tests --many
```
**To train the training model**
```python
python3 train.py --train
```
**To generate data numbers 1-9 from sudoku board**

Step 1: Collect 9x9 tabular data

Crawl tables sudoku 9x9 from google, run the command
```python
python crawl.py
```
From the books on the sudoku algorithm (key data for train) (https://book4you.org/s/sudoku%20puzzle)

Step 2: Automatically generate data with digits 1 to 9
```python
python make_raw_data.py --model models / sudoku_weights.h5
```

Step 3: Refine the data manually (of course, due to training error)
