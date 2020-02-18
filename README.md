# CSE517 Final Project

## Data

Fetch the datasets from the SemEval STS tasks 2012-2016 with

```Shell
cd dataset
chmod +x fetch_data.sh
./fetch_data.sh
```

Create a file named sts.csv with all the sentences together.

```Python
python make_dataset.py
```
