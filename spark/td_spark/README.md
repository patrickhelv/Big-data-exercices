# Apache spark 

This is another test exercice for apache spark.

## Requirements

- python
- jupyter

## How to run

Go to [colab](https://colab.research.google.com/). Add the jupyter `colab`
file and also add the data in the `data` folder. 

## Run locally using Docker

If you do not want to use databrics, you can try to run them locally 
using docker.

### Create a venv

Venv creation command

```sh
python -m venv venv
./venv/bin/activate
```

Then we pip install all the dependencies

```sh
pip install jupyter
pip install pyspark
```

You can now run the *non-colab* version of the notebook.