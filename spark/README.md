# Apache spark 

This some test exercices for apache spark.

## Requirements

- docker (if local)
- python 3.8
- jupyter

## How to run

Go to [databrics](https://community.cloud.databricks.com/). Add all
Jupyter notebooks in `/databrics` **not** the ones under `/apache_spark` directory. 
Then add the data in the data folder. Now you are able to run the notebooks.

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
pip install -r requrirements.txt
```

Now we need to create our spark master and workers.
To do that run the docker-compose file.

```sh
# To start
docker-compose up -d
```

```sh
# To stop
docker-compose down
```

Note `SPARK_MASTER_URL` env is completely stupid
you cannot change it. Your master node is required to 
be named `spark` or else it will throw a stupid 
error in your worker saying it does not manage to 
connect to your master node. 


To check if both `spark_master` and `spark_worker` 
containers are running.

```sh
docker ps
```

Now you should be able to access the local spark cluster. And you can
go to the first jupyter notebook under `/apache_spark`. 
