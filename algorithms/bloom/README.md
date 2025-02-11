# Bloom filter

This is a implementation of the bloom filter algorithm.

### Requirements

- python
- numpy `(pip install numpy)`

### How to run

In `default_parameters.ini` we have the following. You can change some of the parameters.

- `h:` represents the number of hash functions. (Integer)
- `N`: represents the number of bits representation for your bloom filter. (The amount of signature matrix rows). (Integer)
- `data`: represents the data source files you want to use, you can either change it to `test` for the test dataset. Or you can change it to `passwords` if you want the full dataset.  (By default `test`). (String) 


```ini
[default]
h = 3
N = 500000
data = test
```

After configuring the default `ini` file you can now execute the python file.

```sh
python bloom.py
```

You will get the following results 


```
Stream reading...
The password :  qwerty  is weak please change your password
6 passwords were read and processed in average 2.2172927856445312e-05 sec per password

Number of false positive passwords :  0
Estimated the number of false positive passwords :  4.665348065175461e-14
```

For the original dataset we can see we have the same password twice (`qwerty`). There 
should be no false positive because we managed to catch the password twice using our
bloom filter. 
