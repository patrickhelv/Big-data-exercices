# LSH

Example implementation of the LSH algorithm.

### Requirements

- python
- numpy `(pip install numpy)`

### How to run

In `default_parameters.ini` we have the following. You can change some of the parameters.

- `k:` represents the number k-Shingles you want to have for each document. (Integer)
- `permutation`: represents the number of permutations. (The amount of signature matrix rows). (Integer)
- `r`: represents the number of bands for the signature matrix. (Integer)
- `buckets`: represents the number of buckets for our hash signature.
- `data`: represents the data source files you want to use, you can either change it to `test` for the test dataset. or `bbc`: if you want the full dataset. (By default `test`). (String)
- `t`: represents the minimum similarity percentage from (0.0 to 1.0) (By default 6.0) (Integer)

```ini
[default]
k = 5
permutations = 10
r = 2
buckets = 10
data = test
naive = true
t = 0.6
```

After configuring the default `ini` file you can now execute the python file.

```sh
python lsh.py
```

You will get the following results 

```
5 documents were read in 0.0010006427764892578 sec

Starting to calculate the similarities of documents...
Calculating the similarities of 10 combinations of documents took 0.0 sec

{'k': 5, 'permutations': 10, 'r': 2, 'buckets': 10, 'data': 'test', 'naive': True, 't': 0.6}
Starting to create all k-shingles of the documents...
[['ys wi', 'with ', 'cat p', 't pla', 'th th', 'he ca', 'the d', 'e cat', 's wit', 'lays ', 'at pl', 'ith t', 'the c', ' cat ', 'e dog', 'plays', ' with', 'h the', ' the ', 'ays w', ' play', 'he do'], ['dog p', 'ys wi', 'with ', 'th th', 'he ca', 'the 
d', 's wit', 'ith t', 'lays ', 'e cat', 'the c', 'e dog', 'plays', ' with', 'h the', ' the ', 'og pl', 'ays w', ' dog ', 'g pla', ' play', 'he do'], ['ys wi', 'with ', 'th th', 'oy pl', 'the b', 'the d', 's wit', 'ith t', 'lays ', 'e dog', 'boy p', 'plays', ' boy ', ' with', 'h the', ' the ', 'ays w', 'he bo', 'e boy', ' play', 'y pla', 'he do'], ['t eat', 'e cat', ' eats', ' a fi', 'the c', 'he ca', 'at ea', 'a fis', ' cat ', 'ts a ', 'cat e', 's a f', 'eats ', 'ats a', ' fish'], [' eats', 'og ea', 'a bon', 'dog e', 's a b', ' bone', ' a bo', 'g eat', 'ts a ', 'the d', 'eats ', 'ats a', 'e dog', 'he do', ' dog ']]        
Representing documents with k-shingles took 0.000997781753540039 sec

Starting to create the signatures of the documents...
['a bon', 'dog p', 'ys wi', 'with ', 'cat p', 't pla', 'th th', 'he ca', 'oy pl', 'the b', 'the d', 'ats a', ' eats', 'e cat', 's wit', 'lays ', 'at pl', 'ith t', ' a fi', 'the c', 'at ea', ' cat ', 'e dog', 'og ea', 'boy p', 't eat', 'plays', 'dog e', 's a b', ' boy ', ' with', 'h the', 'a fis', ' the ', 'ts a ', 's a f', 'og pl', ' fish', 'ays w', 'he bo', ' dog ', 'g pla', 'e boy', ' play', ' bone', ' a bo', 'g eat', 'cat e', 'eats ', 'y pla', 'he do']
[[0. 0. 0. 0. 1.]
...
 [1. 1. 1. 0. 1.]]
Signatures representation took 0.0081329345703125 sec

Starting to simulate the MinHash Signature Matrix...
[array([3., 2., 3., 1., 6.]), array([3., 1., 2., 7., 6.]), array([3., 3., 2., 4., 1.]), array([2., 7., 7., 6., 1.]), array([1., 1., 1., 2., 4.]), array([2., 2., 2., 3., 1.]), array([1., 4., 3., 2., 5.]), array([1., 1., 1., 2., 9.]), array([4., 4., 1., 
2., 4.]), array([1., 1., 1., 4., 4.])]
Simulation of MinHash Signature Matrix took 0.002187013626098633 sec

Starting the Locality-Sensitive Hashing...
LSH took 0.001003265380859375 sec

Starting to calculate similarities of the candidate documents...
Candidate documents similarity calculation took 0.0 sec


Starting to get the pairs of documents with over  0.6 % similarity...
The pairs of documents are:

(0, 1)


Starting to calculate the false negatives and positives...

naive similarity matrix:
[{1.0: (0, 1)}, {0.6666666666666666: (0, 2)}, {0.25: (0, 3)}, {0.25: (0, 4)}, {0.6666666666666666: (1, 2)}, {0.25: (1, 3)}, {0.25: (1, 4)}, {0.1111111111111111: (2, 3)}, {0.25: (2, 4)}, {0.42857142857142855: (3, 4)}]


lsh similarity matrix:
[{1.0: (0, 1)}, {0.6: (0, 2)}, {0.2: (0, 3)}, {0.0: (0, 4)}, {0.6: (1, 2)}, {0.2: (1, 3)}, {0.0: (1, 4)}, {0.0: (2, 3)}, {0.2: (2, 4)}, {0.2: (3, 4)}]
False negatives =  9 
False positives =  0


Naive similarity calculation took 0.0 sec
LSH process took in total 0.012320995330810547 sec
```

We can see that document 0 and 1 has similarty score of 1.0 for the naive method 
and the same score for lsh.
It has also relatively close score with document 0 and 2. 
While document 0 and 4 have similarity score of 0.0 for lsh and 0.25 for the naive method.

We have to keep in mind that the result for LSH will change each runtime, because of the shingle
representation will vary from run to run and thus the signatures for each document will be different. 

For more information about LSH see [here](https://medium.com/@omkarsoak/from-min-hashing-to-locality-sensitive-hashing-the-complete-process-b88b298d71a1).