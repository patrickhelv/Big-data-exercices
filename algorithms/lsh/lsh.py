# This is the code for the LSH project of TDT4305

import configparser  # for reading the parameters file
import sys  # for system errors and printouts
from pathlib import Path  # for paths of files
import os  # for reading the input data
import time  # for timing
import random # for random
import numpy as np # for numpy

# Global parameters
parameter_file = 'default_parameters.ini'  # the main parameters file
data_main_directory = Path('data')  # the main path were all the data directories are
parameters_dictionary = dict()  # dictionary that holds the input parameters, key = parameter name, value = value
document_list = dict()  # dictionary of the input documents, key = document id, value = the document

band = 0


# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'
def read_parameters():
    config = configparser.ConfigParser()    
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == 'data':
                parameters_dictionary[key] = config[section][key]
            elif key == 'naive':
                parameters_dictionary[key] = bool(config[section][key])
            elif key == 't':
                parameters_dictionary[key] = float(config[section][key])
            else:
                parameters_dictionary[key] = int(config[section][key])


# Reads all the documents in the 'data_path' and stores them in the dictionary 'document_list'
def read_data(data_path):
    for (root, dirs, file) in os.walk(data_path):
        for f in file:
            file_path = data_path / f
            doc = open(file_path).read().strip().replace('\n', ' ')
            file_id = int(file_path.stem)
            document_list[file_id] = doc


# Calculates the Jaccard Similarity between two documents represented as sets
def jaccard(doc1, doc2):
    return len(doc1.intersection(doc2)) / float(len(doc1.union(doc2)))


# Define a function to map a 2D matrix coordinate into a 1D index.
def get_triangle_index(i, j, length):
    if i == j:  # that's an error.
        sys.stderr.write("Can't access triangle matrix with i == j")
        sys.exit(1)
    if j < i:  # just swap the values.
        temp = i
        i = j
        j = temp

    # Calculate the index within the triangular array. Taken from pg. 211 of:
    # http://infolab.stanford.edu/~ullman/mmds/ch6.pdf
    # adapted for a 0-based index.
    k = int(i * (length - (i + 1) / 2.0) + j - i) - 1

    return k


# Calculates the similarities of all the combinations of documents and returns the similarity triangular matrix
def naive():
    docs_Sets = []  # holds the set of words of each document

    for doc in document_list.values():
        docs_Sets.append(set(doc.split()))
    # Using triangular array to store the similarities, avoiding half size and similarities of i==j
    num_elems = int(len(docs_Sets) * (len(docs_Sets) - 1) / 2)
    similarity_matrix = [0 for x in range(num_elems)]
    for i in range(len(docs_Sets)):
        for j in range(i + 1, len(docs_Sets)):
            similarity_matrix[get_triangle_index(i, j, len(docs_Sets))] = jaccard(docs_Sets[i], docs_Sets[j])

    return similarity_matrix

# Creates the k-Shingles of each document and returns a list of them
def k_shingles():
    docs_k_shingles = []  # holds the k-shingles of each document

    k = parameters_dictionary["k"]

    for doc in document_list.values():
        shingles = []
        for i in range(len(doc) - k + 1):
            shingles.append(doc[i:i + k])
        docs_k_shingles.append(list(set(shingles)))
    
    print(docs_k_shingles)

    return docs_k_shingles

# Creates a signatures set of the documents from the k-shingles list
def signature_set(k_shingles):
    amount_of_docs = len(k_shingles)
    temp = []
    
    for i in range(len(k_shingles)): # creates a new list with all unique shingles
        for j in range(len(k_shingles[i])):
            temp.append(k_shingles[i][j])
    
    list_of_every_shingle = list(set(temp))
    print(list_of_every_shingle)
    docs_sig_sets = np.zeros((len(list_of_every_shingle), amount_of_docs)) # creates a zero matrix

    shingle_to_index = {shingle: i for i, shingle in enumerate(list_of_every_shingle)} # assign each shingle to and index

    for i, shingle_docs in enumerate(k_shingles):
        
        for shingle in shingle_docs:
            index = shingle_to_index[shingle]
            docs_sig_sets[index, i] = 1 # replaces all each shingle that are in a doc with 1
    
    print(docs_sig_sets)

    return docs_sig_sets, list_of_every_shingle
    
def hash(a, b, p, x): # hash function
    temp = ((a * x + b) % p)
    return temp


# Creates the minHash signatures after simulation of permutations

def minHash(docs_signature_sets):

    # this takes some time with bbc data can take up to 15 mins
    min_hash_signatures = []    
    Prime = 4294967311
    docs = {}
    nb_of_docs = docs_signature_sets.shape[1]
    index_list = list(range(1, len(docs_signature_sets)))

    for i in range(parameters_dictionary["permutations"]): 

        hashs = index_list
        random.shuffle(hashs) #convert it to a list to shuffle the indexes

        index_next_min = 0
        signature = np.full((docs_signature_sets.shape[1]), np.inf) # create a signature vector
        signature_ass = 0

        while(signature_ass < nb_of_docs):
            idx = hashs[index_next_min]
            vect = docs_signature_sets[idx-1] # takes the index in the dictionnary to find the vector of 0 and 1

            index_docs = np.where(vect == 1) # finds every entry that has a 1
    
           
            nb_iterations = len(index_docs[0]) # calculates the number of iterations base on the number of 1
            
        
            for j in range(nb_iterations):
                if (signature[index_docs[0][j]] > index_next_min): # checks if signature is allready min
                    signature[index_docs[0][j]] = index_next_min # if not replace that entry
                    signature_ass += 1
                

            index_next_min += 1

        min_hash_signatures.append(signature) # append the signature when finished

    
    print(min_hash_signatures)
    return min_hash_signatures

def hash_buckets(sig_list, N, lenght_of_docs):
    p = 0
    if(lenght_of_docs > 409):
        primes = 10007
        p = primes
    else:
        primes = [347, 349, 353, 359, 367, 373, 379, 383,  389, 397,  401, 409]
        while(p < len(sig_list)):
            p = random.choice(primes)

    a = random.randint(1, lenght_of_docs+1)
    b = random.randint(1, lenght_of_docs+1)
    while(a == b):
        b = random.randint(1, lenght_of_docs+1)
    
    hashi = 0
    d = {}
    for i in range(N):
        d.update({i: []})
    i = 0
    
    for x in sig_list:
        hashi = (hash(a, b, p, x) % N)
        d[hashi].append(i)
        i += 1
    
    
    return d


# Hashes the MinHash Signature Matrix into buckets and find candidate similar documents
def lsh(m_matrix):
    candidates = []  # list of candidate sets of documents for checking similarity

    k = len(m_matrix)
    r = parameters_dictionary['r']
    if(len(m_matrix) > 20): # finds the band from r
        if(k % 10) == 0:
            b = int(k/r)
        else:
            b = 2
    else:
        if(k % 2) == 0:
            b = int((k/r))
        elif(k % 5) == 0:
            b = int((k/r))
        elif(k % 10) == 0:
            b = int((k/r))


    list_of_sig = []
    temp = ""
    d = {}
    z = 0
    global band 
    band = b 
    for k in range(b): # go through each band
        for i in range(len(m_matrix[0])): # takes each document
            temp = [int(m_matrix[j+k][i]) for j in range(r)] # converts it to int
            string = ''.join(str(j) for j in temp) # convert each int to string
            list_of_sig.append(int(string)) # convert each string to int
        d = hash_buckets(list_of_sig, parameters_dictionary["buckets"], len(m_matrix[0])) # hash each entry

        for i in range(parameters_dictionary["buckets"]): 
            if len(d[i]) > 1: # if there are more than 2 entries in one bucket append it to candidate
                candidates.append((z, d[i]))
        
        list_of_sig = []
        z += 1
    return candidates

def similarity(fraction, band):

    return fraction / band

def fact(n):
    if n == 0:
        return 1
    result = 1 
    for i in range(2, n+1):
        result = result * i
    return result

def nCr(n, r):
    return (fact(n)) / (fact(r) * fact(n - r))


def combination_list(amount_of_docs, list_of_combinations):
    N = nCr(amount_of_docs, 2)
    main_index = 0
    follow_index = 0
    combinations = []

    for i in range(int(N)): # find the combination for all docs
        if i != 0 and (main_index + 1) % (amount_of_docs) == 0:
            follow_index += 1
            main_index = follow_index
        
        combinations.append((list_of_combinations[follow_index],
                             list_of_combinations[main_index+1]))
        main_index += 1
    
    return combinations

# Calculates the similarities of the candidate documents
def candidates_similarities(candidate_docs, min_hash_matrix):
    similarity_matrix = []
    global band

    if(band == 0):
        band = 2
    amount_of_docs = len(min_hash_matrix[0])
    list_of_combinations = []
    
    for i in range(amount_of_docs):
        list_of_combinations.append(i) # find all the combinations of docs

    combinations = combination_list(amount_of_docs, list_of_combinations)

    combination_d = {} # find the combinations for a dictionnary
    for i in range(len(combinations)):
        combination_d.update({(combinations[i]): 0})
    
    for i in range(len(candidate_docs)):
        if(len(candidate_docs[i][1]) == 2):
            # for each candidate doc append 1 if they have appeared multiple times
            combination_d[(candidate_docs[i][1][0], candidate_docs[i][1][1])] += 1 
        else:
            combinations = combination_list(len(candidate_docs[i][1]),
                                            candidate_docs[i][1])
            for j in range(len(combinations)):
                combination_d[combinations[j]] += 1
            
    for i in combination_d:
        similarity_matrix.append({
                                  similarity(combination_d[i], band): i})


    return similarity_matrix


# Returns the document pairs of over t% similarity
def return_results(lsh_similarity_matrix):
    document_pairs = []

    for i in lsh_similarity_matrix: # if over 0.6 append document_pairs
        if list(i.keys())[0] > parameters_dictionary["t"]:
            document_pairs.append(list(i.values())[0])


    return document_pairs


def count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix):
    false_negatives = 0
    false_positives = 0
    
    print()
    print("lsh similarity matrix:")
    print(lsh_similarity_matrix)

    for i in range(len(lsh_similarity_matrix)): # check if false negative and false positive
        if list(lsh_similarity_matrix[i].keys())[0] < naive_similarity_matrix[i]:
            false_negatives += 1
        if list(lsh_similarity_matrix[i].keys())[0] > naive_similarity_matrix[i]:
            false_positives += 1

    return false_negatives, false_positives

def dictionnary_naive_similarity_matrix(naive_similarity_matrix, nb_docs):
    combinations = []
    naive_similarity_dic = []

    start = True
    i = 0

    for j in range(nb_docs - 1):
        while i < nb_docs:
            if start:
                i += j + 1
                start = False
            combinations.append((j, i))
            i += 1
        start = True
        i = 0

    for i in range(len(naive_similarity_matrix)):
    
        naive_similarity_dic.append({naive_similarity_matrix[i]: combinations[i]})
    
    return naive_similarity_dic


# The main method where all code starts
if __name__ == '__main__':
    # Reading the parameters
    read_parameters()

    # Reading the data
    print("Data reading...")
    data_folder = data_main_directory / parameters_dictionary['data']
    t0 = time.time()
    read_data(data_folder)
    document_list = {k: document_list[k] for k in sorted(document_list)}
    t1 = time.time()
    print(len(document_list), "documents were read in", t1 - t0, "sec\n")

    # Naive
    naive_similarity_matrix = []
    if parameters_dictionary['naive']:
        print("Starting to calculate the similarities of documents...")
        t2 = time.time()
        naive_similarity_matrix = naive()
        t3 = time.time()
        print("Calculating the similarities of", len(naive_similarity_matrix),
              "combinations of documents took", t3 - t2, "sec\n")

    # k-Shingles
    print(parameters_dictionary)
    print("Starting to create all k-shingles of the documents...")
    t4 = time.time()
    all_docs_k_shingles = k_shingles()
    t5 = time.time()
    print("Representing documents with k-shingles took", t5 - t4, "sec\n")

    # signatures sets
    print("Starting to create the signatures of the documents...")
    t6 = time.time()
    signature_sets, list_of_every_shingle = signature_set(all_docs_k_shingles)
    t7 = time.time()
    print("Signatures representation took", t7 - t6, "sec\n")

    # Permutations
    print("Starting to simulate the MinHash Signature Matrix...")
    t8 = time.time()
    min_hash_signatures = minHash(signature_sets)
    t9 = time.time()
    print("Simulation of MinHash Signature Matrix took", t9 - t8, "sec\n")

    # LSH
    print("Starting the Locality-Sensitive Hashing...")
    t10 = time.time()
    candidate_docs = lsh(min_hash_signatures)
    t11 = time.time()
    print("LSH took", t11 - t10, "sec\n")

    # Candidate similarities
    print("Starting to calculate similarities of the candidate documents...")
    t12 = time.time()
    lsh_similarity_matrix = candidates_similarities(candidate_docs, min_hash_signatures)
    t13 = time.time()
    print("Candidate documents similarity calculation took", t13 - t12, "sec\n\n")

    # Return the over t similar pairs
    print("Starting to get the pairs of documents with over ", parameters_dictionary['t'], "% similarity...")
    t14 = time.time()
    pairs = return_results(lsh_similarity_matrix)
    t15 = time.time()
    print("The pairs of documents are:\n")
    for p in pairs:
        print(p)
    print("\n")

    naive_similarity_matrix_dic = dictionnary_naive_similarity_matrix(naive_similarity_matrix, len(document_list))

    # Count false negatives and positives
    if parameters_dictionary['naive']:
        print("Starting to calculate the false negatives and positives...")
        print()
        print("naive similarity matrix: ")
        print(naive_similarity_matrix_dic)
        print()
        t16 = time.time()
        false_negatives, false_positives = count_false_neg_and_pos(lsh_similarity_matrix, naive_similarity_matrix)
        t17 = time.time()
        print("False negatives = ", false_negatives, "\nFalse positives = ", false_positives, "\n\n")

    if parameters_dictionary['naive']:
        print("Naive similarity calculation took", t3 - t2, "sec")

    print("LSH process took in total", t13 - t4, "sec")
