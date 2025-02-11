# This is the code for the Bloom Filter project of TDT4305

import configparser  # for reading the parameters file
from pathlib import Path  # for paths of files
import time  # for timing
import random
import numpy as np
import math

# Global parameters
parameter_file = 'default_parameters.ini'  # the main parameters file
# the main path were all the data directories are
data_main_directory = Path('data')
# dictionary that holds the input parameters, key = parameter name, value = value
parameters_dictionary = dict()
hash_function_list = dict() #fix global var hash_function_list
Bloom_filter = 0
false_pos = 0
used_pass = {}


# Reads the parameters of the project from the parameter file 'file'
# and stores them to the parameter dictionary 'parameters_dictionary'
def read_parameters():
    config = configparser.ConfigParser()
    config.read(parameter_file)
    for section in config.sections():
        for key in config[section]:
            if key == 'data':
                parameters_dictionary[key] = config[section][key]
            else:
                parameters_dictionary[key] = int(config[section][key])

# main method for the bloom_filter
def bloom_filter(new_pass, small):
    global false_pos

    if small:
        used_pass[new_pass] = True 

    if(not check_bloom_Filter(new_pass)): # checks if a password is in the bloom filter
        hashes = hash_func(new_pass) # if not than hash it and append the hash to the bloom_filter
        for i in range(len(hashes)):
            if(Bloom_filter[hashes[i]] == 0): 
                Bloom_filter[hashes[i]] = 1
    else: # if allready present mark it as allready used
        print("The password : ", new_pass, " is weak please change your password")
        if not new_pass in used_pass:
            false_pos += 1


def estimated_false_positives(n, m, k):
    return (1 - math.exp(-k * n / m)) ** k

def check_bloom_Filter(password):
    # create a check for passwords
    hashes = hash_func(password)
    count = 0
    for i in range(len(hashes)): # checks if for every entry for the hash there is a one
        if Bloom_filter[hashes[i]] == 1:
            count += 1 # counts all the occurences of 1s

    if count == len(hashes):  # if there is not the same amount of 1s with the amount of hashes then return True
        #print("The password : ", password, " could be in the set")
        return True
    else: # this means that the password is not in the set and we need to include it in the bloom_filter
        #print("The password : ", password, " is 100% not in the set")
        return False

# Reads all the passwords one by one simulating a stream and calls the method bloom_filter(new_password)
# for each password read
def read_data(file):
    time_sum = 0
    pass_read = 0
    small = True

    with file.open(encoding='utf-8') as f:
        for line in f:
            pass_read += 1
            new_password = line[:-3]
            ts = time.time()
            bloom_filter(new_password, small)
            te = time.time()
            time_sum += te - ts
            if pass_read > 1000 and small:
                print("They are more than 1000s passwords, they are no longer getting stored")
                small = False

    return pass_read, time_sum  

def isPrime(x):
    count = 0
    for i in range(int(x/2)):
        if x % (i+1) == 0:
            count = count + 1
    return count == 1

def hashs(s, p): # the hash function
    sum = 0
    for k in range(len(s)):
        sum += ord(s[k]) * pow(p, k)
    return sum

def hash_func(s):
    hash_list = []
    for h in range(len(hash_function_list)): # hashes all the passwords
        p = hash_function_list[h]
        
        sum = hashs(s, p) % parameters_dictionary["n"]

        hash_list.append(sum) # appends all hashes in a list

    return hash_list

# Created h number of hash functions
def hash_functions(): # picks a random number of primes depending on the param h and appends them in a list

    hash_function_list = []
    max_p = 10000 # max prime number

    primes = [i for i in range(max_p) if isPrime(i)]
    
    for j in range(parameters_dictionary["h"]):
        
        p = random.choice(primes)
        if(len(hash_function_list) == 0):
            hash_function_list.append(p)
        else:
            while(hash_function_list.__contains__(p)):
                p = random.choice(primes)           
            hash_function_list.append(p)

    return hash_function_list


if __name__ == '__main__':
    # Reading the parameters
    read_parameters()
    # Creating the hash functions
    hash_function_list = hash_functions()
    Bloom_filter = np.zeros(parameters_dictionary["n"])
    # Reading the data
    print("Stream reading...")
    data_file = (data_main_directory /
                 parameters_dictionary['data']).with_suffix('.csv')
    passwords_read, times_sum = read_data(data_file)
    print(passwords_read, "passwords were read and processed in average", times_sum / passwords_read,
          "sec per password\n")
    print("Number of false positive passwords : ", false_pos)
    
    print("Estimated the number of false positive passwords : ", estimated_false_positives(passwords_read, parameters_dictionary["n"], parameters_dictionary["h"]))
