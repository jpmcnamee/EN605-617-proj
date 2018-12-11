## Description
An accelerated version of a k-means algorithm to perform document clustering on a set of document vectors. Expects to be given a binary data file of document vectors.
Number of clusters and vectors is configurable, the program will run untill the clusters have converged or untill the max trials has been reached.
The purpose of the programs is to test the performance of GPU acceleration with a standard clustering algorithm.

## Running
To run the program simply execute the run.sh script. 
That script should build the code if not already done and extract any data if not already done, as well as run through 4 sample tests

## Building
Simply run make clean, and then make
Intermediate object files will be stored in an objects/ directory
The final program should be called cluster.exe

## Data
There is a sample binary compressed data file in the code base; data/fireNew.zip. Simply unzip that file in the data dir (unzip data/fireNew.zip -d data/).

## Usage
./cluster.exe : This will run the program with a small internally generated test set

./cluster.exe "num clusters" "path to data" "num vectors" "max trials" 

num clusters - the value of k in the k-means algorithm

path to data - path to vectors file

num vectors - the max number of vectors to use from the vectors file (-1 for all, this is the default)

max trials - Max number of iterations the algorithm will perfrom (default 10)