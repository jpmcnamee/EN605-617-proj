#include<stdio.h>
#include <stdlib.h>

#include "cluster.h"

/*
    The functions in this file primarily focus on reading and writing data to disk, as well
    as allocating/deleting most host memory for the program.
*/

/*
   Used to deal with binary file in different endian format
   Read a floating point number
   Assume IEEE format
   http://paulbourke.net/dataformats/reading/
*/
int readFloat(FILE *fptr,float *n)
{
   unsigned char *cptr,tmp;

   if (fread(n,4,1,fptr) != 1)
      return 1;
   cptr = (unsigned char *)n;
   tmp = cptr[0];
   cptr[0] = cptr[3];
   cptr[3] =tmp;
   tmp = cptr[1];
   cptr[1] = cptr[2];
   cptr[2] = tmp;

   return 0;
}


/*
   Used to deal with binary file in different endian format
   Read an integer, swapping the bytes
   http://paulbourke.net/dataformats/reading/
*/
int readInt(FILE *fptr,int *n)
{
   unsigned char *cptr,tmp;

   if (fread(n,4,1,fptr) != 1)
      return 1;
   cptr = (unsigned char *)n;
   tmp = cptr[0];
   cptr[0] = cptr[3];
   cptr[3] = tmp;
   tmp = cptr[1];
   cptr[1] = cptr[2];
   cptr[2] = tmp;

   return 0;
}

/*
    Reads a binary file from disk and loads the weighted vectors into a host memory matrix
    in sparse COO format. Alos reads and loads the magnitude for each vector and stores in a 
    separate host vector
*/
int loadData(const char * fileName, sparse_matrix ** matrix, float ** magnitudes, int maxRecords) {
    printf("Will load data from file %s\n", fileName);
    
    int success = 0;
    FILE * vFile = fopen ( fileName , "rb" );
    
    if (vFile==NULL) {
        printf("Error opening file: %s\n", fileName);
        return 1;
    }
    
    int numVectors;
    int nonZeroLength;
    int dictionarySize;
    int read = 0;
    
    //read = fread((void*)(&numVectors), sizeof(numVectors), 1, vFile);
    read += readInt(vFile, &numVectors);
    printf("Reading in file containing %d vectors\n", numVectors);
    
    if (maxRecords > 0 && maxRecords < numVectors) {
        printf("Capping vectors at %d of %d\n", maxRecords, numVectors);
        numVectors = maxRecords;    
    }
    read += readInt(vFile, &nonZeroLength);
    printf("Total vector lengths: %d\n", nonZeroLength);
    
    read += readInt(vFile, &dictionarySize);
    printf("dictionary size: %d\n", dictionarySize);
    
    if (read != 0) {
        printf("Problem reading file\n");
        fclose (vFile);
        return 1;
    }
    
    
    printf("allocating host memory\n");
    *magnitudes = (float*) malloc(numVectors * sizeof(float));
    (*matrix)->rowIndexPtr = (int*) malloc(nonZeroLength * sizeof(int));
    (*matrix)->colIndexPtr = (int*) malloc(nonZeroLength * sizeof(int));
    (*matrix)->valPtr = (float*) malloc(nonZeroLength * sizeof(float));
    (*matrix)->valLength = nonZeroLength;
    (*matrix)->numRows = numVectors;
    (*matrix)->numCols = dictionarySize;

    if (*magnitudes == NULL || (*matrix)->rowIndexPtr == NULL || (*matrix)->colIndexPtr == NULL || (*matrix)->valPtr == NULL) {
        printf("Error allocating host memory\n");
        success = 1;
    } else {
        printf("Reading vectors\n");
        int matrixCounter = 0;
        
        int vectorLength;
        int docID;
        int termID;
        float weight;
        float magnitude;
        int separator;
        
        //Start reading vectors
        for (int i =0; i < numVectors; i++) {
            //Get the document ID
            readInt(vFile, &docID);
            //Get the vector length
            readInt(vFile, &vectorLength);
            //Read TFIDF weights
            
            for (int j = 0; j < vectorLength; j++) {
                //Read term ID (column)
                readInt(vFile, &termID);
                //Read weight (value)
                readFloat(vFile, &weight);
                
                (*matrix)->rowIndexPtr[matrixCounter] = i;
                (*matrix)->colIndexPtr[matrixCounter] = termID;
                (*matrix)->valPtr[matrixCounter] = weight;
                
                matrixCounter++;
            }
            //Read vector magnitude
            readFloat(vFile, &magnitude);
            (*magnitudes)[i] = magnitude;
            
            //2 byte separator between each vector
            fread((void*)(&separator), 2, 1, vFile);
        }
        printf("Finished reading vectors\n");
        
        if (maxRecords > 0 && maxRecords <= numVectors) {
            printf("Capping nonZeroLength at %d of %d\n", matrixCounter, nonZeroLength);
            (*matrix)->valLength = matrixCounter;   
        }
    }
        
        
    fclose (vFile);
    return success;
}

/*
    Used to delete sparse matrix allocations in host memory
*/
void delete_sparse_matrix_host(sparse_matrix * matrix) {
    if (matrix->rowIndexPtr)
        free(matrix->rowIndexPtr);
    
    if (matrix->colIndexPtr)
        free(matrix->colIndexPtr);
        
    if (matrix->valPtr)
        free(matrix->valPtr);
        
    if (matrix)
        free(matrix);  
}