#include<stdio.h>
#include <stdlib.h>

#include "cluster.h"

/*
    The functions in this file are used for testing/debugging purposes and will not be a part of the final 
    program.
*/


 /* create the following sparse test matrix in COO format */
    /* |1.0     2.0 3.0|
       |    4.0        |
       |5.0     6.0 7.0|
       |    8.0     9.0| */

int loadTestData(sparse_matrix ** matrix, float ** magnitudes) {
    int numVectors = 4;
    int nonZeroLength = 9;

    *magnitudes = (float*) malloc(numVectors * sizeof(float));
    (*matrix)->rowIndexPtr = (int*) malloc(nonZeroLength * sizeof(int));
    (*matrix)->colIndexPtr = (int*) malloc(nonZeroLength * sizeof(int));
    (*matrix)->valPtr = (float*) malloc(nonZeroLength * sizeof(float));

    if (*magnitudes == NULL || (*matrix)->rowIndexPtr == NULL || (*matrix)->colIndexPtr == NULL || (*matrix)->valPtr == NULL){
        printf("host malloc failed\n");
        return 1; 
    }

    (*matrix)->valLength = nonZeroLength;
    (*matrix)->numRows = numVectors;
    (*matrix)->numCols = 4;

    (*matrix)->rowIndexPtr[0]=0; (*matrix)->colIndexPtr[0]=0; (*matrix)->valPtr[0]=1.0;  
    (*matrix)->rowIndexPtr[1]=0; (*matrix)->colIndexPtr[1]=2; (*matrix)->valPtr[1]=2.0;  
    (*matrix)->rowIndexPtr[2]=0; (*matrix)->colIndexPtr[2]=3; (*matrix)->valPtr[2]=3.0;  
    (*matrix)->rowIndexPtr[3]=1; (*matrix)->colIndexPtr[3]=1; (*matrix)->valPtr[3]=4.0;  
    (*matrix)->rowIndexPtr[4]=2; (*matrix)->colIndexPtr[4]=0; (*matrix)->valPtr[4]=5.0;  
    (*matrix)->rowIndexPtr[5]=2; (*matrix)->colIndexPtr[5]=2; (*matrix)->valPtr[5]=6.0;
    (*matrix)->rowIndexPtr[6]=2; (*matrix)->colIndexPtr[6]=3; (*matrix)->valPtr[6]=7.0;  
    (*matrix)->rowIndexPtr[7]=3; (*matrix)->colIndexPtr[7]=1; (*matrix)->valPtr[7]=8.0;  
    (*matrix)->rowIndexPtr[8]=3; (*matrix)->colIndexPtr[8]=3; (*matrix)->valPtr[8]=9.0;
    
    (*magnitudes)[0] = 3.74; 
    (*magnitudes)[1] = 4;
    (*magnitudes)[2] = 10.48;
    (*magnitudes)[3] = 12.04;
    printf("host memory allocated successfully\n");
    return 0;
}

int pickTestFutureClusters(sparse_matrix **futureClusters, device_context *context) {
    sparse_matrix *hostFutureClusters = (sparse_matrix*) malloc(sizeof(sparse_matrix)); 
    int nonZeroLength = 2;
    
    hostFutureClusters->rowIndexPtr = (int*) malloc(nonZeroLength * sizeof(int));
    hostFutureClusters->colIndexPtr = (int*) malloc(nonZeroLength * sizeof(int));
    hostFutureClusters->valPtr = (float*) malloc(nonZeroLength * sizeof(float));
    hostFutureClusters->valLength = nonZeroLength;
    hostFutureClusters->numRows = 4;
    hostFutureClusters->numCols = 2;
    
    hostFutureClusters->valPtr[0]=1.0; hostFutureClusters->rowIndexPtr[0]=1; hostFutureClusters->colIndexPtr[0]=1;
    hostFutureClusters->valPtr[1]=1.0; hostFutureClusters->rowIndexPtr[1]=3; hostFutureClusters->colIndexPtr[1]=0;
    
    int success = setUpDeviceVectors(hostFutureClusters, futureClusters, context);
    
    delete_sparse_matrix_host(hostFutureClusters); 
    
    return success;
    
}

void printMagnitudesDevice(float *deviceMagnitudes, int magnitudeLength) {
    float *hostMagnitudes = (float*) malloc(magnitudeLength * sizeof(float));
    
    if (transferDeviceVector(&hostMagnitudes, deviceMagnitudes, magnitudeLength) != 0) {
        printf("ERROR copying magnitude from cuda memory\n");
        free(hostMagnitudes);
        return;
    }
    
    printf("\n");
    for (int i = 0; i < magnitudeLength; i++) {
        printf(" %f ", hostMagnitudes[i]);    
    }
    printf("\n");
    
    free(hostMagnitudes);
}

void printSparseMatrixDevice(sparse_matrix *deviceMatrix, int details, int cscFormat) {
    sparse_matrix *hostMatrix = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    
    if (transferDeviceMatrix(&hostMatrix, deviceMatrix, cscFormat) != 0) {
        printf("Error transfering device back to host. Cannot print\n");
        delete_sparse_matrix_host(hostMatrix);
        return;
    }
    int rowLength = hostMatrix->numRows +1;
    int colLength = hostMatrix->valLength;
    
    if (cscFormat == 0) {
        for (int i =0; i < hostMatrix->numRows; i++) {
            int rowOffset = hostMatrix->rowIndexPtr[i];

            for (int j = 0; j < hostMatrix->numCols; j++) {
                float printVal = 0.0;

                if (rowOffset == hostMatrix->rowIndexPtr[i+1]) {
                     printVal = 0.0;    
                } else if (j == hostMatrix->colIndexPtr[rowOffset]) {
                    printVal = hostMatrix->valPtr[rowOffset];
                    rowOffset++;
                }


                printf(" %f ", printVal);    
            }
            printf("\n");
        }
    } else {
         printf("CSC format not available for pretty printing");
         rowLength = hostMatrix->valLength;
         colLength = hostMatrix->numCols +1;
    }
    
    if (details != 0) {
        printf("\nVals = ");
        for (int i = 0; i < hostMatrix->valLength; i++) {
            printf(" %f ", hostMatrix->valPtr[i]);
        }
        printf("\nrows = ");
        for (int i = 0; i < (rowLength); i++) {
            printf(" %d ", hostMatrix->rowIndexPtr[i]);
        }  
        printf("\ncols = ");
        for (int i = 0; i < colLength; i++) {
            printf(" %d ", hostMatrix->colIndexPtr[i]);
        }     
        printf("\n"); 
        printf("Rows %d, Cols %d, Total nonZero %d\n", hostMatrix->numRows, hostMatrix->numCols, hostMatrix->valLength);
    }
    
    delete_sparse_matrix_host(hostMatrix);
    return;
    
}

int checkConsistencyOfDeviceMatrix(sparse_matrix *deviceMatrix) {
    int status = 0;
    sparse_matrix *hostMatrix = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    
    if (transferDeviceMatrix(&hostMatrix, deviceMatrix, 0) != 0) {
        printf("Error transfering device back to host. Cannot print\n");
        delete_sparse_matrix_host(hostMatrix);
        return 1;
    }
    
    /*
    float sum = 0;
    for (int i = 0; i < hostMatrix->valLength; i++) {
        sum+= hostMatrix->valPtr[i];
        if (hostMatrix->colIndexPtr[i] != i) {
            printf("Unexpected column %d instead of %d\n", hostMatrix->colIndexPtr[i], i);
            break;
        }
    }
    printf("total nnz summed up to %f\n", sum);
    */
    int prev = 0;
    int colsChecked = 0;
    int stop = 0;
    for (int i = 0; i < hostMatrix->numRows; i++) {
        if (hostMatrix->rowIndexPtr[i] < prev) {
            printf("Unexpected row %d instead of %d\n", hostMatrix->rowIndexPtr[i], prev);
            status = 1;
            break;
        }
        if (hostMatrix->rowIndexPtr[i] != hostMatrix->rowIndexPtr[i + 1]) {
            int prevCol = 0;
            for (int j = hostMatrix->rowIndexPtr[i]; j < hostMatrix->rowIndexPtr[i + 1]; j++) {
                colsChecked++;
                if (hostMatrix->colIndexPtr[j] < prevCol) {
                    printf("Unexpected col val should be in increasing order %d !< %d. At row %d\n", prevCol, hostMatrix->colIndexPtr[j], i);
                    stop = 1;
                    status = 1;
                    break;
                }
                prevCol = hostMatrix->colIndexPtr[j];
            }
        }
        if (stop == 1) {
            break;    
        }
        prev = hostMatrix->rowIndexPtr[i];
    }
    printf("Columns checked: %d\n", colsChecked);
    printf("Rows %d, Cols %d, Total nonZero %d\n", hostMatrix->numRows, hostMatrix->numCols, hostMatrix->valLength);
    printf("Rows setup correctly %d = %d ?\n", hostMatrix->valLength, hostMatrix->rowIndexPtr[hostMatrix->numRows]);
    
    if (hostMatrix->valLength != hostMatrix->rowIndexPtr[hostMatrix->numRows]) {
        status = 1;
    }
    
    delete_sparse_matrix_host(hostMatrix);
    return status;
}