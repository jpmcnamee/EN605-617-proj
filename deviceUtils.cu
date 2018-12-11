#include <stdio.h>
#include <stdlib.h>
    
#include "cusparse.h"
#include "cluster.h"    
 
/*
    The functions in this file primarily deal with allocating device memory, transfering
    memory back and forth between device and host, converting sparse matrix formats into 
    compressed row format (used by most cusparse library functions)
*/

//Wrapper function used to printout cuda error status code    
int checkCuda(cudaError_t status) {
    if (status == cudaSuccess) {
        return 0;
    } else {
        printf("Cuda error code: %d. Msg: %s\n", status, cudaGetErrorString(status));
        return 1;
    }
}
 
//Wrapper function used to printout cuSparse error status code      
int checkcuSparse(cusparseStatus_t status) {
    if (status == CUSPARSE_STATUS_SUCCESS) {
        return 0;
    } else {
        printf("cuSparse error code: %d\n", status);
        return 1;
    }
}

//Sets up cuSparse handle and matrix descripters used throughout the program    
int initDeviceContext(device_context **context) {

    /* initialize cusparse library */
    if (cusparseCreate(&((*context)->handle)) != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSparse library init failed\n");
        return 1;
    }

    /* create and setup matrix descriptor */  
    if (cusparseCreateMatDescr(&((*context)->descr)) != CUSPARSE_STATUS_SUCCESS) {
        printf("Matrix descripter init failed\n");
        return 1;
    }
    cusparseSetMatType((*context)->descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase((*context)->descr,CUSPARSE_INDEX_BASE_ZERO);
    return 0;
}

//Frees cuSparse resources    
void deleteDeviceContext(device_context *context) {
    if (context->descr)
        cusparseDestroyMatDescr(context->descr);
    
    if (context->handle)
        cusparseDestroy(context->handle);
    
    if (context)
        free(context);
}

/*
  Initializes device memory for cluster matricies. (its intended that the memory for cluster matricies is 
  re-used, so this shoudl only be called 2 times, one for current and one for future cluster setups).
*/
int initializeEmptyDeviceCluster(sparse_matrix **deviceClusters, int k, int numVectors, device_context *context) {
    sparse_matrix *hostClusters = (sparse_matrix*) malloc(sizeof(sparse_matrix)); 
    int nonZeroLength = numVectors;
    
    hostClusters->rowIndexPtr = (int*) malloc(nonZeroLength * sizeof(int));
    hostClusters->colIndexPtr = (int*) malloc(nonZeroLength * sizeof(int));
    hostClusters->valPtr = (float*) malloc(nonZeroLength * sizeof(float));
    hostClusters->valLength = nonZeroLength;
    hostClusters->numRows = numVectors;
    hostClusters->numCols = k;
    
    for (int i = 0; i < numVectors; i++) {
        hostClusters->rowIndexPtr[i] = i; 
        hostClusters->colIndexPtr[i] = 0;
        hostClusters->valPtr[i] = 1.0;
    }
    
    int success = setUpDeviceVectors(hostClusters, deviceClusters, context);
    
    delete_sparse_matrix_host(hostClusters); 
    
    return success; 
    
}

/*
    Helper function for integer comparison. Taken from here:
    https://www.tutorialspoint.com/c_standard_library/c_function_qsort.htm
*/
int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}    
    
/*
  Each centroid will be randomly assigned 1 vector. Building a sparse matrix where each column will have one element
  set to '1.0' not all rows (vectors) will be assigned to a column (eg rows will either be all 0's or have one '1.0'
  set in them.
*/
int pickRandomFutureClusters(sparse_matrix **deviceRandomClusters, int k, int numVectors, device_context *context) {
    sparse_matrix *hostRandomClusters = (sparse_matrix*) malloc(sizeof(sparse_matrix));      
    
    hostRandomClusters->rowIndexPtr = (int*) malloc(k * sizeof(int));
    hostRandomClusters->colIndexPtr = (int*) malloc(k * sizeof(int));
    hostRandomClusters->valPtr = (float*) malloc(k * sizeof(float));
    hostRandomClusters->valLength = k;
    hostRandomClusters->numRows = numVectors;
    hostRandomClusters->numCols = k;
    
    int clustersAssigned = 0;
    
    do {
        int randVector = rand() % numVectors;
        //Check to see if vector was already chosen
        for (int i = 0; i < clustersAssigned; i++) {
            if (hostRandomClusters->rowIndexPtr[i] == randVector) {
                continue; //vector already chosen
            }
        }
        
        hostRandomClusters->valPtr[clustersAssigned] = 1.0;
        hostRandomClusters->rowIndexPtr[clustersAssigned] = randVector;
        hostRandomClusters->colIndexPtr[clustersAssigned] = clustersAssigned;
        
        clustersAssigned++;
    } while (clustersAssigned < k);
    
    qsort(hostRandomClusters->rowIndexPtr, k, sizeof(int), cmpfunc); 
    
    int success = setUpDeviceVectors(hostRandomClusters, deviceRandomClusters, context);
    
    delete_sparse_matrix_host(hostRandomClusters); 
    
    return success;   
}    

/*
  Sets up device memory for the vector and centroid magnitude vectors. Also copies the host magnitude array into device memory.  
*/
int setUpDeviceMagnitudes(float *hostVectorMagnitudes, float **deviceVectorMagnitudes, int numVectors, float **deviceCentroidMagnitudes, int k){
     
    if (cudaMalloc((void**)deviceVectorMagnitudes ,numVectors*sizeof(float)) != cudaSuccess) {
        printf("ERROR allocating cuda memory\n");
        return 1;
    }
    
    if (cudaMalloc((void**)deviceCentroidMagnitudes ,k*sizeof(float)) != cudaSuccess) {
        printf("ERROR allocating cuda memory\n");
        return 1;
    }
    
    if (cudaMemcpy((*deviceVectorMagnitudes), hostVectorMagnitudes, 
                           (size_t)(numVectors*sizeof(float)), 
                           cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("ERROR copying to cuda memory\n");
        return 1;
    }
    
    return 0;
    
}

//Frees device memory allocated for magnitude vectors.    
void deleteDevicemagnitude(float *deviceMagnitudes) {
    if (deviceMagnitudes) {
        cudaFree(deviceMagnitudes);
    }
}

/*
    This function moves a sparse matrix from host memory in COO format to device memory in CSR format.
    Takes sparse matrix containing 3 host vectors in coo format who have already been initialized, 
    and sparse matrix containing 3 device pointer that have not been initialized. After successful
    completion of this function the 3 devicce pointers will point to initialized
    device memory containing the csr format of the host vectors.
    
    It is up to the caller of this function to free host and device memory.
    Returns and int status code, 0 for success 1 otherwise.
*/
int setUpDeviceVectors(sparse_matrix *hostMatrix, sparse_matrix **deviceMatrix, device_context *context) {
    unsigned int length = hostMatrix->valLength;
    unsigned int numRows = hostMatrix->numRows;
    unsigned int numCols = hostMatrix->numCols;
    int * cooRowIndex=0;
    
    if (cudaMalloc((void**)&cooRowIndex ,length*sizeof(int)) != cudaSuccess) {
        printf("ERROR allocating cuda memory\n");
        return 1;
    }
    if (cudaMalloc((void**)&((*deviceMatrix)->colIndexPtr), length*sizeof(int)) != cudaSuccess) {
        cudaFree(cooRowIndex);
        printf("ERROR allocating cuda memory\n");
        return 1;
    }
    
    if (cudaMalloc((void**)&((*deviceMatrix)->valPtr), length*sizeof(float)) != cudaSuccess) {
        cudaFree(cooRowIndex);
        printf("ERROR allocating cuda memory\n");
        return 1;
    }
    
    if (cudaMemcpy(cooRowIndex, hostMatrix->rowIndexPtr, 
                           (size_t)(length*sizeof(int)), 
                           cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(cooRowIndex);
        printf("ERROR copying to cuda memory\n");
        return 1;
    }
    if (cudaMemcpy((*deviceMatrix)->colIndexPtr, hostMatrix->colIndexPtr, 
                           (size_t)(length*sizeof(int)), 
                           cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(cooRowIndex);
        printf("ERROR copying to cuda memory\n");
        return 1;
    }
    
    if (cudaMemcpy((*deviceMatrix)->valPtr, hostMatrix->valPtr,      
                           (size_t)(length*sizeof(float)),      
                           cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(cooRowIndex);
        printf("ERROR copying to cuda memory\n");
        return 1;
    }
    
    if (cudaMalloc((void**)&((*deviceMatrix)->rowIndexPtr),(numRows+1)*sizeof(int)) != cudaSuccess) {
        cudaFree(cooRowIndex);
        printf("ERROR allocating cuda memory\n");
        return 1;
    }
    
    if (cusparseXcoo2csr(context->handle,cooRowIndex,length,numRows,
                             (*deviceMatrix)->rowIndexPtr,CUSPARSE_INDEX_BASE_ZERO) != CUSPARSE_STATUS_SUCCESS) {
        cudaFree(cooRowIndex);
        printf("ERROR converting coo to csr\n");
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    (*deviceMatrix)->valLength = length;
    (*deviceMatrix)->numRows = numRows;
    (*deviceMatrix)->numCols = numCols;
    cudaFree(cooRowIndex);
    return 0;
}    

/*
    Copies a magnitude vector from device memory to host memory. Mostly used for debugging.
    Both host and device memory should already have been allocated.
*/
int transferDeviceVector(float **hostVector, float *deviceVector, int vectorLength) {
    if (cudaMemcpy((*hostVector), deviceVector,       
                           (size_t)(vectorLength*sizeof(float)),      
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("ERROR copying magnitude from cuda memory\n");
        return 1;
    }
    
    return 0;
}
    
int checkConsistencyOfDeviceVectorF(float *deviceVector, unsigned int vectorLength) {
    int success = 0;
    float *hostVector = (float*) malloc(vectorLength * sizeof(float));
    
    if (checkCuda(cudaMemcpy(hostVector, deviceVector,       
                           (size_t)(vectorLength*sizeof(float)),      
                           cudaMemcpyDeviceToHost))) {
        printf("ERROR copying int vector from cuda memory\n");
        success = 1;
    }
    
    free(hostVector);
    return success;
}  
    
int checkConsistencyOfDeviceVectorI(int *deviceVector, unsigned int vectorLength) {
    int success = 0;
    int *hostVector = (int*) malloc(vectorLength * sizeof(int));
    
    if (checkCuda(cudaMemcpy(hostVector, deviceVector,       
                           (size_t)(vectorLength*sizeof(int)),      
                           cudaMemcpyDeviceToHost))) {
        printf("ERROR copying int vector from cuda memory\n");
        success = 1;
    }
    
    free(hostVector);
    return success;
}    

/*
    Copies a sparse matrix from device memory to host memory. Mostly used for debugging.
    Memory is allocated for the host matrix, it is up to the caller to free it.
*/
int transferDeviceMatrix(sparse_matrix **hostMatrix, sparse_matrix *deviceMatrix, int csrFormat) {
    unsigned int nonZeroLength = deviceMatrix->valLength;
    unsigned int rowLength, colLength;
    if (csrFormat != 0) {
        rowLength = nonZeroLength;
        colLength = deviceMatrix->numCols + 1;
    } else {
        rowLength = deviceMatrix->numRows + 1;
        colLength = nonZeroLength;
    }
    
    
    (*hostMatrix)->rowIndexPtr = (int*) malloc(rowLength * sizeof(int));
    (*hostMatrix)->colIndexPtr = (int*) malloc(colLength * sizeof(int));
    (*hostMatrix)->valPtr = (float*) malloc(nonZeroLength * sizeof(float));
    (*hostMatrix)->valLength = nonZeroLength;
    (*hostMatrix)->numRows = deviceMatrix->numRows;;
    (*hostMatrix)->numCols = deviceMatrix->numCols;      
    
    if (checkCuda(cudaMemcpy((*hostMatrix)->rowIndexPtr, deviceMatrix->rowIndexPtr,       
                           (size_t)(rowLength*sizeof(int)),      
                           cudaMemcpyDeviceToHost))) {
        printf("ERROR copying rows from cuda memory. Allocated %lu bytes to copy vector of size %u\n", (rowLength*sizeof(int)), rowLength );
        return 1;
    }
    
    if (checkCuda(cudaMemcpy((*hostMatrix)->colIndexPtr, deviceMatrix->colIndexPtr,       
                           (size_t)(colLength*sizeof(float)),      
                           cudaMemcpyDeviceToHost))) {
        printf("ERROR copying cols from cuda memory\n");
        return 1;
    }
    
    if (cudaMemcpy((*hostMatrix)->valPtr, deviceMatrix->valPtr,       
                           (size_t)(nonZeroLength*sizeof(float)),      
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("ERROR copying vals from cuda memory\n");
        return 1;
    }
    
    return 0;
}

/*
    Deletes a sparse matrix whose arrays point to device memory.
*/
void delete_sparse_matrix_device(sparse_matrix * matrix) {
    if (matrix->rowIndexPtr)
        cudaFree(matrix->rowIndexPtr);
    
    if (matrix->colIndexPtr)
        cudaFree(matrix->colIndexPtr);
        
    if (matrix->valPtr)
        cudaFree(matrix->valPtr);
        
    if (matrix)
        free(matrix); 
}
    
void clear_sparse_matrix_device(sparse_matrix * matrix) {
    if (matrix->rowIndexPtr){
        cudaFree(matrix->rowIndexPtr);
    }
    matrix->rowIndexPtr = 0;
    
    if (matrix->colIndexPtr) {
        cudaFree(matrix->colIndexPtr);
    }
    
    matrix->colIndexPtr = 0;
    
    if (matrix->valPtr) {
        cudaFree(matrix->valPtr);
    }
    
    matrix->valPtr = 0;
    matrix->valLength = 0;
    matrix->numRows = 0;
    matrix->numCols = 0;
        
    
}    
    