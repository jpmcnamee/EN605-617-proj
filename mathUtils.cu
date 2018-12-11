#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/host_vector.h>    
#include <thrust/reduce.h>
    
#include "cusparse.h"
#include "cluster.h"

/*
    The functions in this file deal with the mathmatical operations that are performed on 
    matricies and vectors in device memory. Uses a combination of cuSparse, CUDA Thrust and 
    custom kernels to achive its purpose.
*/
    

/*
    Uses cuSparse to multiply 2 sparse matricies (A and B) and store the result in (C) all matricies should live
    in device memory.
    
    This method will uses cuSparse library to compute the memroy dimensions of C
    
    Function is based off of example code here:
    https://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csrgemm
*/
int multiplySparseMatricies(sparse_matrix *device_a, sparse_matrix *device_b, sparse_matrix **device_c, device_context *context) {
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;
    unsigned int m = device_a->numRows;
    unsigned int n = device_b->numCols;
    unsigned int k = device_a->numCols;
    
    cusparseSetPointerMode(context->handle, CUSPARSE_POINTER_MODE_HOST);
    
    //First need to set up memory for the result C matrix (and compute how much mem it needs)
    if(checkCuda(cudaMalloc((void**)&((*device_c)->rowIndexPtr), sizeof(int)*(m+1)))) {
        printf("ERROR allocating C row cuda memory\n");
        return 1;
    }

    int baseC, nnzC;
    int *nnzTotalDevHostPtr = &nnzC;
    
    if ( checkcuSparse(cusparseXcsrgemmNnz(context->handle, op, op, m, n, k, 
            context->descr, device_a->valLength, device_a->rowIndexPtr, device_a->colIndexPtr,
            context->descr, device_b->valLength, device_b->rowIndexPtr, device_b->colIndexPtr,
            context->descr, (*device_c)->rowIndexPtr, nnzTotalDevHostPtr))) {
        printf("ERROR calculating dimentions of new C matrix\n");
        
        printf("Checking consistency of device sparse matrix A\n");
        checkConsistencyOfDeviceMatrix(device_a);
    
        printf("Checking consistency of device sparse matrix B\n");
        checkConsistencyOfDeviceMatrix(device_b);
        return 1;
    }
    
    if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(&nnzC, (*device_c)->rowIndexPtr+m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, (*device_c)->rowIndexPtr, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }
    
    
    if(checkCuda(cudaMalloc((void**)&((*device_c)->colIndexPtr), sizeof(int) * nnzC))) {
        printf("ERROR allocating C col cuda memory (%d) bytes.\n", nnzC);
        return 1;
    }
    
    if(checkCuda(cudaMalloc((void**)&((*device_c)->valPtr), sizeof(float) * nnzC))) {
        printf("ERROR allocating C values cuda memory(%d)\n", nnzC);
        return 1;
    }
    
    //perform the multiplication
    if (checkcuSparse(cusparseScsrgemm(context->handle, op, op, m, n, k,
        context->descr, device_a->valLength,
        device_a->valPtr, device_a->rowIndexPtr, device_a->colIndexPtr,
        context->descr, device_b->valLength,
        device_b->valPtr, device_b->rowIndexPtr, device_b->colIndexPtr,
        context->descr,
        (*device_c)->valPtr, (*device_c)->rowIndexPtr, (*device_c)->colIndexPtr))) {
    
        printf("ERROR performing sparse matrix multiplication\n");
        return 1;
    }
    cudaDeviceSynchronize();
    (*device_c)->numRows = m;
    (*device_c)->numCols = n;
    (*device_c)->valLength = nnzC;
    
    return 0;
}

/*
  Computes centroid vectors (Vectors x Clusters = Centroids)  
*/
int computeCentriods(sparse_matrix *vectors, sparse_matrix *clusters, sparse_matrix **centroids, device_context *context) {
    return multiplySparseMatricies(vectors, clusters, centroids, context);
}

/*
    When a new centroid matrix is computed the lengths of each centroid vector need to be udpated. This function
    takes a centroid matrix and uses a custom kernel to update the centroid magnitude vector in device memroy.
*/
int updateCentroidMagnitudes(sparse_matrix *centroids, float *deviceCentroidMagnitudes, device_context *context) {
    
    //Get centroid transpose matrix
    sparse_matrix *deviceCentroidsTranspose = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    
    if (computeTranspose(centroids, &deviceCentroidsTranspose, context) != 0) {
        delete_sparse_matrix_device(deviceCentroidsTranspose);
        printf("ERROR: updating centroid magnitudes\n");
        return 1;
    }
    
    //Figure out how many threads to run with kernel
    int numCentroids = deviceCentroidsTranspose->numRows;
    
    int blockSize = 64;
    if (numCentroids > blockSize) {
        blockSize = 512;
    }
    int numBlocks = numCentroids/blockSize;
    if (numCentroids % blockSize != 0) {
		++numBlocks;
    }
    
    //Kernel that computes the length of each centroid vector
    kernel_compute_centroid_lengths<<<numBlocks, blockSize>>>(
                                    deviceCentroidsTranspose->rowIndexPtr, deviceCentroidsTranspose->valPtr, 
                                    deviceCentroidMagnitudes, numCentroids);
    
    cudaDeviceSynchronize();
    return 0;
    
}
    
/*
  Creates the transpose of a sparse matrix. Given matrix A and pointer to B, allocate the memory for B and store
  the transpose of A. (this is computed by converted matrix A in CSR format to CSC format and then switching the row 
   column pointers.)
*/
int computeTranspose(sparse_matrix *deviceMatrix, sparse_matrix **deviceMatrixTranspose, device_context *context) {
    unsigned int m = deviceMatrix->numRows;
    unsigned int n = deviceMatrix->numCols;
    unsigned int nnz = deviceMatrix->valLength;
    
    if(cudaMalloc((void**)&((*deviceMatrixTranspose)->rowIndexPtr), sizeof(int)*(nnz)) != cudaSuccess) {
        printf("ERROR allocating centroids row cuda memory\n");
        return 1;
    }
    if(cudaMalloc((void**)&((*deviceMatrixTranspose)->colIndexPtr), sizeof(int) * (n+1)) != cudaSuccess) {
        printf("ERROR allocating centroids col cuda memory\n");
        return 1;
    }
    
    if(cudaMalloc((void**)&((*deviceMatrixTranspose)->valPtr), sizeof(float) * nnz) != cudaSuccess) {
        printf("ERROR allocating centroid values cuda memory\n");
        return 1;
    }
    
    if (cusparseScsr2csc(context->handle, m, n, nnz,
                 deviceMatrix->valPtr, deviceMatrix->rowIndexPtr, deviceMatrix->colIndexPtr, 
                 (*deviceMatrixTranspose)->valPtr, (*deviceMatrixTranspose)->rowIndexPtr,(*deviceMatrixTranspose)->colIndexPtr, 
                 CUSPARSE_ACTION_NUMERIC, 
                 CUSPARSE_INDEX_BASE_ZERO) != CUSPARSE_STATUS_SUCCESS) {
        printf("ERROR matrix transpose\n");
        return 1;
    }
    
    cudaDeviceSynchronize();
    
    //Swap row and col indicies to put back into CSR format which is not the transpose of the original
    int **rowAddr = &((*deviceMatrixTranspose)->rowIndexPtr);
    int **colAddr = &((*deviceMatrixTranspose)->colIndexPtr);
    int *temp = *rowAddr;
    *rowAddr = *colAddr;
    *colAddr = temp;
    
    (*deviceMatrixTranspose)->numRows = n;
    (*deviceMatrixTranspose)->numCols = m;
    (*deviceMatrixTranspose)->valLength = nnz;
 
    return 0;
}

/*
    Multiplies Vector X Centroid matricies, uses a custom kernel on the result to normalize
    the distances and find the closest centroids to each vector. 
    The column memory of future clusters will be overwritten with valid positions.
*/
int findFutureClusters(sparse_matrix *vectors, float * deviceVectorMagnitudes, 
                        sparse_matrix *centroids, float * deviceCentroidMagnitudes,
                        sparse_matrix *futureClusters, device_context *context) {

    //allocate result matrix
    sparse_matrix *distanceMatrix = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    
    //perform multiplication
    if(multiplySparseMatricies(vectors, centroids, &distanceMatrix, context) != 0) {
        delete_sparse_matrix_device(distanceMatrix);
        printf("Error finding future clusters\n");
        return 1;
    }
    
    //Figure out how many threads to run with kernel
    int numVectors = distanceMatrix->numRows;
    int numCentroids = distanceMatrix->numCols;
    
    int blockSize = 64;
    if (numVectors > blockSize) {
        blockSize = 512;
    }
    int numBlocks = numVectors/blockSize;
    if (numVectors % blockSize != 0) {
		++numBlocks;
    }
    
    //run kernel to normalize and find max store in future clusters
    kernel_normalize_and_find_closest<<<numBlocks, blockSize>>>(
                                       distanceMatrix->rowIndexPtr, distanceMatrix->colIndexPtr, distanceMatrix->valPtr, 
                                       deviceVectorMagnitudes, numVectors, 
                                       deviceCentroidMagnitudes, numCentroids, futureClusters->colIndexPtr);
    
    cudaDeviceSynchronize();
    
    delete_sparse_matrix_device(distanceMatrix);
    return 0;    
}

/*
  Alters the column state of pastClusters, should not be re-used for calculations, 
    though the allocated memory can be reset and then used for later calculations.  
*/
int findNumDiffs(int *numDiffs, sparse_matrix *pastClusters, sparse_matrix *futureClusters) {
    //Figure out how many threads to run with kernel
    int length = futureClusters->valLength;
    
    int blockSize = 64;
    if (length > blockSize) {
        blockSize = 512;
    }
    int numBlocks = length/blockSize;
    if (length % blockSize != 0) {
		++numBlocks;
    }
    
    //kernel_subtract_abs<<<numBlocks, blockSize>>>(pastClusters->colIndexPtr, futureClusters->colIndexPtr, length);
    kernel_diff<<<numBlocks, blockSize>>>(pastClusters->colIndexPtr, futureClusters->colIndexPtr, length);
    cudaDeviceSynchronize();
    
    thrust::device_ptr<int> devVectorDiffs(pastClusters->colIndexPtr);
    int sum = thrust::reduce(devVectorDiffs, devVectorDiffs + length, (int) 0, thrust::plus<int>());
    *numDiffs = sum;
    
    return 0;    
}

/*
    Once the future clusters have been found, the future cluster matrix will be used to compute
    the next iteration of the centroid matrix. (Vector x Clusters = centroids)
    Before we can do that we need to ensure all the columns of the Cluster matrix are unit vectors.
    
    Here we use Thrust to sort and reduce to count up how many vectors have been assigned to each cluster
    Then a custome kernel is used to change the value of each centroid vector to be 1/num vectors in cluster.
*/
int averageClusters(sparse_matrix *futureClusters, int k) {
    int length = futureClusters->valLength;
    //Use thrust to reduce by key
    thrust::device_ptr<int> device_clusterA(futureClusters->colIndexPtr);
    thrust::device_ptr<float> device_clusterB(futureClusters->valPtr);
    
    thrust::device_vector<int> dev_A(device_clusterA, device_clusterA + length);
    thrust::sort(dev_A.begin(), dev_A.end());
    
    thrust::device_vector<float> dev_B(device_clusterB, device_clusterB + length);
    thrust::device_vector<int> dev_C(k);
    thrust::device_vector<float> dev_D(k);
    
    thrust::equal_to<int> binary_pred;
    thrust::plus<float> binary_op;
    
    try {
        thrust::reduce_by_key(dev_A.begin(), dev_A.end(), dev_B.begin(), dev_C.begin(), dev_D.begin(), binary_pred, binary_op);
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
        return 1;
    }
    thrust::device_ptr<float> devPtrD = dev_D.data();
    float *centoidsCounts = thrust::raw_pointer_cast(devPtrD);
    
    
    //Figure out how many threads to run with kernel
    int numVectors = futureClusters->valLength;
    
    int blockSize = 64;
    if (numVectors > blockSize) {
        blockSize = 512;
    }
    int numBlocks = numVectors/blockSize;
    if (numVectors % blockSize != 0) {
		++numBlocks;
    }
    
    //Call kernel to perform division on future cluster vals
    kernel_average_clusters<<<numBlocks, blockSize>>>(centoidsCounts, numVectors, futureClusters->colIndexPtr, futureClusters->valPtr);
    
    cudaDeviceSynchronize();
    
    /*
    thrust::host_vector<int> hostC = dev_C;
    thrust::host_vector<float> hostD = dev_D;
    
    printf("Reduced keys\n");
    for (int i = 0; i < hostC.size(); i++) {
        printf(" %d ", hostC[i]);                                 
    }
    printf("\nReduced values\n");                                 
    for (int i = 0; i < hostD.size(); i++) {
        printf(" %f ", hostD[i]);                                 
    }  
    
    */
    
    return 0;
}
    
