#include <math.h>
  
/*
  This file contains the custome kernels used by the program  
*/
    
//Helper function for device used to compute global thread ID    
__device__
unsigned int getThreadID() {
    const unsigned int threadsPerBlock  = blockDim.x * blockDim.y;
    const unsigned int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    const unsigned int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;
    const unsigned int thread_idx = blockNumInGrid * threadsPerBlock + threadNumInBlock;
    return thread_idx;
}

/*
  Intended to be used on the result of (Vector Matrix) X (Centroid Matrix)
  Each thread iterates over all the distances between a single vector and the other centroids.
  This will normalize the distance, and keep track of the centroid whose distance is maximized (most similar)
  The result will be stored in the result vector, which will hold the closest centroid for each vector.
*/
__global__
void kernel_normalize_and_find_closest(int *rowIndex, int *colIndex, float *valIndex, 
                                       float *vectorMagnitude, int numVectors, 
                                       float *centroidMagnitude, int numCentroids, int *result) {

    unsigned int thread_idx = getThreadID();

    if (thread_idx < numVectors) {
                               
        float vectorLength = vectorMagnitude[thread_idx]; 
        float centroidLength;                       
        int rowStartOffset = rowIndex[thread_idx];
        int rowEndOffset = rowIndex[thread_idx + 1];
                               
        int closestCentroid = 0;
        int currentCentroid = 0;                       
        float maxDistance = -1000;
        float normalizedDistance;
        float distance;                       
                               
        for (int i = rowStartOffset; i < rowEndOffset; i++) {
            currentCentroid = colIndex[i];
            centroidLength = centroidMagnitude[currentCentroid];
            distance = valIndex[i];
            normalizedDistance = distance / (vectorLength * centroidLength);

            if (normalizedDistance > maxDistance) {
                maxDistance = normalizedDistance;
                closestCentroid = currentCentroid;
            }
        }
                               
        result[thread_idx] = closestCentroid;                      
    }
}

/*
    Computes the magnitude for each centroid vector, Each thread (instance of this kernel)
    will loop over all the values in the centroid vector and compute the length.
*/
__global__
void kernel_compute_centroid_lengths(int *rowIndex, float *valIndex, 
                                       float *centroidMagnitude, int numCentroids) {

    unsigned int thread_idx = getThreadID();

    if (thread_idx < numCentroids) {
                       
        int rowStartOffset = rowIndex[thread_idx];
        int rowEndOffset = rowIndex[thread_idx + 1];
                               
        float length = 0;                   
        float val;                        
        for (int i = rowStartOffset; i < rowEndOffset; i++) {
            val = valIndex[i];
            length += (val * val);
        }
                               
        centroidMagnitude[thread_idx] = sqrt(length);                      
    }
}  

/*
    Simple kernel that is used to subract 2 vectors and taking the absolute value
    of the result for each element-wise subtraction. This is used as part of finding the 
    differences between the column assignments of 2 cluster matricies.
    
    This kernel doesn't require extra memory to store the result rather the results are 
    stored in the first argument arry.
*/
__global__
void kernel_subtract_abs(int *colIndexOld, int *colIndexNew, int length) {    
    unsigned int thread_idx = getThreadID();
    
    if (thread_idx < length) {
        int result = colIndexOld[thread_idx] - colIndexNew[thread_idx];
        colIndexOld[thread_idx] = abs(result);                    
    }                        
    
}
                            
__global__
void kernel_diff(int *colIndexOld, int *colIndexNew, int length) {    
    unsigned int thread_idx = getThreadID();
    
    if (thread_idx < length) {
        if (colIndexOld[thread_idx] == colIndexNew[thread_idx]) {
            colIndexOld[thread_idx] = 0;
        } else {
            colIndexOld[thread_idx] = 1;
        }
                            
    }                        
    
}                            

/*
    Used to average a centroid vector. A centroid is the averaged sum of all the vectors in its cluster.
    Loading up the cluster matrix with values (1/numVectors in a cluster) is done first before computing the
    centroid verticies.
*/
__global__
void kernel_average_clusters(float *centroidTotals, int numVectors, int *colIndex, float *valIndex) {    
    unsigned int thread_idx = getThreadID();
    
    if (thread_idx < numVectors) {
        int centroid = colIndex[thread_idx];
        
        float centroidCount = centroidTotals[centroid];
    
        valIndex[thread_idx] = 1.0/centroidCount;                   
    }                        
    
}                            
    