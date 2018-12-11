#include<stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cluster.h"

/*
    File contains the main function and cluster implimentation.
    Will initialize all variables and start the clustering algorithm.
    
*/

//Frees up host memory
void cleanUpHostResources(sparse_matrix * hostVectorMatrix,
                         float * hostVectorMagnitudes) {
    
    printf("Cleaning up host matrix\n");
    delete_sparse_matrix_host(hostVectorMatrix);
    
    printf("Cleaning up host magnitudes\n");
    if (hostVectorMagnitudes) {
        free(hostVectorMagnitudes);
    }
    
}

//Frees up device memory that has been allocated
void cleanUpDeviceResources(sparse_matrix * deviceVectorMatrix,
                            sparse_matrix * deviceFutureClusters,
                            sparse_matrix * deviceCurrentClusters,
                            sparse_matrix *deviceCentroids,
                            sparse_matrix *deviceVectorMatrixTranspose,
                            float *deviceVectorMagnitudes,
                            float *deviceCentroidMagnitudes,
                            device_context * context) {
    
    printf("Cleaning up device resources\n");
    delete_sparse_matrix_device(deviceVectorMatrix);
    delete_sparse_matrix_device(deviceFutureClusters);
    delete_sparse_matrix_device(deviceCurrentClusters);
    delete_sparse_matrix_device(deviceCentroids);
    delete_sparse_matrix_device(deviceVectorMatrixTranspose);
    deleteDeviceContext(context);
    deleteDevicemagnitude(deviceVectorMagnitudes);
    deleteDevicemagnitude(deviceCentroidMagnitudes);
    
}

//Helper function used to swap the pointers of 2 matrix data structs
void swapMatrixPointers(sparse_matrix** a, sparse_matrix** b){
    sparse_matrix* temp = *a;
    *a = *b;
    *b = temp;
}

/*
    Takes a bunch of device pointers that need to be setup, allocates the device memory and 
    populates it appropriatly. Any host memory declaired here will be freed by the time this function
    terminates. The caller is responsible for freeing device memory.
    
    Can optionaly be setup using test data or real data from a configurable data file.
*/
int initialize(sparse_matrix **deviceVectorMatrix, 
                  sparse_matrix **deviceVectorMatrixTranspose, 
                  float **deviceVectorMagnitudes, 
                  sparse_matrix **deviceCentroids, 
                  float **deviceCentroidMagnitudes, 
                  sparse_matrix **deviceCurrentClusters, 
                  sparse_matrix **deviceFutureClusters, 
                  device_context **context, int k, char *dataFile, int maxVectors) {
    
    sparse_matrix *hostVectorMatrix = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    float *hostVectorMagnitudes = 0;
    
    
    if (dataFile == NULL) {
        printf("loading test data to host memory\n");
        if (loadTestData(&hostVectorMatrix, &hostVectorMagnitudes) != 0) {
            cleanUpHostResources(hostVectorMatrix, hostVectorMagnitudes);
            return 1;
        }
    } else {
        printf("loading real data from file %s to host memory\n", dataFile);
        if (loadData(dataFile, &hostVectorMatrix, &hostVectorMagnitudes, maxVectors) != 0) {
            cleanUpHostResources(hostVectorMatrix, hostVectorMagnitudes);
            return 1;
        }
    }
    
    printf("loading vector magnitudes into device memory\n");
    if (setUpDeviceMagnitudes(hostVectorMagnitudes, deviceVectorMagnitudes, hostVectorMatrix->numRows, deviceCentroidMagnitudes,k) != 0) {
        cleanUpHostResources(hostVectorMatrix, hostVectorMagnitudes);
        return 1;
    }
    
    printf("initializing cusparse resources\n");
    if (initDeviceContext(context) != 0) {
        cleanUpHostResources(hostVectorMatrix, hostVectorMagnitudes);
        return 1;  
    }
        
    printf("loading vector sparse matrix into device memory\n");
    if (setUpDeviceVectors(hostVectorMatrix, deviceVectorMatrix, *context) != 0) {
        cleanUpHostResources(hostVectorMatrix, hostVectorMagnitudes);
        return 1; 
    }
    
    //dont need the host vectors anymore.
    cleanUpHostResources(hostVectorMatrix, hostVectorMagnitudes);

    if (computeTranspose(*deviceVectorMatrix, deviceVectorMatrixTranspose, *context) != 0) {
        return 1; 
    }
    
    if (initializeEmptyDeviceCluster(deviceCurrentClusters, k, (*deviceVectorMatrix)->numRows, *context) != 0) {
        return 1; 
    }
    
    if (initializeEmptyDeviceCluster(deviceFutureClusters, k, (*deviceVectorMatrix)->numRows, *context) != 0) {
        return 1; 
    }
    
    sparse_matrix *deviceInitialClusters = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    if (pickRandomFutureClusters(&deviceInitialClusters, k, (*deviceVectorMatrix)->numRows, *context) != 0) {
        printf("Error picking intial random clusters\n");
        delete_sparse_matrix_device(deviceInitialClusters);
        return 1; 
    }
    
    
    if (computeCentriods(*deviceVectorMatrixTranspose, deviceInitialClusters, 
                         deviceCentroids, *context) != 0) {
        printf("Error computing initla centroids\n");
        return 1;    
    }
    
    //finished with initial centroid calculation
    delete_sparse_matrix_device(deviceInitialClusters);
    
    if (updateCentroidMagnitudes(*deviceCentroids, *deviceCentroidMagnitudes, *context) != 0) {
        return 1;
    }
    
    return 0;
}

/*
    Primary clustering algorithm. 
        Start with random centroids
        For each vector find closest centroid
        create new clusters based on this grouping of vectors to their closest centroids
        See how many vectors change cluster since the last iteration
        Calculate new centroids based on the new set of vectors in its group
        repeate until number of vector swaps stop (convergence) or after a fixed number of cycles
        
    the caller of this function is responseible for setting up all initial memroy and freeing all memory.
*/
int kMeansCluster(sparse_matrix *deviceVectorMatrix, 
                  sparse_matrix *deviceVectorMatrixTranspose, 
                  float *deviceVectorMagnitudes, 
                  sparse_matrix *deviceCentroids, 
                  float *deviceCentroidMagnitudes, 
                  sparse_matrix *deviceCurrentClusters, 
                  sparse_matrix *deviceFutureClusters, 
                  device_context *context, int k, int maxTrials) {
    int numDiffs = 1;
    int numTrials = 0;
    clock_t start, stop;
    double elapsed = 0, totalElapsed = 0;
    
    printf("Beggining clustering algorithm\n");
    
    do {
        start = clock();
        //deviceFutureClusters col memory will be filled in with correct cluster assignments
        if (findFutureClusters(deviceVectorMatrix, deviceVectorMagnitudes, 
                        deviceCentroids, deviceCentroidMagnitudes,
                        deviceFutureClusters, context) != 0) {
            printf("Error finding future clusters. Trial: %d\n", numTrials);
            return 1;  
        }    
        
        
        //deviceCurrentClusters col memory will be overwritten here
        if (findNumDiffs(&numDiffs, deviceCurrentClusters, deviceFutureClusters) != 0) {
            printf("Error finding num diffs. Trial: %d\n", numTrials);
            return 1;  
        }
        //swap pointers current=future. calculations with deviceCurrentClusters is valid from here on
        swapMatrixPointers(&deviceCurrentClusters, &deviceFutureClusters);
        if (averageClusters(deviceCurrentClusters, k) != 0) {
            printf("Error averaging current cluster. Trial: %d\n", numTrials);  
            return 1;  
        }
        
        clear_sparse_matrix_device(deviceCentroids);
        if (computeCentriods(deviceVectorMatrixTranspose, deviceCurrentClusters, 
                         &deviceCentroids, context) != 0) {
            printf("Error computing next centroids. Trial: %d\n", numTrials);
            return 1;    
        }
        
        if (updateCentroidMagnitudes(deviceCentroids, deviceCentroidMagnitudes, context) != 0) {
            printf("Error updating centroid magnitudes. Trial: %d\n", numTrials);
            return 1;
        }
        numTrials++;
        
        stop = clock();
        elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
        totalElapsed += elapsed;
        printf("Completed trial %d in %f ms. Num Diffs=%d\n", numTrials, elapsed, numDiffs);
    } while (numDiffs > 0 && numTrials < maxTrials);
    
    printf("Total algorithm runtime: %f ms. %f ms/trial\n", totalElapsed, totalElapsed / numTrials);
    
    return 0;
}

/*
    Main function. Takes optional argumetns:
        k = number of clusters
        dataFile = location on disk of a data file of vectors to use for clustering.
        maxVectors = the max number of vectors to read from the data file (-1 for inf)
*/
int main(int argc, char *argv[]) {
    int k = 2;
    int maxVectors = -1;
    char *dataFile = NULL;
    int maxTrials = 10;
    
    //start timing
    clock_t start = clock();
    
    if (argc >= 2) {
		k = atoi(argv[1]);
	}
    
    if (argc >= 3) {
		dataFile = argv[2];
	}
    
    if (argc >= 4) {
		maxVectors = atoi(argv[3]);
	}
    
    if (argc >= 5) {
		maxTrials = atoi(argv[4]);
	}

    float *deviceVectorMagnitudes = 0;
    float *deviceCentroidMagnitudes = 0;
    
    device_context *context = (device_context*) malloc(sizeof(device_context));
    sparse_matrix *deviceVectorMatrix = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    sparse_matrix *deviceCurrentClusters = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    sparse_matrix *deviceFutureClusters = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    sparse_matrix *deviceCentroids = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    sparse_matrix *deviceVectorMatrixTranspose = (sparse_matrix*) malloc(sizeof(sparse_matrix));
    
    //Setup all the variables
    if (initialize(&deviceVectorMatrix, 
                  &deviceVectorMatrixTranspose, 
                  &deviceVectorMagnitudes, 
                  &deviceCentroids, 
                  &deviceCentroidMagnitudes, 
                  &deviceCurrentClusters, 
                  &deviceFutureClusters, 
                  &context, k, dataFile, maxVectors) != 0) {
        printf("Error initializing device memory\n");
        cleanUpDeviceResources(deviceVectorMatrix, deviceFutureClusters, deviceCurrentClusters, 
                               deviceCentroids, deviceVectorMatrixTranspose, deviceVectorMagnitudes, 
                               deviceCentroidMagnitudes, context);
        return 1;          
    }
    
    
    //Run the algorithm
    if (kMeansCluster(deviceVectorMatrix, 
                  deviceVectorMatrixTranspose, 
                  deviceVectorMagnitudes, 
                  deviceCentroids, 
                  deviceCentroidMagnitudes, 
                  deviceCurrentClusters, 
                  deviceFutureClusters, 
                  context, k, maxTrials) != 0) {
        printf("Error runing clustering algorithm\n");
        
        cleanUpDeviceResources(deviceVectorMatrix, deviceFutureClusters, deviceCurrentClusters, 
                               deviceCentroids, deviceVectorMatrixTranspose, deviceVectorMagnitudes, 
                               deviceCentroidMagnitudes, context);
        return 1;   
    }
    
    
    
    cleanUpDeviceResources(deviceVectorMatrix, deviceFutureClusters, deviceCurrentClusters, 
                           deviceCentroids, deviceVectorMatrixTranspose, deviceVectorMagnitudes, 
                           deviceCentroidMagnitudes, context);
    
    printf("Program finished successfully\n");
    
    //stop timing
    clock_t stop = clock();
    double elapsed = (double)(stop - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("Using CPU timer (Wall clock time) took %f ms\n", elapsed);
    
    return 0;
}