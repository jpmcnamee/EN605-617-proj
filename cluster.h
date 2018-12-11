#include "cusparse.h"

/*
    File contains primary functions so that they are visible to all that import this header
    Also contains and structs that are used throughout the program
*/

/*
    Stores a sparse matrix either in CSR, COO, or CSC format.
    Basically this means that it holds only the non-zero values of the matrix and the row and column
    arrays hold the indecies of the non-zero values.
*/
struct sparse_matrix {
	int * rowIndexPtr; // value row index array
    int * colIndexPtr; //value column index array
    float * valPtr; //array of non-zero values
    unsigned int valLength;
    unsigned int numRows;
    unsigned int numCols;
};

/*
    A wrapper that holds cusparse handle and descriptors used for most cusparse operation
*/
struct device_context {
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;  
};

//Device util methods
int initDeviceContext(device_context **context);
void deleteDeviceContext(device_context *context);
int transferDeviceMatrix(sparse_matrix **hostMatrix, sparse_matrix *deviceMatrix, int csrFormat);
int setUpDeviceVectors(sparse_matrix *hostMatrix, sparse_matrix **deviceMatrix, device_context *context);
void delete_sparse_matrix_device(sparse_matrix * matrix);
int pickRandomFutureClusters(sparse_matrix **deviceRandomClusters, int k, int numVectors, device_context *context);
int setUpDeviceMagnitudes(float *hostVectorMagnitudes, float **deviceVectorMagnitudes, int numVectors, float **deviceCentroidMagnitudes, int k);
void deleteDevicemagnitude(float *deviceMagnitudes);
int transferDeviceVector(float **hostVector, float *deviceVector, int vectorLength);
int initializeEmptyDeviceCluster(sparse_matrix **deviceClusters, int k, int numVectors, device_context *context);
int checkcuSparse(cusparseStatus_t status);
int checkCuda(cudaError_t status);
void clear_sparse_matrix_device(sparse_matrix * matrix);
int checkConsistencyOfDeviceVectorI(int *deviceVector, unsigned int lenvectorLengthgth);
int checkConsistencyOfDeviceVectorF(float *deviceVector, unsigned int lenvectorLengthgth);

//Kerne methods
__global__ void kernel_normalize_and_find_closest(int *rowIndex, int *colIndex, float *valIndex, 
                                       float *vectorMagnitude, int numVectors, 
                                       float *centroidMagnitude, int numCentroids, int *result);
__global__ void kernel_compute_centroid_lengths(int *rowIndex, float *valIndex, 
                                       float *centroidMagnitude, int numCentroids);
__global__ void kernel_subtract_abs(int *colIndexOld, int *colIndexNew, int length);
__global__ void kernel_average_clusters(float *centroidTotals, int numVectors, int *colIndex, float *valIndex);
__global__ void kernel_diff(int *colIndexOld, int *colIndexNew, int length);

//Math util methods
int multiplySparseMatricies(sparse_matrix *device_a, sparse_matrix *device_b, sparse_matrix **device_c, device_context *context);
int computeCentriods(sparse_matrix *vectors, sparse_matrix *clusters, sparse_matrix **centroids, device_context *context);
int updateCentroidMagnitudes(sparse_matrix *centroids, float *deviceCentroidMagnitudes, device_context *context);

int computeTranspose(sparse_matrix *deviceMatrix, sparse_matrix **deviceMatrixTranspose, device_context *context);
int findFutureClusters(sparse_matrix *vectors, float * deviceVectorMagnitudes, 
                        sparse_matrix *centroids, float * deviceCentroidMagnitudes,
                        sparse_matrix *futureClusters, device_context *context);
int findNumDiffs(int *numDiffs, sparse_matrix *pastClusters, sparse_matrix *futureClusters);
int averageClusters(sparse_matrix *futureClusters, int k);    
    

//File util methods
int loadData(const char * fileName, sparse_matrix ** matrix, float ** magnitudes, int maxRecords);

void delete_sparse_matrix_host(sparse_matrix * matrix);


//Test util methods
int loadTestData(sparse_matrix ** matrix, float ** magnitudes);
int pickTestFutureClusters(sparse_matrix **futureClusters, device_context *context);
void printSparseMatrixDevice(sparse_matrix *deviceMatrix, int details, int cscFormat);
void printMagnitudesDevice(float *deviceMagnitudes, int magnitudeLength);
int checkConsistencyOfDeviceMatrix(sparse_matrix *deviceMatrix);