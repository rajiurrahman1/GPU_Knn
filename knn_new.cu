// Author: Rajiur Rahman ( rajiurrahman.bd@gmail.com )
// Department of Computer Science, Wayne State University
// knn implemented for GPU. 
// have to provide training data, trianing data label, test data, test data label in separate text files. All the files should be ' ' space separated. 

/* 	Instruction for compiling and running 
*	the commands should be following
*		compile - nvcc knn.cu -o knn.o
*		run - ./knn.o numTrainRow numTestRow numCol k
*		
*			For example: 
*			./knn.o 20 5 10 4
*			./knn.o 15997 4000 30000 5
*			./knn_new.o 69 20 1000 5
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */
#include <math.h>
#include <time.h>


/*__global__ void kernel(int *a)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
}*/

__device__ void showData(float* data, int numRow, int numCol, int tidx){
	for(int i =tidx*numCol; i<((tidx*numCol)+numCol); i++){
		printf("%d-%f\t", i, data[i]);
	}
}

// this function calculates the distance between two rows. 
__device__ float calculate_distance(float* d_trainData, float* d_testData, int numTrainRow, int numTestRow, int numCol, int currentTestIndex, int currentTrainIndex){
	float distance = (float) 100.0;
	for (int i = 0; i<numCol; i++){
		distance += ( d_testData[currentTestIndex*numCol+i] - d_trainData[currentTrainIndex*numCol+i] ) * ( d_testData[currentTestIndex*numCol+i] - d_trainData[currentTrainIndex*numCol+i] );
		
	}
	//distance = distance/(float)numCol;
	//distance = (float)sqrt(distance);
	
	return distance;
}

// this function will return nearest neighbor information for a particular test row. 
// called from main kernel // d_neighborhoodMatrix has size numTestRow*numTrainRow .. it is a flat array 
__device__ float* calculate_distance_matrix(float* d_trainData, float* d_testData, float* d_neighborhoodMatrix, int numTrainRow, int numTestRow, int numCol, int tidx){
	//printf("Dealing with test data row %d\n", tidx);
	for(int i=0; i<numTrainRow; i++){
		//distance form  rows testData[tidx] <--> trainData[i] 
		d_neighborhoodMatrix[tidx*numTrainRow+i] = calculate_distance(d_trainData, d_testData, numTrainRow, numTestRow, numCol, tidx, i); 
	}
	return d_neighborhoodMatrix;
}


// kernel function that will perform k-nearest neighbor classification
// There will be one karnel launched for each row in the test data. 
// This function will manage finding nearest neighbors from test data row to training data 
__global__ void calculate_similarity_matrix(float* d_trainData, float* d_testData, float* d_neighborhoodMatrix, int numTrainRow, int numTestRow, int numCol){
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	//printf("Inside Kernel %d\n", tidx);
	//showData(d_testData, numTestRow, numCol, tidx);
	
	//call function for calculating the nearest neighbor of tidx'th row of test data
	d_neighborhoodMatrix = calculate_distance_matrix(d_trainData, d_testData, d_neighborhoodMatrix, numTrainRow, numTestRow, numCol, tidx);
		
}




// this funciton will read the data. It takes parameters file name as string, number of rows and columns.
// It will read a 2-d matrix and save as a flat 1-D array and return the pointer
float* read_data(char* fileName, int numRow, int numCol){
	float* data;
	data = (float*)malloc(numRow*numCol*sizeof(float)); 
	FILE *file;
    file=fopen(fileName, "r");
	for (int i=0; i<numRow*numCol; i++){
		if (!fscanf(file, "%f", &data[i])){
				break;
		}
		//printf("%d-%f\n",i, data[i]); 
	}
	fclose(file);
	return data;
}

void show_data(float* data, int numRow, int numCol){
	/*printf("numrow-%d   numcol-%d\n\n", numRow, numCol);
	for(int i =0; i<numRow*numCol; i++){
		printf("%d-%f\t", i, data[i]);
	}*/
	
	for(int i=0; i< numRow; i++){
		for(int j=0; j<numCol; j++){
			printf("%f  ",data[i*numCol+j]);
		}
		printf("\n");
	}	
}



//comment
__device__ int calculate_nearest_neighbor_index(float* d_neighborhoodMatrix, int tidx, int numTrainRow){
	float minValue = 99999.0;
	int minIndex = 0;
	
	for (int i=(tidx*numTrainRow); i< (tidx*numTrainRow)+numTrainRow; i++){
		if (d_neighborhoodMatrix[i] <= minValue ){
			minValue = d_neighborhoodMatrix[i];
			minIndex = i-(tidx*numTrainRow);
		}
	}
	
	return minIndex;
}

// this function will calculate the nearest neighbors from similarity matrix
// a kernel will be launched for each test instance  
__global__ void calculate_nearest_neighbor(float* d_neighborhoodMatrix, int* d_nearestNeighborIndices, int numTrainRow, int numTestRow, int numCol, int k){
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	for(int i=0; i<k; i++){
		int minIndex = calculate_nearest_neighbor_index(d_neighborhoodMatrix, tidx, numTrainRow);
		d_nearestNeighborIndices[tidx*k+i] = minIndex;
		d_neighborhoodMatrix[tidx*numTrainRow+minIndex] = (float)99999.0;		
	}	
	
}

//calculate the label of a test instance
float calculate_label(int* host_nearestNeighborIndices, float* trainLabel, int numTestRow, int numTrainRow, int k, int currentInstance){
	float label = (float)0.0;
	float sum1 = (float)0.0;
	int currentIndex = 0;
	for(int i=0; i<k; i++){
		currentIndex = host_nearestNeighborIndices[currentInstance*k + i];
		//printf("%d - %f\n", currentIndex, trainLabel[currentIndex]);
		sum1 += trainLabel[currentIndex];
		//printf("currentIndex -%d, valueIndex-%f\n",currentIndex, trainLabel[currentIndex]);
	}
	printf("\n");
	//printf("nstance-%d sum-%f\n", currentInstance, sum1);
	
	if(sum1 >= (k/2) ){
		label = (float)1.0;
	}
	return label;
}

//predict class labels of test data from calculated nearest neighbors
float* predict_label(int* host_nearestNeighborIndices, float* trainLabel, int numTestRow, int numTrainRow, int k){
	float* predictedLabel;
	predictedLabel = (float*)malloc( numTestRow*sizeof(float) );
	for(int i=0; i<numTestRow; i++){
		predictedLabel[i] = calculate_label(host_nearestNeighborIndices, trainLabel, numTestRow, numTrainRow, k, i);
		//printf("label prediction of instance %d\n", i);
	}
	
	return predictedLabel;
}

//calculate accuracy of prediction from original class labels and predicted labels
float calculate_accuracy(float* predictedLabel, float* testLabel, int numTestRow){
	int correctPredictionCount = 0;
	for(int i=0; i<numTestRow; i++ ){
		//printf("original %f \t predicted %f\n", testLabel[i], predictedLabel[i]);
		if(predictedLabel[i] == testLabel[i]){
			correctPredictionCount ++;
		}
	}
	//printf("\n\n");
	return (float)100*((float)correctPredictionCount/(float)numTestRow);
}

void show_data_int(int* data, int numRow, int numCol){	
	for(int i=0; i< numRow; i++){
		for(int j=0; j<numCol; j++){
			printf("%d  ",data[i*numCol+j]);
		}
		printf("\n");
	}	
}

void show_data_nearest_neighbor_labels(int* data, float* trainLabel, int numRow, int numCol){	
	for(int i=0; i< numRow; i++){
		for(int j=0; j<numCol; j++){
			printf("%d  ",data[i*numCol+j]);
		}
		printf("\n");
	}	
}

int main(int argc, char* argv[])
{	
	//start the timer
	clock_t begin_time = clock();
	
	// first, catch the arguments from command line 
	int numTrainRow = atoi(argv[1]);
	int numTestRow = atoi(argv[2]);
	int numCol = atoi(argv[3]);
	int k = atoi(argv[4]);
	
	printf("\n**************** Hello World ! ******************\n");

	// read the data files
	float* trainData, *testData, *trainLabel, *trainLabel_1, *testLabel, *predictedLabel;
	trainData = read_data("train.txt", numTrainRow, numCol);	
	testData = read_data("test.txt", numTestRow, numCol);
	trainLabel = read_data("label_train.txt", numTrainRow, 1);
	testLabel = read_data("label_test.txt", numTestRow, 1);
	
	//trainData = read_data("vec_1k_train.txt", numTrainRow, numCol);	
	//testData = read_data("vec_1k_test.txt", numTestRow, numCol);
	//trainLabel = read_data("label_1k_train.txt", numTrainRow, 1);
	//testLabel = read_data("label_1k_test.txt", numTestRow, 1);
	
	printf("Data Read Complete\n");
	
	//show_data(testData, numTestRow, numCol);
	//printf("\n\n\n");
	//show_data(trainLabel, numTrainRow, 1);	
	//printf("\n\n\n");
	
	
	// allocate memory and copy read files to device (GPU) memory from host (CPU) memory
	float *d_trainData, *d_testData, *d_neighborhoodMatrix, *host_neighborhoodMatrix; //neighborhood matrix will have numTestRow rows and numTrainRow columns
	int *d_nearestNeighborIndices, *host_nearestNeighborIndices;		//it has numTestRow rows and k columns 
	const size_t trainSize = sizeof(float) * size_t(numTrainRow*numCol);
	const size_t testSize = sizeof(float) * size_t(numTestRow*numCol);
	const size_t neighborhoodMatrixSize = sizeof(float)*size_t(numTestRow*numTrainRow);
	cudaMalloc((void **)&d_trainData, trainSize);
	cudaMemcpy(d_trainData, trainData, trainSize, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_testData, testSize);
	cudaMemcpy(d_testData, testData, testSize, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_neighborhoodMatrix, neighborhoodMatrixSize);
	cudaMalloc((void **)&d_nearestNeighborIndices, (numTestRow*k*sizeof(int)) );
		
	calculate_similarity_matrix<<<1,numTestRow>>>(d_trainData, d_testData, d_neighborhoodMatrix, numTrainRow, numTestRow, numCol); 
	cudaFree(d_trainData);
	cudaFree(d_testData);
	printf("Similarity matrix building complete\n");
	
	// copy nearest neighbor matrix from device to CPU and print to view it
	host_neighborhoodMatrix = (float*)malloc(neighborhoodMatrixSize);
	cudaMemcpy(host_neighborhoodMatrix, d_neighborhoodMatrix, neighborhoodMatrixSize, cudaMemcpyDeviceToHost);
	printf("\ncopying similarity matrix from device to host complete\n" );
	//printf("\n\nSimilarity matrix\n");
	show_data(host_neighborhoodMatrix, numTestRow, numTrainRow);
	
	calculate_nearest_neighbor<<<1,numTestRow>>>(d_neighborhoodMatrix, d_nearestNeighborIndices, numTrainRow, numTestRow, numCol, k);
	printf("Nearest Neighbour calculation complete\n");
	
	// copy nearest neighbour indices from device (GPU) to host (CPU)
	host_nearestNeighborIndices = (int*)malloc(numTestRow*k*sizeof(int));
	cudaMemcpy(host_nearestNeighborIndices, d_nearestNeighborIndices, (numTestRow*k*sizeof(int)), cudaMemcpyDeviceToHost);
	
	//printf("\nCopying nearest neighbour indices from device to host complete\n");
	
	//printf("indices of nearest neighbour\n");
	//show_data_int(host_nearestNeighborIndices, numTestRow, k);
	
	predictedLabel = predict_label(host_nearestNeighborIndices, trainLabel, numTestRow, numTrainRow, k);	
	printf("\nClass label prediction complete\n");
	
	//show_data(predictedLabel, numTestRow, 1);
	float acc = calculate_accuracy(predictedLabel, testLabel, numTestRow);
	printf("\nPrediction Accuracy: %f", acc);
	
	
	
	//take the end time and print time taken for running the program
	clock_t end_tiem = clock();
	double diff_time = (end_tiem - begin_time) / CLOCKS_PER_SEC;
	printf("\n\nTime taken for running the program: %lf\n\n", diff_time);
	
	free(testData);
	free(trainData);
	free(host_nearestNeighborIndices);

	free(host_neighborhoodMatrix);
	cudaFree(d_neighborhoodMatrix);
	
	
	return 0;
}



