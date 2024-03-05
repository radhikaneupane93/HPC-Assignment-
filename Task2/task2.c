/*
File: task2.c
Name: Radhika Neupane
UNI ID: 2227097

This program performs matrix multiplication using multithreading.

Compilation in Terminal:
gcc -o task2 task2.c -lpthread

Execution
./task2 matrixfile.txt numthreads
./task2 matrixfile.txt 4
*/

#include <stdio.h> // Standard input/output functions
#include <stdlib.h> //Memory allocation functions
#include <string.h> //String manipulation functions
#include <pthread.h> //Multi-threading functions

// Struct to represent matrix
typedef struct Matrix {
    int row;
    int col;
    double **values;
} Matrix;

// Struct to hold information about a thread
typedef struct ThreadInfo {
    int rank;
    int start;
    int end;
} ThreadInfo;

ThreadInfo *ti;
Matrix matrixA;
Matrix matrixB;
Matrix matrixC;

// Functional Declarations
void *threadRunner(void *rank);
void getSize(FILE *fp, Matrix *matrix);
void readFile(char filename[], Matrix *matrix);
void printMatrix(Matrix *matrix);
int canMultiply(Matrix *matrixA, Matrix *matrixB);
void multiply(int rank);
void saveMatrix(Matrix *matrix);

// Function to allocate memory for a matrix
Matrix allocateMatrix(int row, int col) {
    Matrix matrix;
    matrix.row = row;
    matrix.col = col;
    matrix.values = (double **)malloc(row * sizeof(double *));
    for (int i = 0; i < row; i++) {
        matrix.values[i] = (double *)malloc(col * sizeof(double));
    }
    return matrix;
}

// Function to read matrix from file
void readMatricesFromFile(const char *filename, Matrix *matrixa, Matrix *matrixb) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    // Read and assign matrixa
    fscanf(file, "%d,%d", &(matrixa->row), &(matrixa->col));
    matrixa->values = (double **)malloc(matrixa->row * sizeof(double *));
    for (int i = 0; i < matrixa->row; i++) {
        matrixa->values[i] = (double *)malloc(matrixa->col * sizeof(double));
        for (int j = 0; j < matrixa->col; j++) {
            fscanf(file, "%lf,", &(matrixa->values[i][j]));
        }
    }

    // Read and assign matrixb
    fscanf(file, "%d,%d", &(matrixb->row), &(matrixb->col));
    matrixb->values = (double **)malloc(matrixb->row * sizeof(double *));
    for (int i = 0; i < matrixb->row; i++) {
        matrixb->values[i] = (double *)malloc(matrixb->col * sizeof(double));
        for (int j = 0; j < matrixb->col; j++) {
            fscanf(file, "%lf,", &(matrixb->values[i][j]));
        }
    }

    fclose(file);
}

// Reading matrix data from the matrixfile.txt
void readFile(char filename[], Matrix *matrix) {
    FILE *numfile = fopen("matrixfile.txt", "r");
    char *piece;
    int currentRow = 0;
    int currentCol = 0;

    if (numfile == NULL) {
        printf("File Read error\n");
        exit(-1);
    }

    getSize(numfile, matrix);

    // Reading each line from the file and reviewing comma-separated values
    char currentLine[1000];
    while (fgets(currentLine, sizeof(currentLine), numfile)) {
        piece = strtok(currentLine, ",");
        while (piece != NULL) {
            matrix->values[currentRow][currentCol] = atof(piece);
            piece = strtok(NULL, ",");
            currentCol++;
        }

        currentRow++;
        currentCol = 0;
    }

    fclose(numfile);
}

void getSize(FILE *fp, Matrix *matrix) {
    char ch;

    // Counting the number of rows in the file and resetting file position to the beginning
    // at last using rewind()
    matrix->row = 1;
    while ((ch = fgetc(fp)) != EOF) {
        if (ch == '\n') {
            matrix->row++;
        }
    }
    rewind(fp);

    // Counting the number of columns in the file and resetting file position to the beginning
    // at last using rewind()
    matrix->col = 1;
    while ((ch = fgetc(fp)) != '\n') {
        if (ch == ',') {
            matrix->col++;
        }
    }
    rewind(fp);

    // Allocating memory for matrix values
    matrix->values = malloc(sizeof(double *) * matrix->row);
    for (int i = 0; i < matrix->row; ++i) {
        matrix->values[i] = malloc(sizeof(double) * matrix->col);
    }
}

// Printing matrix values in a readable format
void printMatrix(Matrix *matrix) {
    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++) {
            printf("%f|", matrix->values[i][j]);
        }
        printf("\n");
    }
}

// Checking if matrices can be multiplied
int canMultiply(Matrix *matrixA, Matrix *matrixB) {
    return (matrixA->col == matrixB->row) ? 1 : 0;
}

// Function for each thread in order to perform matrix multiplication on assigned rows
void *threadRunner(void *rank) {
    int threadRank = *(int *) rank;
    for (int i = ti[threadRank].start; i <= ti[threadRank].end; ++i) {
        multiply(i);
    }
    pthread_exit(NULL);
}

// Multiplying specific rows multiplication
void multiply(int rank) {
    for (int i = 0; i < matrixC.col; ++i) {
        for (int j = 0; j < matrixA.col; ++j) {
            matrixC.values[rank][i] += matrixA.values[rank][j] * matrixB.values[j][i];
        }
    }
}

// Saving the matrix in a file called Output.txt
void saveMatrix(Matrix *matrix) {
    FILE *fp = fopen("Output.txt", "w");
    if (fp == NULL) {
        printf("Error opening file for writing.\n");
        exit(-1);
    }

    for (int i = 0; i < matrix->row; i++) {
        for (int j = 0; j < matrix->col; j++) {
            if (j == (matrix->col - 1)) {
                fprintf(fp, "%f", matrix->values[i][j]);
            } else {
                fprintf(fp, "%f,", matrix->values[i][j]);
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

// Driver Code
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s matrixfile.txt numthreads\n", argv[0]);
        exit(-1);
    }

    char *file = argv[1];
    int threadNum = atoi(argv[2]);

    readMatricesFromFile(file, &matrixA, &matrixB);

    printf("Matrix A:\n");
    printMatrix(&matrixA);

    printf("Matrix B:\n");
    printMatrix(&matrixB);

    // Checking if matrices can be multiplied or not
    if (!canMultiply(&matrixA, &matrixB)) {
        printf("Matrices cannot be multiplied.\n");
        exit(0);
    }

    // Allocating memory for the result matrix and thread information
    matrixC.row = matrixA.row;
    matrixC.col = matrixB.col;

    // Debugging print statements
    printf("Allocating memory for rows: %d\n", matrixC.row);
    printf("Allocating memory for columns: %d\n", matrixC.col);

    matrixC.values = malloc(sizeof(double *) * matrixC.row);
    if (matrixC.values == NULL) {
        printf("Error allocating memory for rows.\n");
        exit(-1);
    }

    for (int i = 0; i < matrixC.row; ++i) {
        matrixC.values[i] = malloc(sizeof(double) * matrixC.col);
        if (matrixC.values[i] == NULL) {
            printf("Error allocating memory for columns in row %d.\n", i);
            exit(-1);
        }
    }

    ti = malloc(sizeof(ThreadInfo) * threadNum);

    int maxThreads = (matrixA.row > matrixB.col) ? matrixB.col : matrixA.row;

    int chunkSize = matrixA.row / maxThreads;

    // Distributing work among threads
    for (int i = 0; i < maxThreads; ++i) {
        ti[i].rank = i;
        if (i == 0) {
            ti[i].start = i;
            ti[i].end = i + chunkSize - 1;
            continue;
        }
        if (i == (maxThreads - 1)) {
            ti[i].start = ti[i - 1].end + 1;
            ti[i].end = (matrixA.row - 1);
            break;
        }
        ti[i].start = ti[i - 1].end + 1;
        ti[i].end = ti[i].start + chunkSize;
    }

    // Creating threads for matrix multiplication
    pthread_t thread_ids[maxThreads];
    for (int i = 0; i < maxThreads; ++i) {
        int returnValue = pthread_create(&thread_ids[i], NULL, threadRunner, (void *) &ti[i].rank);
        if (returnValue != 0) {
            printf("Error occurred while creating threads. value=%d\n", returnValue);
        }
    }

    for (int local_rank = 0; local_rank < maxThreads; ++local_rank) {
        pthread_join(thread_ids[local_rank], NULL);
    }

    printf("\nMatrix C is:\n");
    printMatrix(&matrixC);
    saveMatrix(&matrixC);
    printf("\nResults saved as Output.txt\n\n");

    // Free memory allocation
    free(ti);
    for (int i = 0; i < matrixA.row; ++i) {
        free(matrixA.values[i]);
    }
    free(matrixA.values);

    for (int i = 0; i < matrixB.row; ++i) {
        free(matrixB.values[i]);
    }
    free(matrixB.values);

    for (int i = 0; i < matrixC.row; ++i) {
        free(matrixC.values[i]);
    }
    free(matrixC.values);

    return 0;
}
