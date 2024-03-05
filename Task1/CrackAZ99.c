/*
File: CrackAZ99.c
Name: Radhika Neupane
UNI ID: 2227097

This program is a multithreaded password cracker designed to decrypt a 4-character password encrypted.
The password format is "LetterLetterNumberNumber", for example, "HP93"
*/


#include <stdio.h> // Standard Input/Output functions
#include <string.h> // String manipulation functions
#include <stdlib.h> //Memory allocation functions
#include <crypt.h> //Cryptography functions for password hashing
#include <unistd.h> //UNIX Standard functions, including sleep
#include <semaphore.h> //Semaphore functions for synchronization
#include <pthread.h> //Multi-threading functions

int count = 0;
int totalThread;
int threadCount = 26;

// Structure to hold start and end values for each thread
struct threadInfo {
    int start;
    int end;
};

char startChar, endChar;

char *salt_and_encrypted; // Pointer to store the salt and encrypted password

sem_t sem; // Semaphore for synchronization

/*
Function to extract a substring from a source string
memcpy function is used to copy a specified number of bytes from the
source string (src) to the destination string (dest)
*/

void substr(char *dest, char *src, int start, int length) {
    memcpy(dest, src + start, length);
    *(dest + length) = '\0';
}

// Function to crack the encrypted password
void *crack(void *args) {
    int x, y, z;
    char salt[7];
    char plain[7];
    char *enc;

    char ascii_to_char;

    substr(salt, salt_and_encrypted, 0, 6);

    /*Casting the void pointer argument to the struct type
    starting and ending ASCII value for this thread*/

    struct threadInfo *tI = (struct threadInfo *)args;
    int startNum = tI->start;
    int endNum = tI->end;

    sem_wait(&sem);

    printf("Looping from ASCII start value: %d\n", startNum);
    printf("Looping to ASCII end value: %d\n", endNum);

    // Loop through ASCII values from startNum to endNum
    for (x = startNum; x <= endNum; x++) {
        ascii_to_char = x;
        for (y = 'A'; y <= 'Z'; y++) {
            for (z = 0; z <= 99; z++) {
                 // For two letter, and two-digit number as per question
                sprintf(plain, "%c%c%02d", ascii_to_char, y, z);
                enc = (char *)crypt(plain, salt);
                count++;
                //if statement to check if the encrypted password matches the target or not
                if (strcmp(salt_and_encrypted, enc) == 0) {
                    printf("#%-8d%s %s\n", count, plain, enc);
                    exit(0);
                }
            }
        }
    }

    sem_post(&sem);

    pthread_exit(0);
}

//Function to set up and execute the parallel cracking process
void functionKernal() {
    int sliceList[threadCount];
    int rem = threadCount % totalThread;

    //Distributing the password combinations equally for all the threads
    for (int i = 0; i < totalThread; i++) {
        sliceList[i] = threadCount / totalThread;
    }

    //For remaining
    for (int j = 0; j < rem; j++) {
        sliceList[j] = sliceList[j] + 1;
    }

    int startList[totalThread];
    int endList[totalThread];

    // Calculating the start and end of ASCII values for each thread
    for (int k = 0; k < totalThread; k++) {
        if (k == 0) {
            startList[k] = 65;
        } else {
            startList[k] = endList[k - 1] + 1;
        }

        endList[k] = startList[k] + sliceList[k] - 1;

        printf("\nstartList[%d] = %d\t\tendList[%d] = %d", k, startList[k], k, endList[k]);
    }

    struct threadInfo threadDetails[totalThread];

    for (int l = 0; l < totalThread; l++) {
        threadDetails[l].start = startList[l];
        threadDetails[l].end = endList[l];
    }
    pthread_t thread_id[totalThread];

    sem_init(&sem, 0, 1);
    printf("\n\n |Creating threads and checking for a matching hash| \n");

     // Defining the encrypted password to be cracked inside ""
    salt_and_encrypted = "$6$AS$9IwGTn5WbHSalUs4ba3JbOfOUX/v1yD71Z4M2F6Yusz5k2WQEOFxqLIY80tudGtcFttqr/Zq6RIPjHkl/t2Pp1";

    printf("salt_and_encrypted: %s\n", salt_and_encrypted);

    //For thread execution
    for (int m = 0; m < totalThread; m++) {
        pthread_create(&thread_id[m], NULL, crack, &threadDetails[m]);
    }
    for (int n = 0; n < totalThread; n++) {
        pthread_join(thread_id[n], NULL);
    }
    sem_destroy(&sem);
}

//Function to display an error message and exit the program if any incorrect command is provided
void inputArguments(char *program_name) {
    fprintf(stderr, "Arguments should be in the order as specified:   %s <number of threads>\n", program_name);
    fprintf(stderr, "Where number of threads should be > 0\n");
    exit(0);
}

//Function to validate command line arguments
void getArguments(int argc, char *argv[]) {
    if (argc != 2) {
        inputArguments(argv[0]);
    }

    totalThread = strtol(argv[1], NULL, 10);

    if (totalThread <= 0) {
        inputArguments(argv[0]);
    }
}

//Driver Code
int main(int argc, char *argv[]) {
    getArguments(argc, argv);
    functionKernal();

    printf("%d solutions explored\n", count);

    return 0;
}
