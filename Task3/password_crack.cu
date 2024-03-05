/*
File Name: password_crack.cu
Radhika Neupane
2227097

**Compilation and Execution command in terminal 
nvcc -o password_crack password_crack.cu
./password_crack

To save the output in text file 
./password_crack > output.txt
*/

#include <stdio.h>
#include <stdlib.h>

//Device Function to copy strings from source to destination 
__device__ char * copy_strings(char *dest, const char *src){
  int i = 0;
  do {
    dest[i] = src[i];}
  while (src[i++] != 0);
  return dest;
}


//Device Function to compare the strings 
__device__ int compare_strings(const char *str_a, const char *str_b, unsigned len = 256){
	int match = 0;
	unsigned i = 0;
	unsigned done = 0;
	while ((i < len) && (match == 0) && !done) {
		if ((str_a[i] == 0) || (str_b[i] == 0)) {
			done = 1;
		}
		else if (str_a[i] != str_b[i]) {
			match = i+1;
			if (((int)str_a[i] - (int)str_b[i]) < 0) {
				match = 0 - (i + 1);
			}
		}
		i++;
	}
	return match;
  }

//Device Function to perform password encrption using CUDA
__device__ char* CudaCrypt(char* rawPassword){

	char * newPassword = (char *) malloc(sizeof(char) * 11);
	
 	//Logic to do encryption 
	newPassword[0] = rawPassword[0] + 2;
	newPassword[1] = rawPassword[0] - 2;
	newPassword[2] = rawPassword[0] + 1;
	newPassword[3] = rawPassword[1] + 3;
	newPassword[4] = rawPassword[1] - 3;
	newPassword[5] = rawPassword[1] - 1;
	newPassword[6] = rawPassword[2] + 2;
	newPassword[7] = rawPassword[2] - 2;
	newPassword[8] = rawPassword[3] + 4;
	newPassword[9] = rawPassword[3] - 4;
	newPassword[10] = '\0';

	for(int i =0; i<10; i++){
		if(i >= 0 && i < 6){
			if(newPassword[i] > 122){
				newPassword[i] = (newPassword[i] - 122) + 97;
			}else if(newPassword[i] < 97){
				newPassword[i] = (97 - newPassword[i]) + 97;
			}
		}else{
			if(newPassword[i] > 57){
				newPassword[i] = (newPassword[i] - 57) + 48;
			}else if(newPassword[i] < 48){
				newPassword[i] = (48 - newPassword[i]) + 48;
			}
		}
	}
	return newPassword;
}

//Global function to crack the encrpted password using CUDA 
__global__ void crack(char * alphabet, char * numbers, char * encPassword) {

	char genRawPass[4];

	genRawPass[0] = alphabet[blockIdx.x];
	genRawPass[1] = alphabet[blockIdx.y];

	genRawPass[2] = numbers[threadIdx.x];
	genRawPass[3] = numbers[threadIdx.y];
	if (compare_strings(CudaCrypt(genRawPass), encPassword) == 0) {
		copy_strings(encPassword, genRawPass);
	}
}

//Driver code 
int main(int argc, char ** argv){
	char cpuAlphabet[26] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
	char cpuNumbers[26] = {'0','1','2','3','4','5','6','7','8','9'};
	char inputEncPass[26] = "jfigac2223";

	char *decryptedPass;

	decryptedPass = (char *)malloc(sizeof(char) * 26);


	char * gpuAlphabet;
	cudaMalloc( (void**) &gpuAlphabet, sizeof(char) * 26); 
	cudaMemcpy(gpuAlphabet, cpuAlphabet, sizeof(char) * 26, cudaMemcpyHostToDevice);

	char * gpuNumbers;
	cudaMalloc( (void**) &gpuNumbers, sizeof(char) * 26); 
	cudaMemcpy(gpuNumbers, cpuNumbers, sizeof(char) * 26, cudaMemcpyHostToDevice);

	char *gpuPassword;
	cudaMalloc( (void**) &gpuPassword, sizeof(char) * 26);
	cudaMemcpy(gpuPassword, inputEncPass, sizeof(char) * 26, cudaMemcpyHostToDevice);

	crack<<< dim3(26,26,1), dim3(10,10,1) >>>( gpuAlphabet, gpuNumbers, gpuPassword );
	cudaMemcpy(decryptedPass, gpuPassword, sizeof(char) * 26, cudaMemcpyDeviceToHost);

	printf("\nEncrypted Password: %s,\tRaw Password: %s\n\n", inputEncPass, decryptedPass);
	free(decryptedPass);
	cudaFree(gpuAlphabet);
	cudaFree(gpuNumbers);
	cudaFree(gpuPassword);

	return 0;
}
