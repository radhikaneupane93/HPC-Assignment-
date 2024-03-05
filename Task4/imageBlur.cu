/*
Filename: ImageBlur.cu
Radhika Neupane 
2227097

The execution and compilation command in terminal 
nvcc -o imageBlur imageBlur.cu lodepng.cpp
./imageBlur input.png

*/

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

//Global Function for applying box blur to an image 
__global__ void ImageBlur(unsigned char * gpu_imgOutput, unsigned char * gpu_imgInput,unsigned int w,unsigned int h,unsigned int blur, unsigned int bluryMD){

	int red = 0;
	int green = 0;
	int blue = 0;
	int x,y;
	int C = 0;


	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int pixel = idx*4;
	
	//Nested loop to go through each nearby pixels in an image and add up their color values
	for(x = (pixel - (4 * blur)); x <=  (pixel + (4 * blur)); x+=4){
		if ((x > 0) && x < (h * w * 4) && ((x-4)/(4*w) == pixel/(4*w))){
			for(y = (x - (16 * w * blur)); y <=  (x + (16 * w *blur)); y+=(4*w)){
				if(y > 0 && y < ((h * w * 4))){
					red += gpu_imgInput[y];
					green += gpu_imgInput[1+y];
					blue += gpu_imgInput[2+y]; 
					C++;
				}
			}
		}
	}

	//Calculating the RGB values average and updating the output pixel
	gpu_imgOutput[pixel] = red / C;
	gpu_imgOutput[1+pixel] = green / C;
	gpu_imgOutput[2+pixel] = blue / C;
	gpu_imgOutput[3+pixel] = gpu_imgInput[3+pixel];
}

//Driver Code 
int main(int argc, char **argv){



	unsigned int bluryMD = 3;
	unsigned int blur = (bluryMD - 1) / 2;
	unsigned int error;
	unsigned int encryptError;
	unsigned char* img;
	unsigned int w;
	unsigned int h;
	const char* filename = "input.png";
	const char* newFileName = "output.png";
	
	//Deconding image file 
	error = lodepng_decode32_file(&img, &w, &h, filename);
	if(error){
		printf("error %u: %s\n", error, lodepng_error_text(error));
	}
	const int ARRAY_SIZE = w*h*4;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned char);

	unsigned char host_imgInput[ARRAY_SIZE * 4];
	unsigned char host_imgOutput[ARRAY_SIZE * 4];

	for (int i = 0; i < ARRAY_SIZE; i++) {
		host_imgInput[i] = img[i];
	}

	unsigned char * d_in;
	unsigned char * d_out;
	
	// Allocating memory on the GPU
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);
	
	//Copying the image from host to device
	cudaMemcpy(d_in, host_imgInput, ARRAY_BYTES, cudaMemcpyHostToDevice);

	//Calling CUDA Kernel to applu blur 
	ImageBlur<<<h, w>>>(d_out, d_in, w, h, blur, bluryMD);

	cudaMemcpy(host_imgOutput, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	//Encoding and Saving the blurred image
	encryptError = lodepng_encode32_file(newFileName, host_imgOutput, w, h);
	if(encryptError){
		printf("error occured %u: %s\n", error, lodepng_error_text(encryptError));
	}
	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
