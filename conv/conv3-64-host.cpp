#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <sys/time.h>
#include <CL/opencl.h>
#include <stdlib.h>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define DPRINTF(...) printf(__VA_ARGS__); fflush(stdout);

#define NUM_QUEUES_TO_CREATE 3
#define NUM_KERNELS_TO_CREATE 3
#define AOCX_FILE "conv3-64.aocx"

#define CHECK(status)                                                           \
        if (status != CL_SUCCESS)                                               \
{                                                                       \
        printf("error %d in line %d.\n", status, __LINE__);    \
        exit(1);                                                        \
}

const int num_Aargs = 0;
const int A_args[num_Aargs] = {};
const int num_Bargs = 0;
const int B_args[num_Bargs] = {};
const int num_Cargs = 0;
const int C_args[num_Cargs] = {};

const char* kernel_name[] = {
"input_serializer",
"weight_loader",
"result_consumer",
};

double compute_kernel_execution_time(cl_event &event, double &start_d, double &end_d)
{
  cl_ulong start, end;

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);

  start_d = (double)1.0e-9 * start;
  end_d   = (double)1.0e-9 * end;
  //return (double)(end-start);
  return (double)1.0e-9 * (end - start); // nanoseconds to seconds
}

int main() {
    float* A;
    float* B;
    float* C;

    float input[BATCH][NOY + KY - 1][NOX + KX - 1][NIF];
    float weight[NOF][KY][KX][NIF];
    float result[BATCH*NOY*NOX*NOF];

    const int TILE_X = NOX / POX;
    const int TILE_Y = NOY / POY;
    const int TILE_NO = NOF / POF;
    float A_serializer[BATCH * TILE_Y*(POY + KY - 1) * TILE_X*(POX + KX - 1) * NIF];
    float B_serializer[NOF * KY * KX * NIF];

    A = &A_serializer[0];
    B = &B_serializer[0];
    C = &result[0];

    int i;
    int num_elem_A = BATCH * TILE_Y*(POY + KY - 1) * TILE_X*(POX + KX - 1) * NIF;
    int num_elem_B = NOF * KY * KX * NIF;
    int num_elem_C = BATCH * NOY * NOX * NOF;

    srand(time(0));
    for (int b = 0; b < BATCH; b++) {
      for (int i = 0; i < NOY + KY - 1; i++) {
        for (int j = 0; j < NOX + KX - 1; j++) {
          for (int ni = 0; ni < NIF; ni++) {
            input[b][i][j][ni] = rand() % 1000;
          }
        }
      }
    }
    
    for (int no = 0; no < NOF; no++) {
      for (int i = 0; i < KY; i++) {
        for (int j = 0; j < KX; j++) {
          for (int ni = 0; ni < NIF; ni++) {
            weight[no][i][j][ni] = (1.0 * rand()) / RAND_MAX;
          }
        }
      }
    }

    int index = 0;
    for (int b = 0; b < BATCH; b++) {
      for (int i = 0; i < NOY; i++) {
        for (int j = 0; j < NOX; j++) {
          for (int no = 0; no < NOF; no++) {
            result[index] = 0;
          }
        }
      }
    }

    int S1 = NIF;
    int S2 = S1 * (NOX + KX - 1);
    int S3 = S2 * (NOY + KY - 1);
    index = 0;
    for (int b = 0; b < BATCH; b++) {
        for (int y = 0; y < TILE_Y; y++) {
            for (int x = 0; x < TILE_X; x++) {
                for (int yy = 0; yy < POY + KY - 1; yy++) {
                    for (int xx = 0; xx < POX + KX - 1; xx++) {
                        for (int ni = 0; ni < NIF; ni++) {
                            A_serializer[index] = input[b][y*POY + yy][x*POX + xx][ni];
                            index++;
                        }
                    }
                }
            }
        }
    }

    index = 0;
    for (int no = 0; no < TILE_NO; no++) {
        for (int i = 0; i < KY; i++) {
            for (int j = 0; j < KX; j++) {
                for (int nn = 0; nn < POF; nn++) {
                    for (int ni = 0; ni < NIF; ni++) {
                        B_serializer[index] = weight[no*POF + nn][i][j][ni];
                        index++;
                    }
                }
            }
        }
    }

    DPRINTF("\n===== Host-CPU setting up the OpenCL platform and device ======\n\n");

    // Use this to check the output of each API call
    cl_int status;

    //----------------------------------------------
    // Discover and initialize the platforms
    //----------------------------------------------
    cl_uint numPlatforms = 0;
    cl_platform_id* platforms = NULL;

    // Use clGetPlatformIDs() to retrieve the
    // number of platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    DPRINTF("Number of platforms = %d\n", numPlatforms);

    // Allocate enough space for each platform
    platforms = (cl_platform_id*) malloc (numPlatforms * sizeof(cl_platform_id));

    DPRINTF("Allocated space for Platform\n");

    // Fill in platforms with clGetPlatformIDs()
    status = clGetPlatformIDs(numPlatforms, platforms, NULL); CHECK(status);
    DPRINTF("Filled in platforms\n")

    //----------------------------------------------
    // Discover and initialize the devices
    //----------------------------------------------

    cl_uint numDevices = 0;

    // Device info
    char buffer[4096];
    unsigned int buf_uint;
    int device_found = 0;
    const cl_uint maxDevices = 4;
    cl_device_id devices[maxDevices];
    DPRINTF("Initializing IDs\n");
    for (int i=0; i<numPlatforms; i++) {
        status = clGetDeviceIDs(platforms[i],
			CL_DEVICE_TYPE_ALL,
			maxDevices,
			devices,
			&numDevices);

	if(status == CL_SUCCESS){
        clGetPlatformInfo(platforms[i],
            CL_PLATFORM_NAME,
            4096,
            buffer,
            NULL);
#if defined(ALTERA_CL)
			if(strstr(buffer, "Altera") != NULL){
				device_found = 1;
			}
			DPRINTF("%s\n", buffer);
#elif defined(NVIDIA_CL)
			if(strstr(buffer, "NVIDIA") != NULL){
				device_found = 1;
			}
#else
			if(strstr(buffer, "Intel") != NULL){
				device_found = 1;
			}
#endif

DPRINTF("Platform found : %s\n", buffer);
device_found = 1;
/*
			if(device_found){
				// Allocate enough space for each device
				devices = (cl_device_iinput_loader_to_convd*)
					malloc (numDevices * sizeof(cl_device_id));
				// Fill in devices with clGetDeviceIDs()
				status = clGetDeviceIDs(platforms[i],
						CL_DEVICE_TYPE_ALL,
						numDevices,
						devices,
						NULL);
				break;
			}
*/
		}
	}

	if(!device_found) {
		DPRINTF("failed to find a OpenCL device\n");
		exit(-1);
	}
	DPRINTF("Total number of devices: %d", numDevices);

	for (i = 0; i < numDevices; i++) {
        clGetDeviceInfo(devices[i],
            CL_DEVICE_NAME,
            4096,
            buffer,
            NULL);
		DPRINTF("\nDevice Name: %s\n", buffer);

		clGetDeviceInfo(devices[i],
            CL_DEVICE_VENDOR,
            4096,
            buffer,
            NULL);
		DPRINTF("Device Vendor: %s\n", buffer);

		clGetDeviceInfo(devices[i],
            CL_DEVICE_MAX_COMPUTE_UNITS,
            sizeof(buf_uint),
            &buf_uint,
            NULL);
		DPRINTF("Device Computing Units: %u\n", buf_uint);

		clGetDeviceInfo(devices[i],
            CL_DEVICE_GLOBAL_MEM_SIZE,
            sizeof(unsigned long),
            &buffer,
            NULL);
		DPRINTF("Global Memory Size: %i\n", *((unsigned long*)buffer));

		clGetDeviceInfo(devices[i],
            CL_DEVICE_MAX_MEM_ALLOC_SIZE,
            sizeof(unsigned long),
            &buffer,
            NULL);
		DPRINTF("Global Memory Allocation Size: %i\n\n", *((unsigned long*)buffer));
	}

	//----------------------------------------------
	// Create a context
	//----------------------------------------------

    DPRINTF("\n===== Host-CPU setting up the OpenCL command queues ======\n\n");

	cl_context context = NULL;

    // Create a context using clCreateContext() and
	// associate it with the device

	context = clCreateContext(
            NULL,
			1,
			devices,
			NULL,
			NULL,
			&status); CHECK(status);

    //----------------------------------------------
	// Create command queues
	//---------------------------------------------

	cl_command_queue cmdQueue[NUM_QUEUES_TO_CREATE+1]; // extra queue for reading buffer C

	// Create a command queue using clCreateCommandQueue(),
	// and associate it with the device you want to execute on
	for(i=0; i<NUM_QUEUES_TO_CREATE; i++) {
        //fDPRINTF(stdout,"cmdQueue i = %d\n", i);
        cmdQueue[i] = clCreateCommandQueue(
            context,
            devices[0],
            CL_QUEUE_PROFILING_ENABLE,
            &status); CHECK(status);
	}

        //fDPRINTF(stdout,"cmdQueue i = %d, a queue for reading the C buffer\n", i);
        cmdQueue[i] = clCreateCommandQueue(
                            context,
                            devices[0],
                            CL_QUEUE_PROFILING_ENABLE,
                            &status); CHECK(status);

    //----------------------------------------------
	// Create device buffers
	//----------------------------------------------

	cl_mem d_matrix_mul_outputC;
	cl_mem d_matrix_mul_inputA;
	cl_mem d_matrix_mul_inputB;

    DPRINTF("\n===== Host-CPU transferring matrices A,B to the FPGA device global memory (DDR4) via PCIe ======\n\n");
    d_matrix_mul_inputA = clCreateBuffer(
		context,
		//CL_MEM_READ_ONLY | CL_MEM_BANK_1_ALTERA,
		CL_MEM_READ_ONLY,
		num_elem_A*sizeof(cl_float),
		NULL,
		&status); CHECK(status);

	d_matrix_mul_inputB = clCreateBuffer(
		context,
		//CL_MEM_READ_ONLY | CL_MEM_BANK_2_ALTERA,
		CL_MEM_READ_ONLY,
		num_elem_B*sizeof(cl_float),
		NULL,
		&status); CHECK(status);

	d_matrix_mul_outputC = clCreateBuffer(
		context,
		//CL_MEM_WRITE_ONLY | CL_MEM_BANK_1_ALTERA,
		CL_MEM_WRITE_ONLY,
		num_elem_C*sizeof(cl_float),
		NULL,
		&status); CHECK(status);


    //----------------------------------------------
	// Write host data to device buffers
	//----------------------------------------------

    // blocking writes
	status = clEnqueueWriteBuffer(
		cmdQueue[0],
		d_matrix_mul_inputA,
		CL_TRUE,
		0,
		num_elem_A*sizeof(cl_float),
		A,
		0,
		NULL,
		NULL); CHECK(status);

	status = clEnqueueWriteBuffer(
		cmdQueue[1],
		d_matrix_mul_inputB,
		CL_TRUE,
		0,
		num_elem_B*sizeof(cl_float),
		B,
		0,
		NULL,
		NULL); CHECK(status);


    //----------------------------------------------
	// Create the program from binaries
	//----------------------------------------------
    DPRINTF("\n===== Host-CPU setting up OpenCL program and kernels ======\n\n");

   	cl_program program;

	size_t binary_length;
	const unsigned char *binary;

    DPRINTF("\nAOCX file: %s\n\n", AOCX_FILE);
    fflush(stdout);
    // create the program using binary already compiled offline using aoc (i.e. the .aocx file)
	FILE *fp = fopen(AOCX_FILE, "rb");

	if (fp == NULL) {
		DPRINTF("Failed to open the AOCX file (fopen).\n");
		return -1;
	}

	fseek(fp, 0, SEEK_END);
	binary_length = ftell(fp);
	binary = (unsigned char*) malloc(sizeof(unsigned char) * binary_length);
	assert(binary && "Malloc failed");
	rewind(fp);

	if (fread((void*)binary, binary_length, 1, fp) == 0) {
		DPRINTF("Failed to read from the AOCX file (fread).\n");
		return -1;
	}
	fclose(fp);

    DPRINTF("Create program with binary\n");
	// Create a program using clCreateProgramWithBinary()
	program = clCreateProgramWithBinary(
			context,
			1,
			devices,
			&binary_length,
			(const unsigned char **)&binary,
			&status,
			NULL); CHECK(status);


    //----------------------------------------------
	// Create the kernel
	//----------------------------------------------

	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(status != CL_SUCCESS) {
		char log[128*1024] = {0};
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 128*1024, log, NULL);
		DPRINTF("%s\n", log);
		CHECK(status);
	}

	cl_kernel kernel[NUM_KERNELS_TO_CREATE];


	for(int j=0; j<NUM_KERNELS_TO_CREATE; j++) {
        DPRINTF("Creating kernel[%d]: %s\n", j,kernel_name[j]);
        kernel[j] = clCreateKernel(program, (const char*)kernel_name[j], &status);
		CHECK(status);
	}
    DPRINTF("All kernels created\n");


    // A
    for (i = 0; i < num_Aargs; i++) {
        status = clSetKernelArg(
    				kernel[0],
    				i,
    				sizeof(int),
    				(void*)&A_args[i]
                               ); CHECK(status);
    }
    status = clSetKernelArg(
  			kernel[0],
  			num_Aargs,
  			sizeof(cl_mem),
  			(void*)&d_matrix_mul_inputA
                        ); CHECK(status);
    // This is for the shared memory
    // status = clSetKernelArg(
  		// 	kernel[0],
  		// 	num_Aargs + 1,
  		// 	sizeof(cl_mem),
  		// 	(void*)NULL
    //                     ); CHECK(status);
    // B
    for (i = 0; i < num_Bargs; i++) {
        status = clSetKernelArg(
    				kernel[1],
    				i,
    				sizeof(int),
    				(void*)&B_args[i]
                               ); CHECK(status);
    }
    status = clSetKernelArg(
  			kernel[1],
  			num_Bargs,
  			sizeof(cl_mem),
  			(void*)&d_matrix_mul_inputB
                        ); CHECK(status);
    // This is for the shared memory
    // status = clSetKernelArg(
  		// 	kernel[1],
  		// 	num_Bargs + 1,
  		// 	sizeof(cl_mem),
  		// 	(void*)NULL
    //                     ); CHECK(status);
    // C
    for (i = 0; i < num_Cargs; i++) {
        status = clSetKernelArg(
    				kernel[2],
    				i,
    				sizeof(int),
    				(void*)&C_args[i]
                               ); CHECK(status);
    }
    status = clSetKernelArg(
  			kernel[2],
  			num_Cargs,
  			sizeof(cl_mem),
  			(void*)&d_matrix_mul_outputC
                        ); CHECK(status);
    // This is for the shared memory
    // status = clSetKernelArg(
  		// 	kernel[2],
  		// 	num_Cargs + 1,
  		// 	sizeof(cl_mem),
  		// 	(void*)NULL
    //                     ); CHECK(status);


    //----------------------------------------------
	// Configure the work-item structure (using only tasks atm)
	//----------------------------------------------

	// Define the number of threads that will be created
	// as well as the number of work groups
	size_t globalWorkSize[1];
	size_t localWorkSize[1];


	//----------------------------------------------
	// Enqueue the kernel for execution
	//----------------------------------------------


    // all kernels are always tasks
	globalWorkSize[0] = 1;
	localWorkSize[0]  = 1;

    cl_event kernel_exec_event[NUM_KERNELS_TO_CREATE];

    DPRINTF("\n===== Host-CPU enqeuing the OpenCL kernels to the FPGA device ======\n\n");
	for(i=0; i<NUM_KERNELS_TO_CREATE; i++) {
    // Alternatively, can use clEnqueueTaskKernel
    DPRINTF("clEnqueueNDRangeKernel[%d]: %s!\n", i,kernel_name[i]);
    status = clEnqueueNDRangeKernel(
				cmdQueue[i],
				kernel[i],
				1,
				NULL,
				globalWorkSize,
				localWorkSize,
				0,
				NULL,
                                &kernel_exec_event[i]
                         );
		CHECK(status);
	}
    DPRINTF(" *** FPGA execution started!\n");

	for(i=0; i < NUM_KERNELS_TO_CREATE ; i++) {
		status = clFlush(cmdQueue[i]);
                CHECK(status);
	}

    for(i=0; i < NUM_QUEUES_TO_CREATE; i++) {
        DPRINTF("cmd queue: %d\n", i);
        fflush(stdout);
        status = clFinish(cmdQueue[i]); CHECK(status);
	}
    DPRINTF(" *** FPGA execution finished!\n");
    DPRINTF("\n\n");

    double k_start_time[NUM_KERNELS_TO_CREATE];
    double k_end_time[NUM_KERNELS_TO_CREATE];
    double k_exec_time[NUM_KERNELS_TO_CREATE];
    double max_time = 0;
	for (i=0; i < NUM_KERNELS_TO_CREATE; i++) {
        k_exec_time[i] = compute_kernel_execution_time(kernel_exec_event[i], k_start_time[i], k_end_time[i]);
        if (k_exec_time[i] > max_time) {
                max_time = k_exec_time[i];
        }
    }
    DPRINTF("Time taken: %lf sec\n\n", max_time);

    DPRINTF("\n===== Host-CPU transferring result matrix C from the FPGA device global memory (DDR4) via PCIe ======\n\n");

    // Read the results back from the device, blocking read
    clEnqueueReadBuffer(
            //cmdQueue[KID_DRAIN_MAT_C],
            cmdQueue[NUM_KERNELS_TO_CREATE], // using a special queue for reading buffer C
            d_matrix_mul_outputC,
            CL_TRUE,
            0,
            num_elem_C*sizeof(cl_float),
            C,
            0,
            NULL,
            NULL); CHECK(status);
   float c = 0;
   int passed = 1;

  index = 0;
  for (int b = 0; b < BATCH; b++) {
        for (int y = 0; y < TILE_Y; y++) {
            for (int x = 0; x < TILE_X; x++) {
                for (int no = 0; no < TILE_NO; no++) {
                for (int yy = 0; yy < POY; yy++) {
                    for (int xx = 0; xx < POX; xx++) {
                        for (int nn = 0; nn < POF; nn++){
                            float c = 0;
                            for (int ky = 0; ky < KY; ky++) {
                                for (int kx = 0; kx < KX; kx++) {
                                    for (int ni = 0; ni < NIF; ni++) {
                                        c += input[b][y*POY + yy + ky][x*POX + xx + kx][ni] * weight[no*POF + nn][ky][kx][ni];
                                    }
                                }
                            }
                            if (c != result[index]) {
                                passed = 0;
                                printf("\n[FAILED]: b:%d y:%d x:%d no:%d result[b][y][x][no]:%f, right answer: %f\n", b, y*POY + yy, x*POX + xx, no*POF + nn, result[index], c);
                            }
                            index++;
                        }
                    }
                }
                }
            }
        }
    }

   if (passed) {
     printf("[PASSED]\n");
   }

}
