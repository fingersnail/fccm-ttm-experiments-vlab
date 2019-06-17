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


#define CHECK(status)                                                           \
        if (status != CL_SUCCESS)                                               \
{                                                                       \
        printf("error %d in line %d.\n", status, __LINE__);    \
        exit(1);                                                        \
}  

const int num_Aargs = 3; 
const int A_args[3] = {K,J,I};
const int num_Bargs = 3;
const int B_args[3] = {K,J,I};
const int num_Cargs = 3;
const int C_args[3] = {K,J,I};

const char* kernel_name[] = {
"kernel_A_loader_channel",
"kernel_B_loader_channel",
"kernel_C_unloader"
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

float A_mat[I*II*III][J*JJ*JJJ][L*LL*LLL];
float B_mat[L*LL*LLL][K*KK*KKK];

float C_ref[I*II*III*J*JJ*JJJ*K*KK*KKK];

float A_serializer[I*II*III*J*JJ*JJJ*L*LL*LLL];
float B_serializer[L*LL*LLL*K*KK*KKK];
float C_deserializer[I*II*III*J*JJ*JJJ*K*KK*KKK];

int main() {
  
  float* A;
  float* B;
  float* C;

  A = &A_serializer[0];
  B = &B_serializer[0];
  C = &C_deserializer[0];

  int i;
  int num_elem_A = (I*II*III)*(J*JJ*JJJ)*(L*LL*LLL);
  int num_elem_B = (L*LL*LLL)*(K*KK*KKK);
  int num_elem_C = (I*II*III)*(J*JJ*JJJ)*(K*KK*KKK);

  for (int i = 0; i < I*II*III; i++) {
    for (int j = 0; j < J*JJ*JJJ; j++) {
      for (int l = 0; l < L*LL*LLL; l++) {
        A_mat[i][j][l] = (i*(J*JJ*JJJ*L*LL*LLL)+j*(L*LL*LLL)+l)/((float)(I*II*III*J*JJ*JJJ*L*LL*LLL));
      }
    } 
  }

  int addr = 0;
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      for (int l= 0; l < L; l++)
        for (int ii = 0; ii < II; ii++)
            for (int ll = 0; ll < LL; ll++)
              for (int iii = 0; iii < III; iii++)
               for (int jjj = 0; jjj < JJJ; jjj++)
                for (int jj = 0; jj < JJ; jj++)
                  for (int lll = 0; lll < LLL; lll++) {
                    int _i = i*II*III + ii*III + iii;
                    int _j = j*JJ*JJJ + jj*JJJ + jjj;
                    int _l = l*LL*LLL + ll*LLL + lll;
		    A_serializer[addr] = A_mat[_i][_j][_l];
		    addr++;
                  }


  for (int l = 0; l < L*LL*LLL; l++) {
    for (int k = 0; k < K*KK*KKK; k++) {
      B_mat[l][k] = (l*(K*KK*KKK)+k)/((float)(L*LL*LLL*K*KK*KKK));
    } 
  }

  addr = 0;
  for (int k = 0; k < K; k++)
    for (int l = 0; l < L; l++)
        for (int kk = 0; kk < KK; kk++)
          for (int ll = 0; ll < LL; ll++)
              for (int kkk = 0; kkk < KKK; kkk++)
                for (int lll = 0; lll < LLL; lll++) {
                    int _k = k*KK*KKK + kk*KKK + kkk;
                    int _l = l*LL*LLL + ll*LLL + lll;
		    B_serializer[addr] = B_mat[_l][_k];
		    addr++;
                  }



  for (int x = 0; x < I*II*III*J*JJ*JJJ*K*KK*KKK; x++) {
    C_deserializer[x] = 0;
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
//  platforms = (cl_platform_id*) acl_aligned_malloc (numplatforms * sizeof(cl_platform_id));
  platforms = (cl_platform_id*) malloc (numPlatforms * sizeof(cl_platform_id));

  DPRINTF("Allocated space for Platform\n");
  
  // Fill in platforms with clGetPlatformIDs()
  status = clGetPlatformIDs(numPlatforms, platforms, NULL); CHECK(status);
  DPRINTF("Filled in platforms\n");    
  
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
				devices = (cl_device_id*)
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
	for (int i = 0; i < numDevices; i++) {
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

	cl_command_queue cmdQueue[NUM_QUEUES_TO_CREATE+1]; // extra queue for reading buffer D
        
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

  for (int i = 0; i < num_Aargs; i++) {
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
  status = clSetKernelArg(
  			kernel[0],
  			num_Aargs+1,
  			sizeof(cl_mem),
  			NULL
                        ); CHECK(status);
  
// B
  for (int i = 0; i < num_Bargs; i++) {
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
  status = clSetKernelArg(
  			kernel[1],
  			num_Bargs+1,
  			sizeof(cl_mem),
  			NULL
                        ); CHECK(status);

 // D
  for (int i = 0; i < num_Cargs; i++) {
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
  status = clSetKernelArg(
  			kernel[2],
  			num_Cargs+1,
  			sizeof(cl_mem),
  			NULL
                        ); CHECK(status);
 

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
        
        printf("\n===== Reporting measured throughput ======\n\n");
        double k_earliest_start_time = k_start_time[0];
        double k_latest_end_time     = k_end_time[0];

        for (i=1; i< NUM_KERNELS_TO_CREATE; i++) {
         if (k_start_time[i] < k_earliest_start_time)
             k_earliest_start_time   = k_start_time[i];

         if (k_end_time[i]   > k_latest_end_time)
             k_latest_end_time = k_end_time[i];
        }

        // IMPORTANT: we care about the finish time of drain_C, once data is drained we are done
        k_latest_end_time           = k_end_time[NUM_KERNELS_TO_CREATE-1];

        for(i=0; i< NUM_KERNELS_TO_CREATE; i++) {
          printf("  Kernel execution time on FPGA: %s, \n   \t\t\t\t\t\t\t\t\texec time = %.5f s, start=%.5f s, end=%.5f s\n", kernel_name[i], k_exec_time[i], k_start_time[i], k_end_time[i]);
        }
    
        double k_overall_exec_time = k_latest_end_time - k_earliest_start_time;
    
        printf("\n");
        printf("  Loader kernels start time\t\t= %.5f s\n", k_earliest_start_time);
        printf("  Drainer kernels end time\t\t= %.5f s\n", k_latest_end_time);
        printf("  FPGA MatMult exec time\t\t= %.5f s\n", k_overall_exec_time);
    
            // multiplied by 1.0e-9 to get G-FLOPs
        printf("\n");
    
        double num_operations = (double)2.0 * (K*KK*KKK) * (double)(I*II*III) * (double)(J*JJ*JJJ) * (double)(L*LL*LLL);
    
        printf("  # operations = %.0f\n", num_operations );
        printf("  Throughput: %.5f GFLOPS\n", (double)1.0e-9 * num_operations / k_overall_exec_time);

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

   	int passed = 1;


   addr = 0;
   for (int i = 0; i < I; i++)
     for (int j = 0; j < J; j++)
       for (int k = 0; k < K; k++)
         for (int ii = 0; ii < II; ii++)
             for (int iii = 0; iii < III; iii++)
               for (int jjj = 0; jjj < JJJ; jjj++)
                 for (int kkk = 0; kkk < KKK; kkk++)
                   for (int jj = 0; jj < JJ; jj++)
                     for (int kk = 0; kk < KK; kk++) {
                       int _i = i*II*III + ii*III + iii;
                       int _j = j*JJ*JJJ + jj*JJJ + jjj;
                       int _k = k*KK*KKK + kk*KKK + kkk;
                       float c = 0;
                       for (int _l = 0; _l < L*LL*LLL; _l++)
                            c += A_mat[_i][_j][_l] * B_mat[_l][_k];
                       C_ref[addr] = c;
                       addr++;
                     }
 
   

   float c;
   for (int x = 0; x < I*II*III*J*JJ*JJJ*K*KK*KKK; x++) {
       c = C_ref[x];
       if ( !( fabs(c-C_deserializer[x]) <= (0.005*fabs(c)))) {
         passed = 0;
         printf("\n[FAILED]: x:%d, ref:%f got:%f\n", x, c, C_deserializer[x]);
       }
       else {
	 printf("\nWorks: x:%d, ref:%f got:%f\n", x, c, C_deserializer[x]);
       }
   }

   if (passed) {
     printf("[PASSED]\n");
   }
}
