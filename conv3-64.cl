/*OpenCL C*/
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include "flat_linebuffer.h"

#define DPRINTF(...) printf(__VA_ARGS__); for ( int f = 0; f < 64*1024; f++ ) printf("");

#define __address_space___shared __local

__kernel void input_serializer_on_chip(__global const FLOAT_VEC * restrict input) {
	const int TOTAL_SIZE = BATCH * (NOX / POX) * (POX + KX - 1) * (NOY / POY) * (POY + KY - 1);
	for (int i = 0; i < TOTAL_SIZE; i++) {
	}
}



channel FLOAT_VEC input_forwarding[POY][POX][POF] __attribute__((depth(200)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY, POX)))
__kernel void input_feeder() {
	int yy = get_compute_id(0);
	int xx = get_compute_id(1);

	FLOAT_VEC input_feeder_ibuffer[KX * KY];
	int ky_kx = 0;

	while (1) {
		write_channel_intel(input_forwarding[yy][xx][0], input_feeder_ibuffer[ky_kx]);
		ky_kx++;
		if (ky_kx == KX * KY) {
			ky_kx = 0;
		}
	}
}


channel FLOAT_VEC weight_scattering[POF] __attribute__((depth(200)));

__kernel void weight_loader(__global const FLOAT_VEC * restrict weight) {
	const int TOTAL1 = BATCH * (NOY / POY) * (NOX / POX);
	const int BUFFER_SIZE = NOF * KX * KY;

	FLOAT_VEC weight_buffer[BUFFER_SIZE];

	for (int i = 0; i < BUFFER_SIZE; i++)
		weight_buffer[i] = *(weight+i);

	const int FLATTEN_SIZE = NOF * 16;   // (KX * KY) -> 16
	const int JUMP_POS = NOF * KX * KY - 1;
	const int JUMP_LENGTH = FLATTEN_SIZE - JUMP_POS;
	for (int i_j = 0; i_j < TOTAL1 * FLATTEN_SIZE; ) {  
	}
}


channel FLOAT_VEC weight_forwarding[POF][POY*POX] __attribute__((depth(100)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POF)))
__kernel void weight_feeder() {
	int nn = get_compute_id(0);

	FLOAT_VEC buffer[2];

	while(1) {
		write_channel_intel(weight_forwarding[nn][0], buffer[0]);
		write_channel_intel(weight_forwarding[nn][0], buffer[1]);
	} 
}



channel float result_channel __attribute__((depth(160))) ;

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY, POX, POF)))
__kernel void convolution() {
	const int yy = get_compute_id(0);
	const int xx = get_compute_id(1);
	const int nn = get_compute_id(2);
	const int input_forward_channel = nn + 1;
	const int weight_channel = yy * POX + xx;
	const int weight_forward_channel = weight_channel + 1;

	//DPRINTF("Begin calculation: %d %d %d\n", xx, yy, nn);

    int j = 16 - KX * KY;
    FLOAT_VEC _1;
    FLOAT_VEC _2;
    float _3 = 0;
    bool read_success_1 = (bool)(0);
 	bool read_success_2 = (bool)(0);
	while (1) {
		if (!read_success_1)
			_1 = read_channel_nb_intel(input_forwarding[yy][xx][nn], &read_success_1);

		if (!read_success_2)
			_2 = read_channel_nb_intel(weight_forwarding[nn][weight_channel], &read_success_2);

		// DPRINTF("Read: %f %f\n", _1, _2);

		if (read_success_1 && read_success_2) {
			if (input_forward_channel < POF)
				write_channel_intel(input_forwarding[yy][xx][input_forward_channel], _1);
			if (weight_forward_channel < POX * POY)
				write_channel_intel(weight_forwarding[nn][weight_forward_channel], _2);

			#pragma unroll
			for (int k = 0; k < NIF; k++) {
				_3 += _1[k]*_2[k];
 			}
			j++;

			read_success_1 = (bool)(0);
 			read_success_2 = (bool)(0);
		}

		if (j == 16) {
			if (yy == POY - 1 && xx == POX - 1 && nn == POF - 1)
				write_channel_intel(result_channel, _3);
			_3 = 0;
			j = 16 - KX * KY;
			// DPRINTF("The result is: %d %d %d %f\n", xx, yy, nn, _3);
		}
	}

	//DPRINTF("Calculation finished: %d %d %d\n", xx, yy, nn);
}

__kernel void result_unloader(__global float * restrict output) {
	int TOTAL = BATCH * NOY * NOX * NOF / (POF * POX * POY);
	for (int i = 0; i < TOTAL; i++) {
		float in = read_channel_intel(result_channel);
		output[i] = in;
		if (i % 32 == 0)
			DPRINTF("%d %f\n", i, in)
	}
}
