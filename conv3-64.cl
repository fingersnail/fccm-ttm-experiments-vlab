/*OpenCL C*/
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include "flat_linebuffer.h"

#define DPRINTF(...) printf(__VA_ARGS__); for ( int f = 0; f < 64*1024; f++ ) printf("");

#define __address_space___shared __local


channel FLOAT_VEC linebuffer_channel[4] __attribute__((depth(3)));

__kernel void input_serializer_on_chip(__global const FLOAT_VEC * restrict input) {
	const int TOTAL_SIZE = BATCH * 32* 6 * 32* 6;
	for (int i = 0; i < TOTAL_SIZE; i++) {
		write_channel_intel(linebuffer_channel[3], input[i]);
	}
}

channel FLOAT_VEC input_loader_to_feeder[4][4] __attribute__((depth(2)));
__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(4)))
__kernel void input_loader() {
	int yy = get_compute_id(0);
   	const int LINEBUFFER_EXTENT = 22;
   	int initial_size =  16 + yy;

	while(1) {
		FLOAT_VEC linebuffer[4];
		
		for (int s_0 = 0; s_0 < 3; s_0++) {
			for (int s_1_i = 0; s_1_i < 3 * 32; s_1_i++) {
        		int s_1 = s_1_i >> 5;
        		int i = s_1_i & 31;

        		if (!s_0 && i < initial_size || s_0 && !s_1 && i < 4 || !i) {
        			if (yy)
        				write_channel_intel(linebuffer_channel[yy-1], linebuffer[0]);
       				#pragma unroll
					for (int j = 0; j < 4 - 1; j++) {
  						linebuffer[j] = linebuffer[j + 1];
					}
					linebuffer[3] = read_channel_intel(linebuffer_channel[yy]);
        		}
        		// First read, then write
       			if (i >= 22) {
       				write_channel_intel(input_loader_to_feeder[yy][0], linebuffer[i - 22]); 
       			}
       			if (i == 25) {
       				s_1_i += 6;
       			}
        	}
        }
        initial_size = 22;
	}
}


channel FLOAT_VEC input_forwarding[4][4][8] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(4, 4)))
__kernel void input_feeder() {
	int yy = get_compute_id(0);
	int xx = get_compute_id(1);

	FLOAT_VEC _input_feeder_ibuffer[3*3];
	int input_scatter_channel = xx + 1;
	int window_size = 4 - xx; //4 3 2 1

	while (1) {
		for (int no_ky_kx_t = 0; no_ky_kx_t < 8 * 16 * 4; no_ky_kx_t++) {
			int ky_kx_t = no_ky_kx_t & 63;
			int t = ky_kx_t & 3;
			if (t < window_size) {
				int ky_kx = ky_kx_t >> 2;
				if (no_ky_kx_t < 36) {
					FLOAT_VEC input = read_channel_intel(input_loader_to_feeder[yy][xx]);
					if (t)
						write_channel_intel(input_loader_to_feeder[yy][input_scatter_channel], input);
					else
						_input_feeder_ibuffer[ky_kx] = input;
				}
				if (t == window_size - 1)
					write_channel_intel(input_forwarding[yy][xx][0], _input_feeder_ibuffer[ky_kx]);
			} else {
				ky_kx_t += (xx - 1);
			}
			if (ky_kx_t == 35)
				no_ky_kx_t += 28;
		}
	}
}


channel FLOAT_VEC weight_scattering[8] __attribute__((depth(2)));

__kernel void weight_loader(__global const FLOAT_VEC * restrict weight) {
	const int TOTAL1 = BATCH * 32 * 32;
	const int TOTAL2 = 64 * 3 * 3;

	FLOAT_VEC weight_buffer[TOTAL2];

	for (int i = 0; i < TOTAL1; i++) {
		// I can insert a buffer at here
		for (int j = 0; j < TOTAL2; j++) {
			if (i == 0) {
				weight_buffer[j] = *(weight+j);
			}
			write_channel_intel(weight_scattering[0], weight_buffer[j]);
		}
	}
}


channel FLOAT_VEC weight_forwarding[8][4*4] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(8)))
__kernel void weight_feeder() {
	int nn = get_compute_id(0);

	FLOAT_VEC weight = 0;
	int weight_size = 8 - nn;
	int weight_scattering_channel = nn+1;

	const int TOTAL2 = 3 * 3;

	while(1) {
		for (int n_time = 0; n_time < weight_size; n_time++) {
			weight = read_channel_intel(weight_scattering[nn]);
			if (n_time) {
				write_channel_intel(weight_scattering[weight_scattering_channel], weight);
			} else {
				write_channel_intel(weight_forwarding[nn][0], weight);
			}
		}
	}   
}



channel float conv_to_drainer_channel[4][4][8] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(4, 4, 8)))
__kernel void convolution() {
	int yy = get_compute_id(0);
	int xx = get_compute_id(1);
	int nn = get_compute_id(2);

	int input_forward_channel = nn + 1;
	bool do_input_forward = true;
	if (nn == 7){
		do_input_forward = false;
	}

	int weight_channel = yy * 4 + xx;
	int weight_forward_channel = weight_channel + 1;
	bool do_weight_forward = true;
	if (weight_forward_channel == 16){
		do_weight_forward = false;
	}

	//DPRINTF("Begin calculation: %d %d %d\n", xx, yy, nn);
	int TOTAL2 = 3 * 3;

    // float4
	while (1) {
		float _3 = 0;
		for (int j = 0; j < TOTAL2; j++) {
			FLOAT_VEC _1 = read_channel_intel(input_forwarding[yy][xx][nn]);
			if (do_input_forward) {
				write_channel_intel(input_forwarding[yy][xx][input_forward_channel], _1);
			}

			FLOAT_VEC _2 = read_channel_intel(weight_forwarding[nn][weight_channel]);
			if (do_weight_forward) {
				write_channel_intel(weight_forwarding[nn][weight_forward_channel], _2);
			}
			// DPRINTF("Read: %f %f\n", _1, _2);
			#pragma unroll
			for (int k = 0; k < NIF; k++)
				_3 += _1[k]*_2[k];
		}
		write_channel_intel(conv_to_drainer_channel[yy][xx][nn], _3);
		// DPRINTF("The result is: %f\n", _3);
	}

	//DPRINTF("Calculation finished: %d %d %d\n", xx, yy, nn);
}


channel float conv_to_result_consumer[4][4][8] __attribute__((depth(2)));
channel float conv_to_result_collector[8][4*4] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(4, 4, 8)))
__kernel void result_consumer() {
	int yy = get_compute_id(0);
	int xx = get_compute_id(1);
	int nn = get_compute_id(2);

	int result_size = yy * 4 + xx;
	int result_write_channel = yy * 4 + xx;
	int result_read_channel = yy * 4 + xx - 1;

	float result;
	while (1) {
		for (int n_time = result_size; n_time >= 0; n_time--) {
			if (n_time) {
				result = read_channel_intel(conv_to_result_collector[nn][result_read_channel]);
			} else {
				result = read_channel_intel(conv_to_drainer_channel[yy][xx][nn]);
			}
			write_channel_intel(conv_to_result_collector[nn][result_write_channel], result);
		}
	}
}


channel float collector_to_consumer[8] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(8)))
__kernel void result_collector() {
	int nn = get_compute_id(0);

	int result_size = 8 - nn;
	int result_gathering_channel = nn+1;
	int collector_channel = 15;

	float result;

	while(1) {
		for (int n_time = 0; n_time < result_size; n_time++) {
			if (n_time) {
				result = read_channel_intel(collector_to_consumer[result_gathering_channel]);
			} else {
				result = read_channel_intel(conv_to_result_collector[nn][collector_channel]);
			}
			write_channel_intel(collector_to_consumer[nn], result);
		}
	}
}


__kernel void result_unloader(__global float * restrict output) {
	int TOTAL = BATCH * 128 * 128 * 64;
	for (int i = 0; i < TOTAL; i++) {
		*(output + i) = read_channel_intel(collector_to_consumer[0]);			   
	}
}
