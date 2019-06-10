/*OpenCL C*/
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include "flat_linebuffer.h"

#define DPRINTF(...) printf(__VA_ARGS__); for ( int f = 0; f < 64*1024; f++ ) printf("");

#define __address_space___shared __local


channel FLOAT_VEC linebuffer_channel[POY] __attribute__((depth(KX)));

__kernel void input_serializer(__global const FLOAT_VEC * restrict input) {
	const int TILE_X = NOX / POX;
	const int TILE_Y = NOY / POY;
	const int write_channel = POY - 1;
	const int TOTAL_SIZE = BATCH * TILE_Y*(POY + KY - 1) * TILE_X*(POX + KX - 1);
	for (int i = 0; i < TOTAL_SIZE; i++) {
		write_channel_intel(linebuffer_channel[write_channel], input[i]);
	}
}

channel FLOAT_VEC input_loader_to_feeder[POY][POX] __attribute__((depth(2)));
__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY)))
__kernel void input_loader() {
	int yy = get_compute_id(0);
	const int INPUT_EXTENT_0 = KX + POX - 1;
   	const int LINEBUFFER_EXTENT = POX + (POY - 1) * INPUT_EXTENT_0;

	while(1) {
		FLOAT_VEC linebuffer[POX];

		int step = LINEBUFFER_EXTENT;

		for (int s_1 = 0; s_1 < KY; s_1++) { 
        	for (int s_0 = 0; s_0 < KX; s_0++) {
        		for (int i = 0; i < step; i++) {
        			if (yy)
        				write_channel_intel(linebuffer_channel[yy-1], linebuffer[0]);
        			#pragma unroll
					for (int j = 0; j < POX - 1; j++) {
      					linebuffer[j] = linebuffer[j + 1];
					}
					linebuffer[POX - 1] = read_channel_intel(linebuffer_channel[yy]);
        		}
        		#pragma unroll
        		for (int dim_0 = 0; dim_0 < POX; dim_0++) {
        			write_channel_intel(input_loader_to_feeder[yy][0], linebuffer[dim_0]); 
        		}
        		step = 1;
        	}
        	step = POX;
        }
	}
}


channel FLOAT_VEC input_forwarding[POY][POX][POF] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY, POX)))
__kernel void input_feeder() {
	int TILE0 = NOY / POY;
	int TILE1 = NOX / POX;
	int TILE2 = NOF / POF;

	int yy = get_compute_id(0);
	int xx = get_compute_id(1);

	FLOAT_VEC _input_feeder_ibuffer[KY*KX];
	int input_scatter_channel = xx + 1;
	int window_size = POX - xx;

	while (1) {
		for (int no = 0; no < TILE2; no++) {
			for (int ky_kx = 0; ky_kx < KY * KX; ky_kx++) {
				for (int n_time = 0; n_time < window_size; n_time++) {
					if (!no) {
						FLOAT_VEC input = read_channel_intel(input_loader_to_feeder[yy][xx]);
						if (n_time)
							write_channel_intel(input_loader_to_feeder[yy][input_scatter_channel], input);
						else
							_input_feeder_ibuffer[ky_kx] = input;
					}
				}
				write_channel_intel(input_forwarding[yy][xx][0], _input_feeder_ibuffer[ky_kx]);
			}
		}
	}
}


channel FLOAT_VEC weight_scattering[POF] __attribute__((depth(2)));

__kernel void weight_loader(__global const FLOAT_VEC * restrict weight) {
	const int TILE0 = NOY / POY;
	const int TILE1 = NOX / POX;
	
	const int TOTAL1 = BATCH * TILE0 * TILE1;
	const int TOTAL2 = NOF * KY * KX;

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


channel FLOAT_VEC weight_forwarding[POF][POX*POY] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POF)))
__kernel void weight_feeder() {
	int nn = get_compute_id(0);

	FLOAT_VEC weight = 0;
	int weight_size = POF - nn;
	int weight_scattering_channel = nn+1;

	const int TOTAL2 = KY * KX;

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



channel float conv_to_drainer_channel[POY][POX][POF] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY, POX, POF)))
__kernel void convolution() {
	int TILE0 = NOY / POY;
	int TILE1 = NOX / POX;
	int TILE2 = NOF / POF;

	int yy = get_compute_id(0);
	int xx = get_compute_id(1);
	int nn = get_compute_id(2);

	int input_forward_channel = nn + 1;
	bool do_input_forward = true;
	if (nn == POF - 1){
		do_input_forward = false;
	}

	int weight_channel = yy * POX + xx;
	int weight_forward_channel = weight_channel + 1;
	bool do_weight_forward = true;
	if (weight_forward_channel == POY * POX){
		do_weight_forward = false;
	}

	//DPRINTF("Begin calculation: %d %d %d\n", xx, yy, nn);
	int TOTAL2 = KY * KX;

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


channel float conv_to_result_consumer[POY][POX][POF] __attribute__((depth(2)));
channel float conv_to_result_collector[POF][POX*POY] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY, POX, POF)))
__kernel void result_drainer() {
	int yy = get_compute_id(0);
	int xx = get_compute_id(1);
	int nn = get_compute_id(2);

	int result_size = yy * POX + xx;
	int result_write_channel = yy * POX + xx;
	int result_read_channel = yy * POX + xx - 1;

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


channel float collector_to_consumer[POF] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POF)))
__kernel void result_collector() {
	int nn = get_compute_id(0);

	int result_size = POF - nn;
	int result_gathering_channel = nn+1;
	int collector_channel = POX*POY - 1;

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


__kernel void result_consumer(__global float * restrict output) {
	int TOTAL = BATCH * NOY * NOX * NOF;
	for (int i = 0; i < TOTAL; i++) {
		*(output + i) = read_channel_intel(collector_to_consumer[0]);			   
	}
}
