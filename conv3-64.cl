/*OpenCL C*/
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include "flat_linebuffer.h"

#define DPRINTF(...) printf(__VA_ARGS__); for ( int f = 0; f < 64*1024; f++ ) printf("");

#define __address_space___shared __local

channel FLOAT_VEC linebuffer_channel_first __attribute__((depth(300)));
channel FLOAT_VEC linebuffer_channel[3] __attribute__((depth(3)));

__kernel void input_serializer_on_chip(__global const FLOAT_VEC * restrict input) {
	const int TOTAL_SIZE = BATCH * 32 * 6 * 32 * 6;
	for (int i = 0; i < TOTAL_SIZE; i++) {
		write_channel_intel(linebuffer_channel_first, input[i]);
	}
}

channel FLOAT_VEC input_loader_to_feeder[4][4] __attribute__((depth(100)));
__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__kernel void input_loader_first() {
	while(1) {
		FLOAT_VEC linebuffer[4];
		for (int s_0 = 0; s_0 < 3; s_0++) { 
        	for (int s_1_i = 0; s_1_i < 3 * 32; s_1_i++) {
        		int i = s_1_i & 31;
        		int s_1 = s_1_i >> 5;

        		if ((s_0|s_1) == 0 && i < 22 || s_0 && s_1 == 0 && i < 4 || i == 0) {
        			write_channel_intel(linebuffer_channel[2], linebuffer[0]);
       				#pragma unroll
					for (int j = 0; j < 3; j++) {
  						linebuffer[j] = linebuffer[j + 1];
					}
					linebuffer[3] = read_channel_intel(linebuffer_channel_first);
        		}
        		// First read, then write
       			if (i >= 22) {
       				int pos = i - 22;
       				write_channel_intel(input_loader_to_feeder[3][0], linebuffer[pos]); 
	   			}
        		
        		if (i == 25) {
        			s_1_i += 6;
        		}
        	}
        }
	}
}

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(3)))
__kernel void input_loader() {
	int yy = get_compute_id(0);
   	int initial_size =  16 + 2 * yy;

	while(1) {
		FLOAT_VEC linebuffer[4];
		for (int s_0 = 0; s_0 < 3; s_0++) { 
        	for (int s_1_i = 0; s_1_i < 3 * 32;) {
        		int i = s_1_i & 31;
        		int s_1 = s_1_i >> 5;

        		if ((s_0|s_1) == 0 && i < initial_size || s_0 && s_1 == 0 && i < 4 || i == 0) {
        			if (yy)
        				write_channel_intel(linebuffer_channel[yy-1], linebuffer[0]);
       				#pragma unroll
					for (int j = 0; j < 3; j++) {
  						linebuffer[j] = linebuffer[j + 1];
					}
					linebuffer[3] = read_channel_intel(linebuffer_channel[yy]);
        		}
        		// First read, then write
       			if (i >= 22) {
       				int pos = i - 22;
       				write_channel_intel(input_loader_to_feeder[yy][0], linebuffer[pos]); 
	   			}
        		s_1_i += (i == 25) ? 7 : 1;
        	}
        }
        initial_size = 22;
	}
}


typedef struct _kernel_vec {FLOAT_VEC data[3 * 3]; } kernel_vec;
channel FLOAT_VEC input_forwarding[4][4][POF] __attribute__((depth(200)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(4, 4)))
__kernel void input_feeder() {
	int yy = get_compute_id(0);
	int xx = get_compute_id(1);
	int TILE = NOF / POF;

	FLOAT_VEC input_feeder_ibuffer[9];
	int input_scatter_channel = xx + 1;
	int window_size = 4 - xx; // 4 3 2 1


	int ky_kx = 0;
	kernel_vec kernel;
	bool write_success = (bool)(0);
 	bool read_success = (bool)(0);
	int scatter_loop = 0;
	int feeder_loop = 0;
	int forward_loop = 0;

	while (1) {
		/*
		if (scatter_loop < weight_size && ky_kx < 9) {
			input = read_channel_nb_intel(input_loader_to_feeder[yy][xx], &read_success);
			if ((scatter_loop < 1) && read_success) {
				input_feeder_ibuffer[ky_kx] = input;
				ky_kx++;
       			scatter_loop++;
			} else if (read_success) {
				write_channel_intel(input_loader_to_feeder[yy][input_scatter_channel], input);
				scatter_loop++;
			}
		}

		if (scatter_loop == weight_size && forward_loop < TILE * 9) {
			write_success = write_channel_nb_intel(input_forwarding[yy][xx][0], input_feeder_ibuffer[ky_kx]);
			if (write_success) {
				forward_loop++;
				ky_kx++;
				if (ky_kx == 9) {
					ky_kx = 0;
				}
			}
		}

		if (forward_loop == TILE) {
			ky_kx = 0;
			scatter_loop = 0;
			forward_loop = 0;
		}

		for (int ky_kx = 0; ky_kx < KY * KX; ky_kx++) {
			for (int n_time = 0; n_time < window_size; n_time++) {
				FLOAT_VEC input = read_channel_intel(input_loader_to_feeder[yy][xx]);
				if (n_time)
					write_channel_intel(input_loader_to_feeder[yy][input_scatter_channel], input);
				else
					input_feeder_ibuffer[ky_kx] = input;
			}
		}

		for (int no = 0; no < TILE2; no++) {
			for (int ky_kx = 0; ky_kx < KY * KX; ky_kx++) {
				write_channel_intel(input_forwarding[yy][xx][0], input_feeder_ibuffer[ky_kx]);
			}
		}
		*/

		for (int no_ky_kx_t = 0; no_ky_kx_t < TILE * 4 * 4 * 4;) {
			int ky_kx_t = no_ky_kx_t & 63;
			int t = ky_kx_t & 3;
			int ky_kx = ky_kx_t >> 2;
			
			if (no_ky_kx_t < 64) { // if(!(no_ky_kx_t)>>6)
				FLOAT_VEC input = read_channel_intel(input_loader_to_feeder[yy][xx]);
				if (t && input_scatter_channel < 4)
					write_channel_intel(input_loader_to_feeder[yy][input_scatter_channel], input);
				else
					input_feeder_ibuffer[ky_kx] = input;
			}
					
			if (t == window_size - 1) {
				write_channel_intel(input_forwarding[yy][xx][0], input_feeder_ibuffer[ky_kx]);
			}

			no_ky_kx_t += (t == window_size-1) ? ((ky_kx_t == 35 - xx) ? 29 + xx :  xx + 1) : ((ky_kx_t == 35-xx) ? 29 : 1);
		}
	}
}


channel FLOAT_VEC weight_scattering[POF] __attribute__((depth(200)));

__kernel void weight_loader(__global const FLOAT_VEC * restrict weight) {
	const int TOTAL1 = BATCH * 32 * 32;

	FLOAT_VEC weight_buffer[64 * 3 * 3];

	for (int i = 0; i < 64 * 3 * 3; i++)
		weight_buffer[i] = *(weight+i);

	for (int i_j = 0; i_j < TOTAL1 * 64 * 16; i_j++) {
		int j = i_j & 1023;
		write_channel_intel(weight_scattering[0], weight_buffer[j]);
		if (j == 575) {
			i_j += 448;
		}
	}
}


channel FLOAT_VEC weight_forwarding[POF][4*4] __attribute__((depth(100)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POF)))
__kernel void weight_feeder() {
	int nn = get_compute_id(0);

	FLOAT_VEC weight = 0;
	int weight_size = POF; //- nn;
	int weight_scattering_channel = nn+1;

	const int TOTAL2 = 3 * 3;

	FLOAT_VEC buffer[2];
	int w = 0;
	int r = 1;
	int first = 1;
	bool write_success = (bool)(0);
 	bool read_success = (bool)(0);
	int scatter_loop = nn;
	int feeder_loop = 0;

	while(1) {
		if (scatter_loop < POF) {
			weight = read_channel_nb_intel(weight_scattering[nn], &read_success);
			if ((scatter_loop < nn + 1) && read_success) {
				buffer[w] = weight;
       			scatter_loop++;
			} else if (read_success) {
				write_channel_intel(weight_scattering[weight_scattering_channel], weight);
				scatter_loop++;
			}
		}

		if (feeder_loop < 1) {
     		if (!(first)) {
      			write_success = write_channel_nb_intel(weight_forwarding[nn][0], buffer[r]);
     		}
     		if (write_success || first)
       			feeder_loop++;
    	}

    	if ((scatter_loop == POF) && (feeder_loop == 1)) {
    		first = (bool)(0);
    		w = !((bool)(w));
    		r = !((bool)(r));
    		scatter_loop = nn;
    		feeder_loop = 0;
    	}

    	/*
    	for (int n_time = 0; n_time < weight_size; n_time++) {
			weight = read_channel_intel(weight_scattering[nn]);
			if (n_time) {
				write_channel_intel(weight_scattering[weight_scattering_channel], weight);
			} else {
				write_channel_intel(weight_forwarding[nn][0], weight);
			}
		}
		*/
	}   
}



channel float conv_to_drainer_channel[4][4][POF] __attribute__((depth(100)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(4, 4, POF)))
__kernel void convolution() {
	int yy = get_compute_id(0);
	int xx = get_compute_id(1);
	int nn = get_compute_id(2);

	int input_forward_channel = nn + 1;
	bool do_input_forward = true;
	if (nn == POF - 1){
		do_input_forward = false;
	}

	int weight_channel = yy * 4 + xx;
	int weight_forward_channel = weight_channel + 1;
	bool do_weight_forward = true;
	if (weight_forward_channel == 4 * 4){
		do_weight_forward = false;
	}

	//DPRINTF("Begin calculation: %d %d %d\n", xx, yy, nn);

    // float4
    int j = 7;
    float _3 = 0;
	while (1) {
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
		j++;

		if (j == 16) {
			write_channel_intel(conv_to_drainer_channel[yy][xx][nn], _3);
			_3 = 0;
			j = 0;
			// DPRINTF("The result is: %d %d %d %f\n", xx, yy, nn, _3);
		}
	}

	//DPRINTF("Calculation finished: %d %d %d\n", xx, yy, nn);
}


channel float conv_to_result_consumer[4][4][POF] __attribute__((depth(100)));
channel float conv_to_result_collector[POF][4*4] __attribute__((depth(100)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(4, 4, POF)))
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


typedef struct _out_vec { float data[POF]; } outvec;

channel outvec C_collector_0_inter_channel __attribute__((depth(160))) ;
channel outvec C_collector_0_channel[POF - 1] __attribute__((depth(100))) ;

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POF)))
__kernel void result_collector() {
	int nn = get_compute_id(0);

	int result_size = POF - nn;
	int result_gathering_channel = nn + 1;
	int collector_channel = 15;

	float result;

	while(1) {
		result = read_channel_intel(conv_to_result_collector[nn][collector_channel]);

		outvec in_data;
   		if (nn != (POF - 1)) {
			in_data = read_channel_intel(C_collector_0_channel[nn]);
   		}

   		outvec out;
   		#pragma unroll
   		for (int x = (POF - 1); x > nn; x--)
     		out.data[x] = in_data.data[x];
   		out.data[nn] = result;

   		if (nn == 0)
     		write_channel_intel(C_collector_0_inter_channel, out);
   		else
     		write_channel_intel(C_collector_0_channel[nn-1], out);

     	/*
     	for (int n_time = 0; n_time < result_size; n_time++) {
			if (n_time) {
				result = read_channel_intel(collector_to_consumer[result_gathering_channel]);
			} else {
				result = read_channel_intel(conv_to_result_collector[nn][collector_channel]);
			}
			write_channel_intel(collector_to_consumer[nn], result);
		}
		*/
	}
}


__kernel void result_unloader(__global outvec * restrict output) {
	int TOTAL = BATCH * 128 * 128 * 64 / POF;
	for (int i = 0; i < TOTAL; i++) {
		outvec in;
		in = read_channel_intel(C_collector_0_inter_channel);
		output[i] = in;
		//if (i % 64 == 0) {
			// DPRINTF("The result is: %d %f\n", i / 64, in.data[0]);
		// }
	}
}
