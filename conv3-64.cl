/*OpenCL C*/
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include "flat_linebuffer.h"

#define DPRINTF(...) printf(__VA_ARGS__); for ( int f = 0; f < 64*1024; f++ ) printf("");

#define __address_space___shared __local

__kernel void input_serializer_on_chip(__global const FLOAT_VEC * restrict input) {
}


channel FLOAT_VEC input_forwarding[POY][POX][POF] __attribute__((depth(200)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY, POX)))
__kernel void input_feeder() {
    int yy = get_compute_id(0);
    int xx = get_compute_id(1);
    
    while (1) {
        FLOAT_VEC _1;
        write_channel_intel(input_forwarding[yy][xx][0], _1);
    }
}


channel FLOAT_VEC weight_scattering[POF] __attribute__((depth(200)));

__kernel void weight_loader(__global const FLOAT_VEC * restrict weight) {
}


channel FLOAT_VEC weight_forwarding[POF][POY*POX] __attribute__((depth(100)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POF)))
__kernel void weight_feeder() {
    int nn = get_compute_id(0);
    FLOAT_VEC weight;
    while(1) {
        write_channel_intel(weight_forwarding[nn][0], weight);
    }   
}



channel float conv_to_drainer_channel[POY][POX][POF] __attribute__((depth(100)));

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

    float buffer[2];
    int w = 0;
	int r = 1;
	bool first = (bool)(1);
    while (1) {
        if (j == 16) {
        	if (!first) {
            	write_channel_intel(conv_to_drainer_channel[yy][xx][nn], buffer[r]);
            	buffer[r] = 0;
        	}
        	w = !((bool)(w));
    		r = !((bool)(r));
            j = 16 - KX * KY;
            first = (bool)(0);

            // DPRINTF("The result is: %d %d %d %f\n", xx, yy, nn, _3);
        }

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

            // Two tile for NIF
            float sum = 0;
            #pragma unroll
            for (int k = 0; k < NIF; k++) {
                sum += _1[k]*_2[k];
            }
            #pragma unroll
	    for (int i=0; i < 2; i++) {
		buffer[i] += (w == i ? sum : 0);
	    }
             
            j++;

            read_success_1 = (bool)(0);
            read_success_2 = (bool)(0);
        }
    }

    //DPRINTF("Calculation finished: %d %d %d\n", xx, yy, nn);
}


channel float conv_to_result_collector[POF][POY*POX] __attribute__((depth(100)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY, POX, POF)))
__kernel void result_consumer() {
    int yy = get_compute_id(0);
    int xx = get_compute_id(1);
    int nn = get_compute_id(2);

    int result_size = yy * POX + xx;
    int result_write_channel = yy * POX + xx;
    int result_read_channel = yy * POX + xx - 1;

    float result;
    bool read_success = (bool)(0);
    int t = 0;
    while (1) { 
        if (t < result_size) {
            result = read_channel_nb_intel(conv_to_result_collector[nn][result_read_channel], &read_success);
            if (read_success)
                t++;
        } else {
            result = read_channel_nb_intel(conv_to_drainer_channel[yy][xx][nn], &read_success);
            if (read_success)
                t = 0;
        }
        if (read_success) {
            write_channel_intel(conv_to_result_collector[nn][result_write_channel], result);
            read_success = (bool)(0);
        }
          
        /*
        // More readable version
        for (int n_time = result_size; n_time >= 0; n_time--) {
            if (n_time) {
                result = read_channel_intel(conv_to_result_collector[nn][result_read_channel]);
            } else {
                result = read_channel_intel(conv_to_drainer_channel[yy][xx][nn]);
            }
            write_channel_intel(conv_to_result_collector[nn][result_write_channel], result);
        }*/
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
    int collector_channel = POX*POY - 1;

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
    }
}


__kernel void result_unloader(__global outvec * restrict output) {
    int TOTAL = BATCH * NOY * NOX * NOF / POF;
    for (int i = 0; i < TOTAL; i++) {
        outvec in;
        in = read_channel_intel(C_collector_0_inter_channel);
        output[i] = in;
    }
}
