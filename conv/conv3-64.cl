/*OpenCL C*/
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include "flat_linebuffer.h"

#define DPRINTF(...) printf(__VA_ARGS__); for ( int f = 0; f < 64*1024; f++ ) printf("");

#define __address_space___shared __local


channel float input_serializer_to_laoder __attribute__((depth(2)));

__kernel void input_serializer(__global float *input, 
                               __address_space___shared int16* __shared) {
    const int TILE_X = NOX / POX;
    const int TILE_Y = NOY / POY;

    const int TOTAL_SIZE = BATCH * TILE_Y*(POY + KY - 1) * TILE_X*(POX + KX - 1) * NIF;
    for (int i = 0; i < TOTAL_SIZE; i++) {
        write_channel_intel(input_serializer_to_laoder, input[i]);
    }
}

channel float input_loader_to_feeder[POY*POX] __attribute__((depth(2)));
__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__kernel void input_loader() {
    while(1) {
        //DPRINTF("Inserting linebuffer\n");
        define_unrolled_flat_linebuffer_2d(float, input_serializer_to_laoder, input_loader_to_feeder, POX, POY, KX, KY, NIF)
        //DPRINTF("Linebuffer finished\n");
    }
}


channel float input_forwarding[POY][POX][POF] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY, POX)))
__kernel void input_feeder() {
    int TILE0 = NOY / POY;
    int TILE1 = NOX / POX;
    int TILE2 = NOF / POF;

    int yy = get_compute_id(0);
    int xx = get_compute_id(1);

    float _input_feeder_ibuffer[KY][KX][NIF];
    int channel_num = yy * POX + xx;
    int input_scatter_channel = channel_num + 1;

    int window_size = POX*POY - channel_num;
    int read_num;
    float input;
    while (1) {
        for (int no = 0; no < TILE2; no++) {
            for (int ky = 0; ky < KY; ky++) {
                for (int kx = 0; kx < KX; kx++) {
                    for (int ni = 0; ni < NIF; ni++) {
                        if (!no) {
                            read_num = window_size;
                            for (int n_time = 0; n_time < window_size; n_time++) {
                                input = read_channel_intel(input_loader_to_feeder[channel_num]);
                                if (read_num == window_size) {
                                    _input_feeder_ibuffer[ky][kx][ni] = input;
                                    read_num = 1;
                                } else {
                                    write_channel_intel(input_loader_to_feeder[input_scatter_channel], input);
                                    read_num++;
                                }
                            }
                        }
                        write_channel_intel(input_forwarding[yy][xx][0], _input_feeder_ibuffer[ky][kx][ni]);
                    }
                }
            }
        }
    }
}


channel float weight_scattering[POF] __attribute__((depth(2)));

__kernel void weight_loader(__global float *weight,
                            __address_space___shared int16* __shared) {
    const int TILE0 = NOY / POY;
    const int TILE1 = NOX / POX;
    const int TILE2 = NOF / POF;
    
    const int TOTAL1 = BATCH * TILE0 * TILE1;
    const int TOTAL2 = TILE2 * KY * KX * NIF;

    float weight_buffer[TOTAL2 * POF];

    for (int i = 0; i < TOTAL1; i++) {
        // I can insert a buffer at here
        int index = 0;
        for (int j = 0; j < TOTAL2; j++) {
            for (int nn = 0; nn < POF; nn++) {
                // if (i == 0) {
                    // weight_buffer[index] = *(weight+index);
                // }
                write_channel_intel(weight_scattering[0], *(weight+index));
                index++;
            }
        }
    }
}


channel float weight_forwarding[POF][POX*POY] __attribute__((depth(2)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POF)))
__kernel void weight_feeder() {
    int nn = get_compute_id(0);

    float weight = 0;
    int weight_size = POF - nn;
    int weight_scattering_channel = nn+1;

    const int TOTAL2 = KY * KX * NIF;

    int read_num = weight_size;
    while(1) {
        for (int n_time = 0; n_time < POF - nn; n_time++) {
            for (int i = 0; i < TOTAL2; i++) {
                weight = read_channel_intel(weight_scattering[nn]);
                // DPRINTF("Read weight: %f\n", weight);
                if (read_num == weight_size) {
                    write_channel_intel(weight_forwarding[nn][0], weight);
                    read_num = 1;
                } else {
                    write_channel_intel(weight_scattering[weight_scattering_channel], weight);
                    read_num++;
                }
            }
        }
    }
}


channel float conv_to_result_consumer[POY][POX][POF] __attribute__((depth(2)));

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

    int TOTAL1 = BATCH * TILE0 * TILE1 * TILE2;
    int TOTAL2 = KY * KX * NIF;

    for (int i = 0; i < TOTAL1; i++) {
        float _3 = 0;
        for (int j = 0; j < TOTAL2; j++) {
            float _1 = read_channel_intel(input_forwarding[yy][xx][nn]);
            if (do_input_forward) {
                write_channel_intel(input_forwarding[yy][xx][input_forward_channel], _1);
            }

            float _2 = read_channel_intel(weight_forwarding[nn][weight_channel]);
            if (do_weight_forward) {
                write_channel_intel(weight_forwarding[nn][weight_forward_channel], _2);
            }
            // DPRINTF("Read: %f %f\n", _1, _2);
            _3 += _1*_2;
        }
        // DPRINTF("The result is: %f\n", _3);
        write_channel_intel(conv_to_result_consumer[yy][xx][nn], _3);
    }

    //DPRINTF("Calculation finished: %d %d %d\n", xx, yy, nn);
}


__kernel void result_consumer(__global float *output,
                              __address_space___shared int16* __shared) {

    int TILE0 = NOY / POY;
    int TILE1 = NOX / POX;
    int TILE2 = NOF / POF;

    int TOTAL = BATCH * TILE0 * TILE1 * TILE2;

    int index = 0;
    float _1;
    for (int i = 0; i < TOTAL; i++) {
        for (int nn = 0; nn < POF; nn++) {
            for (int yy = 0; yy < POY; yy++) {
                for (int xx = 0; xx < POX; xx++) {
                    _1 = read_channel_intel(conv_to_result_consumer[yy][xx][nn]);
                    *(output + index) = _1;
                    index++;
                }
            }
        }
    }
}
