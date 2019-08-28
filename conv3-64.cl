/*OpenCL C*/
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_channels : enable
#include "flat_linebuffer.h"

#define DPRINTF(...) printf(__VA_ARGS__); for ( int f = 0; f < 64*1024; f++ ) printf("");

#define __address_space___shared __local

channel FLOAT_VEC linebuffer_channel_first __attribute__((depth(300)));
channel FLOAT_VEC linebuffer_channel[POY - 1] __attribute__((depth(3))); // This there is wrong?

__kernel void input_serializer_on_chip(__global const FLOAT_VEC * restrict input) {
    const int TOTAL_SIZE = BATCH * (NOX / POX) * (POX + KX - 1) * (NOY / POY) * (POY + KY - 1);
    for (int i = 0; i < TOTAL_SIZE; i++) {
        write_channel_intel(linebuffer_channel_first, input[i]);
    }
}

channel FLOAT_VEC input_loader_to_feeder[POY][POX] __attribute__((depth(100)));
__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__kernel void input_loader_first() {
    const int INPUT_EXTENT_0 = KX + POX - 1;  // 6
       const int LINEBUFFER_EXTENT = POX + (POY - 1) * INPUT_EXTENT_0; //22

    int log_v = 1;
    int value = 2;
    while (value < LINEBUFFER_EXTENT + POX) {
        value = value << 1;
        log_v++;
    }

    const int JUMP_POS = LINEBUFFER_EXTENT + POX - 1;  // 25
    const int JUMP_LENGTH = value - JUMP_POS;

    while(1) {
        FLOAT_VEC linebuffer[POX];
        for (int s_0 = 0; s_0 < KY; s_0++) { 
            for (int s_1_i = 0; s_1_i < KX * value;) {
                int i = s_1_i & (value - 1);
                int s_1 = s_1_i >> log_v;

                if ((s_0|s_1) == 0 && i < LINEBUFFER_EXTENT || s_0 && s_1 == 0 && i < POX || i == 0) {
                    write_channel_intel(linebuffer_channel[POY - 2], linebuffer[0]);
                       #pragma unroll
                    for (int j = 0; j < POX - 1; j++) {
                          linebuffer[j] = linebuffer[j + 1];
                    }
                    linebuffer[POX - 1] = read_channel_intel(linebuffer_channel_first);
                }
                // First read, then write
                   if (i >= LINEBUFFER_EXTENT) {
                       int pos = i - LINEBUFFER_EXTENT;
                       write_channel_intel(input_loader_to_feeder[POY - 1][0], linebuffer[pos]); 
                   }
                
                s_1_i += (i == JUMP_POS) ? JUMP_LENGTH : 1;
            }
        }
    }
}

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY - 1)))
__kernel void input_loader() {
    int yy = get_compute_id(0);
    const int INPUT_EXTENT_0 = KX + POX - 1;  // 6
       const int LINEBUFFER_EXTENT = POX + (POY - 1) * INPUT_EXTENT_0; //22
       int initial_size = LINEBUFFER_EXTENT - (KX - 1) * (POY - yy - 1); // 16 + 2 * yy;

    int log_v = 1;
    int value = 2;
    while (value < LINEBUFFER_EXTENT + POX) {
        value = value << 1;
        log_v++;
    }

    const int JUMP_POS = LINEBUFFER_EXTENT + POX - 1;  // 25
    const int JUMP_LENGTH = value - JUMP_POS;

    while(1) {
        FLOAT_VEC linebuffer[POX];
        for (int s_0 = 0; s_0 < KY; s_0++) { 
            for (int s_1_i = 0; s_1_i < KX * value;) {     // 32 -> LINEBUFFER_EXTENT + POX
                int i = s_1_i & (value - 1);
                int s_1 = s_1_i >> log_v;

                if ((s_0|s_1) == 0 && i < initial_size || s_0 && s_1 == 0 && i < POX || i == 0) {
                    if (yy)
                        write_channel_intel(linebuffer_channel[yy-1], linebuffer[0]);
                       #pragma unroll
                    for (int j = 0; j < POX - 1; j++) {
                          linebuffer[j] = linebuffer[j + 1];
                    }
                    linebuffer[POX - 1] = read_channel_intel(linebuffer_channel[yy]);
                }
                // First read, then write
                   if (i >= LINEBUFFER_EXTENT) {
                       int pos = i - LINEBUFFER_EXTENT;
                       write_channel_intel(input_loader_to_feeder[yy][0], linebuffer[pos]); 
                   }
                s_1_i += (i == JUMP_POS) ? JUMP_LENGTH : 1;
            }
        }
        initial_size = LINEBUFFER_EXTENT;
    }
}


channel FLOAT_VEC input_forwarding[POY][POX][POF] __attribute__((depth(200)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POY, POX)))
__kernel void input_feeder() {
    int yy = get_compute_id(0);
    int xx = get_compute_id(1);
    int TILE = NOF / POF;

    FLOAT_VEC input_feeder_ibuffer[KX * KY];
    int input_scatter_channel = xx + 1;
    int window_size = POX - xx; // 4 3 2 1


    int ky_kx = 0;
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

        int lox_pox = 0;
        int pox = POX;
        while (pox > 1) {
            pox = pox >> 1;
            lox_pox++;
        }
        const int LOG_POX = lox_pox;
        const int FLATTEN_SIZE = 16 * POX;     // (KX * KY) -> 16
        const int JUMP_POS = KX * KY * POX - 1;  // 35
        const int JUMP_LENGTH = FLATTEN_SIZE - JUMP_POS;
        for (int no_ky_kx_t = 0; no_ky_kx_t < TILE * FLATTEN_SIZE;) {   
            int ky_kx_t = no_ky_kx_t & (FLATTEN_SIZE - 1);
            int t = ky_kx_t & (POX - 1);
            int ky_kx = ky_kx_t >> LOG_POX;
            
            if (no_ky_kx_t < FLATTEN_SIZE) { // if(!(no_ky_kx_t)>>6)
                FLOAT_VEC input = read_channel_intel(input_loader_to_feeder[yy][xx]);
                if (t && input_scatter_channel < POX)
                    write_channel_intel(input_loader_to_feeder[yy][input_scatter_channel], input);
                else
                    input_feeder_ibuffer[ky_kx] = input;
            }
                    
            if (t == window_size - 1) {
                write_channel_intel(input_forwarding[yy][xx][0], input_feeder_ibuffer[ky_kx]);
            }

            no_ky_kx_t += (t == window_size-1) ? ((ky_kx_t == JUMP_POS - xx ) ? JUMP_LENGTH + xx : xx + 1) : ((ky_kx_t == JUMP_POS - xx) ? JUMP_LENGTH : 1);
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
        int j = i_j & (FLATTEN_SIZE - 1);
        write_channel_intel(weight_scattering[0], weight_buffer[j]);
        i_j += (j == JUMP_POS) ? JUMP_LENGTH : 1;
    }
}


channel FLOAT_VEC weight_forwarding[POF][POY*POX] __attribute__((depth(100)));

__attribute__((max_global_work_dim(0)))__attribute__((autorun))
__attribute__((num_compute_units(POF)))
__kernel void weight_feeder() {
    int nn = get_compute_id(0);

    FLOAT_VEC weight = 0;
    int weight_size = POF; //- nn;
    int weight_scattering_channel = nn+1;

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

    int j = 0;
    FLOAT_VEC _1;
    FLOAT_VEC _2;
    bool read_success_1 = (bool)(0);
    bool read_success_2 = (bool)(0);
    float buffer[KX * KY];
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

            float sum = 0;
            #pragma unroll
            for (int k = 0; k < NIF; k++) {
                sum += _1[k]*_2[k];
            }
            #pragma unroll
            for (int i=0; i < KX * KY; i++) {
                buffer[i] = (j == i ? sum : 0);
            }

            j++;

            read_success_1 = (bool)(0);
            read_success_2 = (bool)(0);
        }

        if (j == KX * KY) {
            float result = 0;
            #pragma unroll
            for (int i = 0; i < KX * KY; i++)
                result += buffer[i];
            write_channel_intel(conv_to_drainer_channel[yy][xx][nn], result);
            j = 0;
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
    int TOTAL = BATCH * NOY * NOX * NOF / POF;
    for (int i = 0; i < TOTAL; i++) {
        outvec in;
        in = read_channel_intel(C_collector_0_inter_channel);
        output[i] = in;
        if (i % NOF == 0) {
            DPRINTF("The result is: %d %f\n", i / NOF, in.data[0]);
        }
    }
}
