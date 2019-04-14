#define check_linebuffer(linebuffer, length) \
    for (int le = 0; le < length; le++) { \
        printf("%f ", linebuffer[le]); \
    } \
    printf("\n");

#define define_flat_linebuffer(\
  type, input_channel, output_channel, IMG_EXTENT_0, IMG_EXTENT_1, IMG_EXTENT_2, IMG_EXTENT_3,\
  STENCIL_EXTENT_0, STENCIL_EXTENT_1, STENCIL_EXTENT_2, STENCIL_EXTENT_3,\
  STEP_0, STEP_1, STEP_2, STEP_3) \
    \
    /*Calculate input extent*/ \
    const size_t INPUT_EXTENT_0 = STENCIL_EXTENT_0 + (IMG_EXTENT_0 - 1) * STEP_0; \
    const size_t INPUT_EXTENT_1 = STENCIL_EXTENT_1 + (IMG_EXTENT_1 - 1) * STEP_1; \
    const size_t INPUT_EXTENT_2 = STENCIL_EXTENT_2 + (IMG_EXTENT_2 - 1) * STEP_2; \
    const size_t INPUT_EXTENT_3 = STENCIL_EXTENT_3 + (IMG_EXTENT_3 - 1) * STEP_3; \
    \
    /* Calculate Stride */\
    const size_t STRIDE_0 = 1; \
    const size_t STRIDE_1 = STRIDE_0 * INPUT_EXTENT_0; \
    const size_t STRIDE_2 = STRIDE_1 * INPUT_EXTENT_1; \
    const size_t STRIDE_3 = STRIDE_2 * INPUT_EXTENT_2; \
    const size_t STENCIL_STRIDE_0 = 1; \
    const size_t STENCIL_STRIDE_1 = STENCIL_STRIDE_0 * STENCIL_EXTENT_0; \
    const size_t STENCIL_STRIDE_2 = STENCIL_STRIDE_1 * STENCIL_EXTENT_1; \
    const size_t STENCIL_STRIDE_3 = STENCIL_STRIDE_2 * STENCIL_EXTENT_2; \
    const size_t STENCIL_STRIDE = STENCIL_STRIDE_3 * STENCIL_EXTENT_3; \
    const size_t STEP_STRIDE_1 = 1; \
    const size_t STEP_STRIDE_2 = STEP_STRIDE_1 * STEP_1; \
    const size_t STEP_STRIDE_3 = STEP_STRIDE_2 * STEP_2; \
    \
    \
    /* Configure linebuffers */ \
    const size_t BASE_EXTENT = (STENCIL_EXTENT_0 - 1) * STRIDE_0 + (STENCIL_EXTENT_1 - 1) * STRIDE_1 \
                                + (STENCIL_EXTENT_2 - 1) * STRIDE_2 + (STENCIL_EXTENT_3 - 1) * STRIDE_3 + 1; \
    const size_t LINEBUFFER_NUM = STEP_1 * STEP_2 * STEP_3; \
    size_t LINEBUFFER_READ[LINEBUFFER_NUM + 1]; \
    size_t LINEBUFFER_WRITE[STENCIL_STRIDE]; \
    size_t LINEBUFFER_OFFSET[LINEBUFFER_NUM + 1]; \
    /* Calculate each linebuffer size */ \
    size_t n = 0; \
    LINEBUFFER_READ[0] = 0; \
    for (size_t step_3 = STENCIL_EXTENT_3 + STEP_3 - 1; step_3 >= STENCIL_EXTENT_3; step_3--) { \
        for (size_t step_2 = STENCIL_EXTENT_2 + STEP_2 - 1; step_2 >= STENCIL_EXTENT_2; step_2--) { \
            for (size_t step_1 = STENCIL_EXTENT_1 + STEP_1 - 1; step_1 >= STENCIL_EXTENT_1; step_1--) { \
                n++; \
                LINEBUFFER_READ[n] = (step_3 / STEP_3 - 1) * STRIDE_3 + (step_2 / STEP_2 - 1) * STRIDE_2 \
                                   + (step_1 / STEP_1 - 1) * STRIDE_1 + (STENCIL_EXTENT_0 - 1) * STRIDE_0 + 1; \
                LINEBUFFER_READ[n] += LINEBUFFER_READ[n - 1]; \
                LINEBUFFER_OFFSET[n] = BASE_EXTENT - (step_3 % STEP_3) * STRIDE_3 - (step_2 % STEP_2) * STRIDE_2 \
                                     - (step_1 % STEP_1) * STRIDE_1; \
            } \
        } \
    } \
    /* Calculate total linebuffer size */ \
    const size_t LINEBUFFER_EXTENT = BASE_EXTENT - (STEP_1 * STEP_2 * STEP_3 - 1) * (INPUT_EXTENT_0 - STENCIL_EXTENT_0); \
    /* allocate the linebuffer */ \
    type linebuffer[LINEBUFFER_EXTENT]; \
    \
    /* The initialization of linebuffer */ \
    n = 0; \
    for (size_t step_3 = 0; step_3 < STEP_3; step_3++) { \
        for (size_t step_2 = 0; step_2 < STEP_2; step_2++) { \
            for (size_t step_1 = 0; step_1 < STEP_1; step_1++) { \
                n++; \
                size_t a = 0; \
                size_t a_0 = 0, a_1 = step_1, a_2 = step_2, a_3 = step_3; \
                while (a < LINEBUFFER_READ[n] - LINEBUFFER_READ[n - 1]) { \
                    size_t offset = a_3 * STRIDE_3 + a_2 * STRIDE_2 + a_1 * STRIDE_1 + a_0 * STRIDE_0; \
                    size_t address = (a + LINEBUFFER_READ[n - 1]); \
                    linebuffer[address] = read_channel_intel(input_channel);\
                    if (a_3 < STENCIL_EXTENT_3 && a_2 < STENCIL_EXTENT_2 && a_1 < STENCIL_EXTENT_1 && a_0 < STENCIL_EXTENT_0) { \
                        size_t stride_offset = a_3 * STENCIL_STRIDE_3 + a_2 * STENCIL_STRIDE_2 + a_1 * STENCIL_STRIDE_1 + a_0 * STENCIL_STRIDE_0; \
                        LINEBUFFER_WRITE[stride_offset] = address; \
                    } \
                    a_0++; \
                    if (a_0 >= INPUT_EXTENT_0) { \
                        a_0 = 0; \
                        a_1 += STEP_1; \
                        if (a_1 >= STENCIL_EXTENT_1) { \
                            a_1 = 0; \
                            a_2 += STEP_2; \
                            if (a_2 >= STENCIL_EXTENT_2) { \
                                a_2 = 0; \
                                a_3 += STEP_3; \
                            } \
                        } \
                    } \
                    a++; \
                } \
            } \
        } \
    } \
    \
    size_t start_address = 0; \
    size_t read_length = 0; \
    size_t total_read = 0; \
    \
    size_t read_extent_0 = 0; \
    size_t read_extent_1 = 0; \
    size_t read_extent_2 = 0; \
    size_t read_extent_3 = 0; \
    for (size_t dim_3 = 0; dim_3 < (unsigned) INPUT_EXTENT_3 - STENCIL_EXTENT_3 + 1; dim_3 += STEP_3) { \
        for (size_t dim_2 = 0; dim_2 < (unsigned) INPUT_EXTENT_2 - STENCIL_EXTENT_2 + 1; dim_2 += STEP_2) { \
            for (size_t dim_1 = 0; dim_1 < (unsigned) INPUT_EXTENT_1 - STENCIL_EXTENT_1 + 1; dim_1 += STEP_1) { \
                for (size_t dim_0 = 0; dim_0 < (unsigned) INPUT_EXTENT_0 - STENCIL_EXTENT_0 + 1; dim_0 += STEP_0) { \
                    /* Read new data */ \
                    /* There is a huuuuge problem for higher dimension */ \
                    start_address = dim_3 * STRIDE_3 + dim_2 * STRIDE_2 + dim_1 * STRIDE_1 + dim_0 * STRIDE_0; \
                    for (size_t r_3 = 0; r_3 < read_extent_3; r_3++) { \
                        for (size_t r_2 = 0; r_2 < read_extent_2; r_2++) { \
                            for (size_t r_1 = 0; r_1 < read_extent_1; r_1++) { \
                                n = (r_1 % STEP_1) * STEP_STRIDE_1 + (r_2 % STEP_2) * STEP_STRIDE_2 + (r_3 % STEP_3) * STEP_STRIDE_3 + 1; \
                                size_t offset = start_address + r_3 * STRIDE_3 + r_2 * STRIDE_2 + r_1 * STRIDE_1; \
                                for (size_t r_0 = 0; r_0 < read_extent_0; r_0++) { \
                                     size_t address = (total_read + r_0 + LINEBUFFER_READ[n]) % LINEBUFFER_EXTENT; \
                                     linebuffer[address] = read_channel_intel(input_channel);; \
                                } \
                            } \
                        } \
                    } \
                    total_read += read_length; \
    \
                    for (size_t s = 0; s < STENCIL_STRIDE; s++) { \
                        size_t offset = (LINEBUFFER_WRITE[s] + total_read) % LINEBUFFER_EXTENT; \
                        write_channel_intel(output_channel, linebuffer[offset]); \
                    } \
                    read_length = STEP_0; \
                    read_extent_0 = STEP_0; \
                    read_extent_1 = STEP_1; \
                    read_extent_2 = STEP_2; \
                    read_extent_3 = STEP_3; \
    \
                } \
                read_length = STENCIL_EXTENT_0; \
                read_extent_0 = STENCIL_EXTENT_0; \
                read_extent_1 = STEP_1; \
                read_extent_2 = STEP_2; \
                read_extent_3 = STEP_3; \
            } \
            read_length = STENCIL_EXTENT_0; \
                read_extent_0 = STENCIL_EXTENT_0; \
                read_extent_1 = STENCIL_EXTENT_1; \
                read_extent_2 = STEP_2; \
                read_extent_3 = STEP_3; \
        } \
        read_length = STENCIL_EXTENT_0; \
                read_extent_0 = STENCIL_EXTENT_0; \
                read_extent_1 = STENCIL_EXTENT_1; \
                read_extent_2 = STENCIL_EXTENT_2; \
                read_extent_3 = STEP_3; \
    \
    } \
    \

#define define_unrolled_flat_linebuffer(\
  type, input_channel, output_channel, IMG_EXTENT_0, IMG_EXTENT_1, IMG_EXTENT_2, IMG_EXTENT_3,\
  STENCIL_EXTENT_0, STENCIL_EXTENT_1, STENCIL_EXTENT_2, STENCIL_EXTENT_3,\
  STEP_0, STEP_1, STEP_2, STEP_3) \
      \
    /*Calculate input extent*/ \
    const size_t INPUT_EXTENT_0 = STENCIL_EXTENT_0 + (IMG_EXTENT_0 - 1) * STEP_0; \
    const size_t INPUT_EXTENT_1 = STENCIL_EXTENT_1 + (IMG_EXTENT_1 - 1) * STEP_1; \
    const size_t INPUT_EXTENT_2 = STENCIL_EXTENT_2 + (IMG_EXTENT_2 - 1) * STEP_2; \
    const size_t INPUT_EXTENT_3 = STENCIL_EXTENT_3 + (IMG_EXTENT_3 - 1) * STEP_3; \
    \
    /* Calculate Stride */\
    const size_t STRIDE_0 = 1; \
    const size_t STRIDE_1 = STRIDE_0 * INPUT_EXTENT_0; \
    const size_t STRIDE_2 = STRIDE_1 * INPUT_EXTENT_1; \
    const size_t STRIDE_3 = STRIDE_2 * INPUT_EXTENT_2; \
    \
    /* Calculate linebuffer size */ \
    /* size_t stencil_start = 0, stencil_end = 0; */ \
    /* The length calculation is wrong */ \
    const size_t LINEBUFFER_EXTENT = (IMG_EXTENT_0*STEP_0 - 1) * STRIDE_0 + (IMG_EXTENT_1*STEP_1 - 1) * STRIDE_1 \
                                   + (IMG_EXTENT_2*STEP_2 - 1) * STRIDE_2 + (IMG_EXTENT_3*STEP_3 - 1) * STRIDE_3 + 1; \
    /* allocate the linebuffer */ \
    type linebuffer[LINEBUFFER_EXTENT]; \
    \
    /* The initialization of linebuffer */ \
    for (size_t a = 0; a < LINEBUFFER_EXTENT; a++) { \
        linebuffer[a] = read_channel_intel(input_channel); \
    } \
    \
    size_t start_address = 0; \
    \
    /*Reverse the stencil loop and the outer loop*/ \
    for (size_t s_3 = 0; s_3 < STENCIL_EXTENT_3; s_3++) { \
        for (size_t s_2 = 0; s_2 < STENCIL_EXTENT_2; s_2++) { \
            for (size_t s_1 = 0; s_1 < STENCIL_EXTENT_1; s_1++) { \
                for (size_t s_0 = 0; s_0 < STENCIL_EXTENT_0; s_0++) { \
                    /* Read new data */ \
                    size_t this_start_address = s_3 * STRIDE_3 + s_2 * STRIDE_2 + s_1 * STRIDE_1 + s_0 * STRIDE_0; \
                    for (size_t a = start_address; a < this_start_address; a++) { \
                        size_t address = a % LINEBUFFER_EXTENT; \
                        linebuffer[address] = read_channel_intel(input_channel);; \
                    } \
                    start_address = this_start_address; \
    \
                    for (size_t dim_3 = 0; dim_3 < (unsigned) IMG_EXTENT_3; dim_3++) { \
                        for (size_t dim_2 = 0; dim_2 < (unsigned) IMG_EXTENT_2; dim_2++) { \
                            for (size_t dim_1 = 0; dim_1 < (unsigned) IMG_EXTENT_1; dim_1++) { \
                                for (size_t dim_0 = 0; dim_0 < (unsigned) IMG_EXTENT_0; dim_0++) { \
                                    size_t offset = dim_3 * STEP_3 * STRIDE_3 + dim_2 * STEP_2 * STRIDE_2 \
                                                  + dim_1 * STEP_1 * STRIDE_1 + dim_0 * STEP_0 * STRIDE_0; \
                                    offset = (start_address + offset) % LINEBUFFER_EXTENT; \
                                    write_channel_intel(output_channel[dim_3][dim_2][dim_1][dim_0], linebuffer[offset]); \
                                } \
                            } \
                        } \
                    } \
    \
                } \
            } \
        } \
    \
    }

#define define_linebuffer(\
  type, input_channel, output_channel, IMG_EXTENT_0, IMG_EXTENT_1, IMG_EXTENT_2, IMG_EXTENT_3,\
  STENCIL_EXTENT_0, STENCIL_EXTENT_1, STENCIL_EXTENT_2, STENCIL_EXTENT_3,\
  STEP_0, STEP_1, STEP_2, STEP_3, unroll)\
  \
    if (unroll) { \
        define_unrolled_flat_linebuffer(type, input_channel, output_channel, IMG_EXTENT_0, IMG_EXTENT_1, IMG_EXTENT_2, IMG_EXTENT_3, \
                                 STENCIL_EXTENT_0, STENCIL_EXTENT_1, STENCIL_EXTENT_2, STENCIL_EXTENT_3,\
                                 STEP_0, STEP_1, STEP_2, STEP_3)\
    } else { \
        define_flat_linebuffer(type, input_channel, output_channel, IMG_EXTENT_0, IMG_EXTENT_1, IMG_EXTENT_2, IMG_EXTENT_3,\
                                  STENCIL_EXTENT_0, STENCIL_EXTENT_1, STENCIL_EXTENT_2, STENCIL_EXTENT_3,\
                                  STEP_0, STEP_1, STEP_2, STEP_3)\
    }\

