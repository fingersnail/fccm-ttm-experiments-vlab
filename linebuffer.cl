/*OpenCL C*/
//AOCX:/home/weihao/tmp/t741535372.aocx
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_channels : enable
float float_from_bits(unsigned int x) {return as_float(x);}
float nan_f32() { return NAN; }
float neg_inf_f32() { return -INFINITY; }
float inf_f32() { return INFINITY; }

#include "flat_linebuffer.h"
#define DPRINTF(...) printf(__VA_ARGS__); for ( int f = 0; f < 64*1024; f++ ) printf("");

#define sqrt_f32 sqrt
#define sin_f32 sin
#define cos_f32 cos
#define exp_f32 exp
#define log_f32 log
#define abs_f32 fabs
#define floor_f32 floor
#define ceil_f32 ceil
#define round_f32 round
#define trunc_f32 trunc
#define pow_f32 pow
#define asin_f32 asin
#define acos_f32 acos
#define tan_f32 tan
#define atan_f32 atan
#define atan2_f32 atan2
#define sinh_f32 sinh
#define asinh_f32 asinh
#define cosh_f32 cosh
#define acosh_f32 acosh
#define tanh_f32 tanh
#define atanh_f32 atanh
#define fast_inverse_f32 native_recip
#define fast_inverse_sqrt_f32 native_rsqrt
int halide_gpu_thread_barrier() {
  barrier(CLK_LOCAL_MEM_FENCE);
  return 0;
}
#define __address_space___shared __local

typedef struct {float data; bool drain_signal; } float_channel_drain_struct_;
channel float _A_loader_0_channel __attribute__((depth(2))) ;
// Address spaces for kernel_A_loader_channel__run_on_device
#define __address_space__A __global
__kernel void kernel_A_loader_channel__run_on_device(
 __address_space__A const float *_A,
 __address_space___shared int16* __shared)
{
 for (int _A_loader_s0_y_y = 0; _A_loader_s0_y_y < 0 + 2; _A_loader_s0_y_y++)
 {
  for (int _A_loader_s0_x_x = 0; _A_loader_s0_x_x < 0 + 2; _A_loader_s0_x_x++)
  {
   for (int _A_loader_s0_y_yy = 0; _A_loader_s0_y_yy < 0 + 10; _A_loader_s0_y_yy++)
   {
    for (int _A_loader_s0_x_xx = 0; _A_loader_s0_x_xx < 0 + 10; _A_loader_s0_x_xx++)
    {
     int _0 = _A_loader_s0_x_x * 8;
     int _1 = _0 + _A_loader_s0_x_xx;
     int _2 = _A_loader_s0_y_y * 8;
     int _3 = _2 + _A_loader_s0_y_yy;
     int _4 = _3 * 18;
     int _5 = _1 + _4;
     float _6 = _A[_5];
     DPRINTF ("A_loader: write_channel _A_loader_0_channel data: %f\n",_6);
     write_channel_intel(_A_loader_0_channel, _6);
    } // for _A_loader_s0_x_xx
   } // for _A_loader_s0_y_yy
  } // for _A_loader_s0_x_x
 } // for _A_loader_s0_y_y
} // kernel kernel_A_loader_channel__run_on_device
#undef __address_space__A
channel float _A_feeder_0_channel __attribute__((depth(2))) ;
// Address spaces for kernel_A_feeder_channel__autorun__run_on_device

__attribute__((max_global_work_dim(0)))__attribute__((autorun))

__attribute__((num_compute_units(1,1,1)))
__kernel void kernel_A_feeder_channel__autorun__run_on_device(
)
{
 for (int _A_feeder_s0_y_y = 0; _A_feeder_s0_y_y < 0 + 2; _A_feeder_s0_y_y++)
 {
  for (int _A_feeder_s0_x_x = 0; _A_feeder_s0_x_x < 0 + 2; _A_feeder_s0_x_x++)
  {
   define_flat_linebuffer(float, _A_loader_0_channel, _A_feeder_0_channel, 8, 8, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1);
  } // for _A_feeder_s0_x_x
 } // for _A_feeder_s0_y_y
} // kernel kernel_A_feeder_channel__autorun__run_on_device
channel float _B_loader_0_channel __attribute__((depth(2))) ;
// Address spaces for kernel_B_loader_channel__run_on_device
#define __address_space__B __global
__kernel void kernel_B_loader_channel__run_on_device(
 __address_space__B const float *_B,
 __address_space___shared int16* __shared)
{
 for (int _B_loader_s0_y_y = 0; _B_loader_s0_y_y < 0 + 2; _B_loader_s0_y_y++)
 {
  for (int _B_loader_s0_x_x = 0; _B_loader_s0_x_x < 0 + 2; _B_loader_s0_x_x++)
  {
   for (int _B_loader_s0_y_yy = 0; _B_loader_s0_y_yy < 0 + 8; _B_loader_s0_y_yy++)
   {
    for (int _B_loader_s0_x_xx = 0; _B_loader_s0_x_xx < 0 + 8; _B_loader_s0_x_xx++)
    {
     for (int _B_loader_s0_win2__y = 0; _B_loader_s0_win2__y < 0 + 3; _B_loader_s0_win2__y++)
     {
      for (int _B_loader_s0_win2__x = 0; _B_loader_s0_win2__x < 0 + 3; _B_loader_s0_win2__x++)
      {
       int _7 = _B_loader_s0_win2__y * 3;
       int _8 = _B_loader_s0_win2__x + _7;
       float _9 = _B[_8];
       DPRINTF ("B_loader: write_channel _B_loader_0_channel data: %f\n",_9);
       write_channel_intel(_B_loader_0_channel, _9);
      } // for _B_loader_s0_win2__x
     } // for _B_loader_s0_win2__y
    } // for _B_loader_s0_x_xx
   } // for _B_loader_s0_y_yy
  } // for _B_loader_s0_x_x
 } // for _B_loader_s0_y_y
} // kernel kernel_B_loader_channel__run_on_device
#undef __address_space__B
channel float _B_feeder_0_channel __attribute__((depth(2))) ;
// Address spaces for kernel_B_feeder_channel__autorun__run_on_device_10

__attribute__((max_global_work_dim(0)))__attribute__((autorun))

__attribute__((num_compute_units(1,1,1)))
__kernel void kernel_B_feeder_channel__autorun__run_on_device_10(
)
{
 for (int _B_feeder_s0_y_y = 0; _B_feeder_s0_y_y < 0 + 2; _B_feeder_s0_y_y++)
 {
  for (int _B_feeder_s0_x_x = 0; _B_feeder_s0_x_x < 0 + 2; _B_feeder_s0_x_x++)
  {
   for (int _B_feeder_s0_y_yy = 0; _B_feeder_s0_y_yy < 0 + 8; _B_feeder_s0_y_yy++)
   {
    for (int _B_feeder_s0_x_xx = 0; _B_feeder_s0_x_xx < 0 + 8; _B_feeder_s0_x_xx++)
    {
     for (int _B_feeder_s0_win2__y = 0; _B_feeder_s0_win2__y < 0 + 3; _B_feeder_s0_win2__y++)
     {
      for (int _B_feeder_s0_win2__x = 0; _B_feeder_s0_win2__x < 0 + 3; _B_feeder_s0_win2__x++)
      {
       float _11;
       DPRINTF ("B_feeder: To read_channel _B_loader_0_channel\n");
       _11 = read_channel_intel(_B_loader_0_channel);
       DPRINTF ("B_feeder: read_channel _B_loader_0_channel data: %f\n",_11);
       DPRINTF ("B_feeder: write_channel _B_feeder_0_channel data: %f\n",_11);
       write_channel_intel(_B_feeder_0_channel, _11);
      } // for _B_feeder_s0_win2__x
     } // for _B_feeder_s0_win2__y
    } // for _B_feeder_s0_x_xx
   } // for _B_feeder_s0_y_yy
  } // for _B_feeder_s0_x_x
 } // for _B_feeder_s0_y_y
} // kernel kernel_B_feeder_channel__autorun__run_on_device_10
// Address spaces for kernel_C_1__run_on_device
#define __address_space__C__1 __global
__kernel void kernel_C_1__run_on_device(
 __address_space__C__1 float *_C__1,
 __address_space___shared int16* __shared)
{
 // block start
 for (int _C__1_s0_y = 0; _C__1_s0_y < 0 + 16; _C__1_s0_y++)
 {
  for (int _C__1_s0_x = 0; _C__1_s0_x < 0 + 16; _C__1_s0_x++)
  {
   int _12 = _C__1_s0_y * 16;
   int _13 = _C__1_s0_x + _12;
   _C__1[_13] = float_from_bits(0 /* 0 */);
  } // for _C__1_s0_x
 } // for _C__1_s0_y
 for (int _C__1_s1_y_y = 0; _C__1_s1_y_y < 0 + 2; _C__1_s1_y_y++)
 {
  for (int _C__1_s1_x_x = 0; _C__1_s1_x_x < 0 + 2; _C__1_s1_x_x++)
  {
   for (int _C__1_s1_y_yy = 0; _C__1_s1_y_yy < 0 + 8; _C__1_s1_y_yy++)
   {
    for (int _C__1_s1_x_xx = 0; _C__1_s1_x_xx < 0 + 8; _C__1_s1_x_xx++)
    {
     for (int _C__1_s1_win2__y = 0; _C__1_s1_win2__y < 0 + 3; _C__1_s1_win2__y++)
     {
      for (int _C__1_s1_win2__x = 0; _C__1_s1_win2__x < 0 + 3; _C__1_s1_win2__x++)
      {
       int _14 = _C__1_s1_x_x * 8;
       int _15 = _14 + _C__1_s1_x_xx;
       int _16 = _C__1_s1_y_y * 8;
       int _17 = _16 + _C__1_s1_y_yy;
       int _18 = _17 * 16;
       int _19 = _15 + _18;
       float _20 = _C__1[_19];
       float _21;
       DPRINTF ("C$1: To read_channel _A_feeder_0_channel\n");
       _21 = read_channel_intel(_A_feeder_0_channel);
       DPRINTF ("C$1: read_channel _A_feeder_0_channel data: %f\n",_21);
       float _22;
       DPRINTF ("C$1: To read_channel _B_feeder_0_channel\n");
       _22 = read_channel_intel(_B_feeder_0_channel);
       DPRINTF ("C$1: read_channel _B_feeder_0_channel data: %f\n",_22);
       float _23 = _21 * _22;
       float _24 = _20 + _23;
       _C__1[_19] = _24;
      } // for _C__1_s1_win2__x
     } // for _C__1_s1_win2__y
    } // for _C__1_s1_x_xx
   } // for _C__1_s1_y_yy
  } // for _C__1_s1_x_x
 } // for _C__1_s1_y_y
 // block end
} // kernel kernel_C_1__run_on_device
#undef __address_space__C__1
