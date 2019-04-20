/*OpenCL C*/
//AOCX:/home/weihao/tmp/t1399651146.aocx 
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_channels : enable
float float_from_bits(unsigned int x) {return as_float(x);}
float nan_f32() { return NAN; }
float neg_inf_f32() { return -INFINITY; }
float inf_f32() { return INFINITY; }

#include "flat_linebuffer.h"


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
__kernel void kernel_A_loader(
 const int _A_min_0,
 const int _A_min_1,
 const int _A_stride_1,
 const int _C_extent_0,
 const int _C_extent_1,
 const int _C_min_0,
 const int _C_min_1,
 __address_space__A const float *_A,
 __address_space___shared int16* __shared)
{
 int _0 = _C_extent_1 + 63;
 int _1 = _0 >> 5;
 for (int _A_loader_s0_y_y = 0; _A_loader_s0_y_y < 0 + _1; _A_loader_s0_y_y++)
 {
  int _2;
  int _3 = _C_extent_1 + 31;
  int _4 = _3 >> 5;
  bool _5 = _A_loader_s0_y_y == _4;
  if (_5)
  {
   _2 = 1;
  } // if _5
  else
  {
   int _6 = _C_extent_0 + 31;
   int _7 = _6 >> 5;
   _2 = _7;
  } // if _5 else
  for (int _A_loader_s0_x_x = 0; _A_loader_s0_x_x < 0 + _2; _A_loader_s0_x_x++)
  {
   for (int _A_loader_s0_y_yy = 0; _A_loader_s0_y_yy < 0 + 34; _A_loader_s0_y_yy++)
   {
    for (int _A_loader_s0_x_xx = 0; _A_loader_s0_x_xx < 0 + 34; _A_loader_s0_x_xx++)
    {
     int _8 = _A_loader_s0_x_x * 32;
     int _9 = _8 + _C_min_0;
     int _10 = _C_min_0 + _C_extent_0;
     int _11 = _10 + -32;
     int _12 = min(_9, _11);
     int _13 = _12 + _A_loader_s0_x_xx;
     int _14 = _A_loader_s0_y_y * 32;
     int _15 = _14 + _C_min_1;
     int _16 = _C_min_1 + _C_extent_1;
     int _17 = _16 + -32;
     int _18 = min(_15, _17);
     int _19 = _18 + _A_loader_s0_y_yy;
     int _20 = _19 * _A_stride_1;
     int _21 = _13 + _20;
     int _22 = _A_min_1 * _A_stride_1;
     int _23 = _A_min_0 + _22;
     int _24 = _21 - _23;
     float _25 = _A[_24];
     write_channel_intel(_A_loader_0_channel, _25);
    } // for _A_loader_s0_x_xx
   } // for _A_loader_s0_y_yy
  } // for _A_loader_s0_x_x
 } // for _A_loader_s0_y_y
} // kernel kernel_A_loader_channel__run_on_device
#undef __address_space__A
channel float _A_feeder_0_channel __attribute__((depth(2))) ;
// Address spaces for kernel_A_feeder_channel

__attribute__((max_global_work_dim(0)))__attribute__((autorun))

__attribute__((num_compute_units(1,1,1)))
__kernel void kernel_A_feeder_channel(
)
{
 while (1)
 {
  define_flat_linebuffer(float, _A_loader_0_channel, _A_feeder_0_channel, 32, 32, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1);
 } // while _A_feeder_s0_y_y_infinite
} // kernel kernel_A_feeder_channel__autorun__run_on_device
channel float _B_loader_0_channel __attribute__((depth(2))) ;
// Address spaces for kernel_B_loader_channel__run_on_device
#define __address_space__B __global
__kernel void kernel_B_loader(
 const int _B_min_0,
 const int _B_min_1,
 const int _B_stride_1,
 const int _C_extent_0,
 const int _C_extent_1,
 __address_space__B const float *_B,
 __address_space___shared int16* __shared)
{
 int _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter;
 int _26 = _C_extent_1 + 63;
 int _27 = _26 >> 5;
 for (int _B_loader_s0_y_y = 0; _B_loader_s0_y_y < 0 + _27; _B_loader_s0_y_y++)
 {
  int _28;
  int _29 = _C_extent_1 + 31;
  int _30 = _29 >> 5;
  bool _31 = _B_loader_s0_y_y == _30;
  if (_31)
  {
   _28 = 1;
  } // if _31
  else
  {
   int _32 = _C_extent_0 + 31;
   int _33 = _32 >> 5;
   _28 = _33;
  } // if _31 else
  for (int _B_loader_s0_x_x = 0; _B_loader_s0_x_x < 0 + _28; _B_loader_s0_x_x++)
  {
   for (int _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter = 0; _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter < 0 + 16384; )
   {
    // block start
    int _34 = _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter & 3;
    int _35 = _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter & 15;
    int _36 = _35 >> 2;
    int _37 = _36 * _B_stride_1;
    int _38 = _34 + _37;
    int _39 = _B_min_1 * _B_stride_1;
    int _40 = _B_min_0 + _39;
    int _41 = _38 - _40;
    float _42 = _B[_41];
    write_channel_intel(_B_loader_0_channel, _42);
    // block start
        int _43 = _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter + 1;
    _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter = _43;
    // block start
    int _44 = _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter & 3;
    bool _45 = _44 == 3;
    if (_45)
    {
          int _46 = _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter + 1;
     _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter = _46;
    } // if _45
    int _47 = _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter & 15;
    bool _48 = _47 == 12;
    if (_48)
    {
          int _49 = _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter + 4;
     _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter = _49;
    } // if _48
    // block end
    // block end
    // block end
   } // for _B_loader_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter
  } // for _B_loader_s0_x_x
 } // for _B_loader_s0_y_y
} // kernel kernel_B_loader_channel__run_on_device
#undef __address_space__B
channel float _B_feeder_0_channel __attribute__((depth(2))) ;
// Address spaces for kernel_B_feeder_channel__autorun__run_on_device_50

__attribute__((max_global_work_dim(0)))__attribute__((autorun))

__attribute__((num_compute_units(1,1,1)))
__kernel void kernel_B_feeder(
)
{
 int _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter;
 while (1)
 {
  for (int _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter = 0; _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter < 0 + 16384; )
  {
   // block start
   float _51;
   _51 = read_channel_intel(_B_loader_0_channel);
   write_channel_intel(_B_feeder_0_channel, _51);
   // block start
      int _52 = _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter + 1;
   _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter = _52;
   // block start
   int _53 = _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter & 3;
   bool _54 = _53 == 3;
   if (_54)
   {
        int _55 = _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter + 1;
    _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter = _55;
   } // if _54
   int _56 = _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter & 15;
   bool _57 = _56 == 12;
   if (_57)
   {
        int _58 = _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter + 4;
    _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter = _58;
   } // if _57
   // block end
   // block end
   // block end
  } // for _B_feeder_s0_y_yy___yy___xx___win2__y___win2__x__flattened_loop__no_loop_counter
 } // while _B_feeder_s0_y_y_infinite
} // kernel kernel_B_feeder_channel__autorun__run_on_device_50
// Address spaces for kernel_C__run_on_device
#define __address_space__C __global
__kernel void kernel_C(
 const int _C_extent_0,
 const int _C_extent_1,
 const int _C_min_0,
 const int _C_min_1,
 const int _C_stride_1,
 __address_space__C float *_C,
 __address_space___shared int16* __shared)
{
 // block start
 int _59 = _C_extent_1 + 1;
 for (int _C_s0_y = _C_min_1; _C_s0_y < _C_min_1 + _59; _C_s0_y++)
 {
  int _60;
  bool _61 = _C_s0_y == _C_extent_1;
  if (_61)
  {
   _60 = 1;
  } // if _61
  else
  {
   _60 = _C_extent_0;
  } // if _61 else
  for (int _C_s0_x = _C_min_0; _C_s0_x < _C_min_0 + _60; _C_s0_x++)
  {
   int _62 = _C_s0_y * _C_stride_1;
   int _63 = _C_s0_x + _62;
   int _64 = _C_min_1 * _C_stride_1;
   int _65 = _C_min_0 + _64;
   int _66 = _63 - _65;
   _C[_66] = float_from_bits(0 /* 0 */);
  } // for _C_s0_x
 } // for _C_s0_y
 int _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter;
 int _67 = _C_extent_1 + 63;
 int _68 = _67 >> 5;
 for (int _C_s1_y_y = 0; _C_s1_y_y < 0 + _68; _C_s1_y_y++)
 {
  int _69;
  int _70 = _C_extent_1 + -1;
  int _71 = _70 >> 5;
  int _72 = _C_s1_y_y - _71;
  bool _73 = _72 == 1;
  if (_73)
  {
   _69 = 1;
  } // if _73
  else
  {
   int _74 = _C_extent_0 + 31;
   int _75 = _74 >> 5;
   _69 = _75;
  } // if _73 else
  for (int _C_s1_x_x = 0; _C_s1_x_x < 0 + _69; _C_s1_x_x++)
  {
   for (int _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter = 0; _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter < 0 + 4096; )
   {
    // block start
    int _76 = _C_s1_x_x * 32;
    int _77 = _76 + _C_min_0;
    int _78 = _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter & 127;
    int _79 = _78 >> 2;
    int _80 = _77 + _79;
    int _81 = _C_s1_y_y * 32;
    int _82 = _81 + _C_min_1;
    int _83 = _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter >> 7;
    int _84 = _82 + _83;
    int _85 = _84 * _C_stride_1;
    int _86 = _80 + _85;
    int _87 = _C_min_1 * _C_stride_1;
    int _88 = _C_min_0 + _87;
    int _89 = _86 - _88;
    float _90 = _C[_89];
    float _91;
    _91 = read_channel_intel(_A_feeder_0_channel);
    float _92;
    _92 = read_channel_intel(_B_feeder_0_channel);
    float _93 = _91 * _92;
    float _94 = _90 + _93;
    _C[_89] = _94;
    // block start
    int _95 = _C_s1_x_x * 32;
    int _96 = _95 + _C_min_0;
    int _97 = _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter & 127;
    int _98 = _97 >> 2;
    int _99 = _96 + _98;
    int _100 = _C_s1_y_y * 32;
    int _101 = _100 + _C_min_1;
    int _102 = _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter >> 7;
    int _103 = _101 + _102;
    int _104 = _103 * _C_stride_1;
    int _105 = _99 + _104;
    int _106 = _C_min_1 * _C_stride_1;
    int _107 = _C_min_0 + _106;
    int _108 = _105 - _107;
    float _109 = _C[_108];
    float _110;
    _110 = read_channel_intel(_A_feeder_0_channel);
    float _111;
    _111 = read_channel_intel(_B_feeder_0_channel);
    float _112 = _110 * _111;
    float _113 = _109 + _112;
    _C[_108] = _113;
    // block start
    int _114 = _C_s1_x_x * 32;
    int _115 = _114 + _C_min_0;
    int _116 = _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter & 127;
    int _117 = _116 >> 2;
    int _118 = _115 + _117;
    int _119 = _C_s1_y_y * 32;
    int _120 = _119 + _C_min_1;
    int _121 = _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter >> 7;
    int _122 = _120 + _121;
    int _123 = _122 * _C_stride_1;
    int _124 = _118 + _123;
    int _125 = _C_min_1 * _C_stride_1;
    int _126 = _C_min_0 + _125;
    int _127 = _124 - _126;
    float _128 = _C[_127];
    float _129;
    _129 = read_channel_intel(_A_feeder_0_channel);
    float _130;
    _130 = read_channel_intel(_B_feeder_0_channel);
    float _131 = _129 * _130;
    float _132 = _128 + _131;
    _C[_127] = _132;
    // block start
        int _133 = _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter + 1;
    _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter = _133;
    int _134 = _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter & 3;
    bool _135 = _134 == 3;
    if (_135)
    {
          int _136 = _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter + 1;
     _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter = _136;
    } // if _135
    // block end
    // block end
    // block end
    // block end
   } // for _C_s1_y_yy___yy___xx___win2__y__flattened_loop__no_loop_counter
  } // for _C_s1_x_x
 } // for _C_s1_y_y
 // block end
} // kernel kernel_C
#undef __address_space__C
