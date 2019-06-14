/*OpenCL C*/
//AOCX:/nfs_home/nsrivas2/tmp/t168203670.aocx 
#pragma OPENCL FP_CONTRACT ON
#pragma OPENCL EXTENSION cl_intel_channels : enable
float float_from_bits(unsigned int x) {return as_float(x);}
float nan_f32() { return NAN; }
float neg_inf_f32() { return -INFINITY; }
float inf_f32() { return INFINITY; }

#define DPRINTF(...) 
//printf(__VA_ARGS__); for ( int f = 0; f < 64*1024; f++ ) printf("");

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

// i,j,k,l,ii(1),jj (unrolled),kk (unrolled),ll, [iii,jjj,kkk] compute buffer ,lll(16)

typedef struct {FLOAT_VEC data; bool drain_signal; } drain_struct;

channel drain_struct _A_loader_0_inter_channel __attribute__((depth(_JJJ_))) ;
channel drain_struct _A_loader_0_channel[_JJ_-1] __attribute__((depth(0))) ;

#define __address_space__A_serializer_mem_channel __global
__kernel void kernel_A_loader_channel(
 const int _C_deserializer_extent_0,
 const int _C_deserializer_extent_1,
 const int _C_deserializer_extent_2,
 __address_space__A_serializer_mem_channel const FLOAT_VEC *_A_serializer_mem_channel,
 __address_space___shared int16* __shared)
{

 int addr = 0;
 int base_addr = 0;

 int _50 = _C_deserializer_extent_2 + 1;
 int _51 = _50;
 for (int _A_loader_s0_i_i = 0; _A_loader_s0_i_i < 0 + _51; _A_loader_s0_i_i++)
 {
  int _52;
  int _53 = _C_deserializer_extent_2;
  int _54 = _53;
  bool _55 = _A_loader_s0_i_i == _54;
  if (_55)
  {
   _52 = 1;
  } // if _55
  else
  {
   int _56 = _C_deserializer_extent_1;
   int _57 = _56;
   _52 = _57;
  } // if _55 else
  for (int _A_loader_s0_j_j = 0; _A_loader_s0_j_j < 0 + _52; _A_loader_s0_j_j++)
  {
   int _58;
   int _59 = _C_deserializer_extent_2;
   int _60 = _59;
   bool _61 = _A_loader_s0_i_i == _60;
   if (_61)
   {
    _58 = 1;
   } // if _61
   else
   {
    int _62 = _C_deserializer_extent_0;
    int _63 = _62;
    _58 = _63;
   } // if _61 else
   for (int _A_loader_s0_k_k = 0; _A_loader_s0_k_k < 0 + _58; _A_loader_s0_k_k++)
   {
    int _64;
    int _65 = _C_deserializer_extent_2;
    int _66 = _65;
    bool _67 = _A_loader_s0_i_i == _66;
    if (_67)
    {
     _64 = _II_*_JJ_*_LL_*_III_*_JJJ_;
    } // if _67
    else
    {
     _64 = _L_*_II_*_JJ_*_LL_*_III_*_JJJ_;
    } // if _67 else
    for (int l_ii_ll_iii_jjj_jj = 0; l_ii_ll_iii_jjj_jj < _64; l_ii_ll_iii_jjj_jj++) {
        addr = ((_A_loader_s0_i_i * _J_) + _A_loader_s0_j_j) * (_L_*_II_*_LL_*_III_*_JJJ_*_JJ_) + l_ii_ll_iii_jjj_jj; 
        FLOAT_VEC _83 = _A_serializer_mem_channel[addr];
        int l = l_ii_ll_iii_jjj_jj / (_II_*_LL_*_III_*_JJJ_*_JJ_);
        int ll = (l_ii_ll_iii_jjj_jj & (_LL_*_III_*_JJJ_*_JJ_-1))/(_III_*_JJJ_*_JJ_);
        drain_struct s = {_83, !((_A_loader_s0_i_i==0) && (_A_loader_s0_j_j==0) && (_A_loader_s0_k_k==0)) && (l==0) && (ll==0)};
        DPRINTF ("A_loader: write_channel _A_loader_0_channel data: %f drain: %d\n",s, s.drain_signal);
        write_channel_intel(_A_loader_0_inter_channel, s);
    }
   } // for _A_loader_s0_k_k
  } // for _A_loader_s0_j_j
 } // for _A_loader_s0_i_i
} // kernel kernel_A_loader_channel
#undef __address_space__A_serializer_mem_channel

channel drain_struct _A_feeder_0_inter_channel[_JJ_] __attribute__((depth(_JJJ_))) ;
channel drain_struct _A_feeder_0_channel[_JJ_][_KK_-1] __attribute__((depth(0))) ;
// Address spaces for kernel_A_feeder_channel_84

__attribute__((max_global_work_dim(0)))__attribute__((autorun))

__attribute__((num_compute_units(_JJ_, 1, 1)))
__kernel void kernel_A_feeder_channel_84(
)
{
 int _85 = get_compute_id(0);
;
 FLOAT_VEC _A_feeder_0_ibuffer[2]; // as kkk is the last removed loop, reuse factor: KKK*KK  = 16*9
 bool pd[2] = {false, false};
 drain_struct ps;

 bool _write_success;
 bool _read_success;
 int _first;
 int _A_feeder_s0_w;
 int _A_feeder_s0_r;
 int _A_feeder_s0__scatter_loop;
 int _A_feeder_s0__scatter_feed_loop__no_loop_counter;
 int _A_feeder_s0__feed_loop;
 // block start
  bool _86 = (bool)(0);
 _write_success = _86;
 // block start
  bool _87 = (bool)(0);
 _read_success = _87;
 // block start
  _first = 1;
 // block start
  _A_feeder_s0_w = 0;
 // block start
  _A_feeder_s0_r = 1;
 // block start
  _A_feeder_s0__scatter_loop = 0;
 // block start
 // block start
  _A_feeder_s0__feed_loop = 0;
 while (1)
 {
   int _88 = _JJ_ - _85;
   int _89 = _88;
   int _90 = _89 + _KKK_; // LLL = 1, reuse factor KKK*KK
    // block start
    int _91 = _JJ_ - _85;
    int _92 = _91;
    bool _93 = _A_feeder_s0__scatter_loop < _92;
    if (_93)
    {
     FLOAT_VEC _94;
     drain_struct ps;

     DPRINTF ("A_feeder(%d): To read_channel_nb _A_loader_0_channel\n",_85);
     if (_85 != 0)
       ps = read_channel_nb_intel(_A_loader_0_channel[_85-1], &_read_success);
     else
       ps = read_channel_nb_intel(_A_loader_0_inter_channel, &_read_success);
     _94 = ps.data;

     DPRINTF ("A_feeder(%d): read_channel_nb _A_loader_0_channel data: %f\n",_85,_94);
     bool _95 = _A_feeder_s0__scatter_loop < 1;
     bool _96 = _read_success && _95;
     if (_96)
     {
      // block start
      _A_feeder_0_ibuffer[_A_feeder_s0_w] = _94;
      pd[_A_feeder_s0_w] = ps.drain_signal;
      _A_feeder_s0__scatter_loop++;
      // block end
      // block end
     } // if _96
     else
     {
      if (_read_success)
      {
       // block start
       DPRINTF ("A_feeder(%d): write_channel _A_loader_0_channel data: %f\n",_85,_94);
       write_channel_intel(_A_loader_0_channel[_85], ps);
       _A_feeder_s0__scatter_loop++;
       // block end
       // block end
      } // if _read_success
     } // if _96 else
    } // if _93
    // block start
    bool _105 = _A_feeder_s0__feed_loop < (_KKK_);
    if (_105)
    {
     // block start
     bool _106 = !(_first);
     if (_106)
     {
      FLOAT_VEC _111 = _A_feeder_0_ibuffer[_A_feeder_s0_r];
      drain_struct s = {_111, pd[_A_feeder_s0_r]};
      DPRINTF ("A_feeder(%d): write_channel_nb _A_feeder_0_channel data: %f drain: %d\n",_85,_111, s.drain_signal);
      _write_success = write_channel_nb_intel(_A_feeder_0_inter_channel[_85], s);
     } // if _106
     bool _112 = _first || _write_success;
     if (_112)
     {
      _A_feeder_s0__feed_loop++;
     } // if _112
    } // if _105
    int _115 = _JJ_ - _85;
    int _116 = _115;
    bool _117 = _A_feeder_s0__scatter_loop == _116;
    bool _118 = _A_feeder_s0__feed_loop == (_KKK_);
    bool _119 = _117 && _118;
    if (_119)
    {
     // block start
          bool _120 = (bool)(0);
     _first = _120;
     // block start
          bool _121 = (bool)(_A_feeder_s0_w);
     bool _122 = !(_121);
     _A_feeder_s0_w = _122;
     // block start
          bool _123 = (bool)(_A_feeder_s0_r);
     bool _124 = !(_123);
     _A_feeder_s0_r = _124;
     // block start
          _A_feeder_s0__scatter_loop = 0;
          _A_feeder_s0__feed_loop = 0;
     // block end
     // block end
     // block end
     // block end
    } // if _119
    // block end
    // block end
 } // while _A_feeder_s0_i_i_infinite
 // block end
 // block end
 // block end
 // block end
 // block end
 // block end
 // block end
 // block end
} // kernel kernel_A_feeder_channel_84

channel FLOAT_VEC _B_loader_0_inter_channel __attribute__((depth(_KKK_))) ;
channel FLOAT_VEC _B_loader_0_channel[_KK_-1] __attribute__((depth(0))) ;

// Address spaces for kernel_B_loader_channel
#define __address_space__C_serializer_mem_channel __global
__kernel void kernel_B_loader_channel(
 const int _C_deserializer_extent_0,
 const int _C_deserializer_extent_1,
 const int _C_deserializer_extent_2,
 __address_space__C_serializer_mem_channel const FLOAT_VEC *_C_serializer_mem_channel,
 __address_space___shared int16* __shared)
{

 int base_addr = 0;
 int addr = 0;
 int _125 = _C_deserializer_extent_2 + 1;
 int _126 = _125;
 for (int _B_loader_s0_i_i = 0; _B_loader_s0_i_i < 0 + _126; _B_loader_s0_i_i++)
 {
  int _127;
  int _128 = _C_deserializer_extent_2;
  int _129 = _128;
  bool _130 = _B_loader_s0_i_i == _129;
  if (_130)
  {
   _127 = 1;
  } // if _130
  else
  {
   int _131 = _C_deserializer_extent_1;
   int _132 = _131;
   _127 = _132;
  } // if _130 else
  for (int _B_loader_s0_j_j = 0; _B_loader_s0_j_j < 0 + _127; _B_loader_s0_j_j++)
  {
   int _133;
   int _134 = _C_deserializer_extent_2;
   int _135 = _134;
   bool _136 = _B_loader_s0_i_i == _135;
   if (_136)
   {
    _133 = 1;
   } // if _136
   else
   {
    int _137 = _C_deserializer_extent_0;
    int _138 = _137;
    _133 = _138;
   } // if _136 else
   for (int _B_loader_s0_k_k = 0; _B_loader_s0_k_k < 0 + _133; _B_loader_s0_k_k++)
   {
    int _139;
    int _140 = _C_deserializer_extent_2;
    int _141 = _140;
    bool _142 = _B_loader_s0_i_i == _141;
    if (_142)
    {
     _139 = 1;
    } // if _142
    else
    {
     _139 = _L_;
    } // if _142 else
    base_addr = _B_loader_s0_k_k*(_L_*_KK_*_LL_*_KKK_);
    for (int l = 0; l < _139; l++) {
      for (int kk_ll_kkk = 0; kk_ll_kkk < (_KK_*_LL_*_KKK_); kk_ll_kkk++) {
        addr = base_addr + l*(_KK_*_LL_*_KKK_) + kk_ll_kkk;
        FLOAT_VEC _158 = _C_serializer_mem_channel[addr];
        DPRINTF ("B_loader: write_channel _B_loader_0_channel data: %f, %f\n",_158[0],_158[1]);
        write_channel_intel(_B_loader_0_inter_channel, _158);
      }
    } // for _B_loader_s0_l__y_l__y___l__y___l__x___kk___ll__x___kkk__flattened_loop
   } // for _B_loader_s0_k_k
  } // for _B_loader_s0_j_j
 } // for _B_loader_s0_i_i
} // kernel kernel_B_loader_channel
#undef __address_space__C_serializer_mem_channel

channel FLOAT_VEC _B_feeder_0_inter_channel[_KK_] __attribute__((depth(_KKK_))) ;
channel FLOAT_VEC _B_feeder_0_channel[_KK_][_JJ_-1] __attribute__((depth(0))) ;
// Address spaces for kernel_B_feeder_channel

__attribute__((max_global_work_dim(0)))__attribute__((autorun))

__attribute__((num_compute_units(_KK_, 1, 1)))
__kernel void kernel_B_feeder_channel(
)
{
 int _159 = get_compute_id(0);
;
 FLOAT_VEC _B_feeder_0_ibuffer[2][_LL_][_KKK_]; // ii is last removed loop LL*KKK, reuse factor: JJ*III*JJJ = 8*16*16 = 2048
 bool _write_success;
 bool _read_success;
 int _first;
 int _B_feeder_s0_w;
 int _B_feeder_s0_r;
 int _B_feeder_s0__scatter_loop;
 int _B_feeder_s0__feed_loop;
 // block start
  bool _160 = (bool)(0);
 _write_success = _160;
 // block start
  bool _161 = (bool)(0);
 _read_success = _161;
 // block start
  _first = 1;
 // block start
  _B_feeder_s0_w = 0;
 // block start
  _B_feeder_s0_r = 1;
 // block start
  _B_feeder_s0__scatter_loop = 0;
 // block start
 // block start
  _B_feeder_s0__feed_loop = 0;
 while (1)
 {
   int _162 = _KK_ - _159;
   int _163 = _162 * _LL_ * _KKK_;
   int _164 = _163 + _II_*_LL_*_III_*_JJJ_*_KKK_;
    // block start
    int _165 = _KK_ - _159;
    int _166 = _165 * _LL_*_KKK_;
    bool _167 = _B_feeder_s0__scatter_loop < _166;
    if (_167)
    {
     FLOAT_VEC _168;
     DPRINTF ("B_feeder(%d): To read_channel_nb _B_loader_0_channel\n",_159);
     if (_159 == 0)
       _168 = read_channel_nb_intel(_B_loader_0_inter_channel, &_read_success);
     else
       _168 = read_channel_nb_intel(_B_loader_0_channel[_159-1], &_read_success);

     DPRINTF ("B_feeder(%d): read_channel_nb _B_loader_0_channel data: %f, %f\n",_159,_168[0],_168[1]);
     bool _169 = _B_feeder_s0__scatter_loop < (_LL_*_KKK_);
     bool _170 = _read_success && _169;
     if (_170)
     {
      // block start
      int _171 = _B_feeder_s0__scatter_loop & (_LL_*_KKK_-1);
      int _172 = _171/_KKK_;
      int _173 = _B_feeder_s0__scatter_loop & (_KKK_-1);
      _B_feeder_0_ibuffer[_B_feeder_s0_w][_172][_173] = _168;
      // block start
            int _174 = _B_feeder_s0__scatter_loop + 1;
      _B_feeder_s0__scatter_loop = _174;
      // block end
      // block end
     } // if _170
     else
     {
      if (_read_success)
      {
       DPRINTF ("B_feeder(%d): write_channel _B_loader_0_channel data: %f, %f\n",_159,_168[0],_168[1]);
       write_channel_intel(_B_loader_0_channel[_159], _168);
       // block start
              int _177 = _B_feeder_s0__scatter_loop + 1;
       _B_feeder_s0__scatter_loop = _177;
       // block end
       // block end
      } // if _read_success
     } // if _170 else
    } // if _167
    // block start
    bool _179 = _B_feeder_s0__feed_loop < (_II_*_LL_*_III_*_JJJ_*_KKK_);
    if (_179)
    {
     // block start
     bool _180 = !(_first);
     if (_180)
     {
      int _181 = _B_feeder_s0__feed_loop & (_LL_*_III_*_JJJ_*_KKK_-1);
      int _182 = _181/(_III_*_JJJ_*_KKK_);
      int _183 = _B_feeder_s0__feed_loop & (_KKK_-1);
      FLOAT_VEC _184 = _B_feeder_0_ibuffer[_B_feeder_s0_r][_182][_183];
      DPRINTF ("B_feeder(%d): write_channel_nb _B_feeder_0_channel data: %f, %f\n",_159,_184[0],_184[1]);
      _write_success = write_channel_nb_intel(_B_feeder_0_inter_channel[_159], _184);
      (void)_184;
     } // if _180
     bool _185 = _first || _write_success;
     if (_185)
     {
      // block start
            int _186 = _B_feeder_s0__feed_loop + 1;
      _B_feeder_s0__feed_loop = _186;
      // block end
     } // if _185
     // block end
    } // if _179
    int _188 = _KK_ - _159;
    int _189 = _188 * (_LL_*_KKK_);
    bool _190 = _B_feeder_s0__scatter_loop == _189;
    bool _191 = _B_feeder_s0__feed_loop == (_II_*_LL_*_III_*_JJJ_*_KKK_);
    bool _192 = _190 && _191;
    if (_192)
    {
     // block start
          bool _193 = (bool)(0);
     _first = _193;
     // block start
          bool _194 = (bool)(_B_feeder_s0_w);
     bool _195 = !(_194);
     _B_feeder_s0_w = _195;
     // block start
          bool _196 = (bool)(_B_feeder_s0_r);
     bool _197 = !(_196);
     _B_feeder_s0_r = _197;
     // block start
          _B_feeder_s0__scatter_loop = 0;
          _B_feeder_s0__feed_loop = 0;
     // block end
     // block end
     // block end
     // block end
    } // if _192
    // block end
    // block end
 } // while _B_feeder_s0_i_i_infinite
 // block end
 // block end
 // block end
 // block end
 // block end
 // block end
 // block end
 // block end
} // kernel kernel_B_feeder_channel

  channel float _D_0_channel[_JJ_][_KK_] __attribute__((depth(_III_*_JJJ_*_KKK_))) ;
  // Address spaces for kernel_D_channel
  
  __attribute__((max_global_work_dim(0)))__attribute__((autorun))
  
  __attribute__((num_compute_units(_JJ_, _KK_, 1)))
  __kernel void kernel_D_channel(
  )
  {
   FLOAT_VEC _rch_1;
   FLOAT_VEC _rch_0;
   int _198 = get_compute_id(0);
   int _199 = get_compute_id(1);
  
   int _D_s1_l__y_l__y___l__y___l__x___ii___ll__y___ll__x___iii___jjj___kkk___lll__y__flattened_loop = 0;
   float _D_0_ibuffer[_II_*_III_*_JJJ_*_KKK_];
   bool first = true;
   // block start
  #pragma unroll 
  for (int _D_s1_k_kk_kkk__scan_elem__D_s1_j_jj_jjj__scan_elem__D_s1_i_ii_iii__scan_elem__D_s1_i_ii_ii__scan_elem_ = 0; _D_s1_k_kk_kkk__scan_elem__D_s1_j_jj_jjj__scan_elem__D_s1_i_ii_iii__scan_elem__D_s1_i_ii_ii__scan_elem_ < (_II_*_III_*_JJJ_*_KKK_); _D_s1_k_kk_kkk__scan_elem__D_s1_j_jj_jjj__scan_elem__D_s1_i_ii_iii__scan_elem__D_s1_i_ii_ii__scan_elem_++)
   {
    _D_0_ibuffer[_D_s1_k_kk_kkk__scan_elem__D_s1_j_jj_jjj__scan_elem__D_s1_i_ii_iii__scan_elem__D_s1_i_ii_ii__scan_elem_] = float_from_bits(0 /* 0 */);
   } // for _D_s1_k_kk_kkk__scan_elem__D_s1_j_jj_jjj__scan_elem__D_s1_i_ii_iii__scan_elem__D_s1_i_ii_ii__scan_elem_
   while (1)
   {

     drain_struct s;
     FLOAT_VEC _232;
     DPRINTF ("D(%d,%d): To read_channel _A_feeder_0_channel\n",_198,_199);
     if (_199 == 0)
       s = read_channel_intel(_A_feeder_0_inter_channel[_198]);
     else
       s = read_channel_intel(_A_feeder_0_channel[_198][_199-1]);
     _232 = s.data;
     _rch_0 = _232;

     DPRINTF ("D(%d,%d): read_channel _A_feeder_0_channel data: %f\n",_198,_199,_232);
     bool _233 = _199 < (_KK_-1);
     if (_233)
     {
      DPRINTF ("D(%d,%d): write_channel _A_feeder_0_channel data: %f\n",_198,_199,_232);
      write_channel_intel(_A_feeder_0_channel[_198][_199], s);
     } // if _233
 

     // block start
     FLOAT_VEC _201;
     DPRINTF ("D(%d,%d): To read_channel _B_feeder_0_channel\n",_198,_199);
     if (_198 == 0)
       _201 = read_channel_intel(_B_feeder_0_inter_channel[_199]);
     else
       _201 = read_channel_intel(_B_feeder_0_channel[_199][_198-1]);

     DPRINTF ("D(%d,%d): read_channel _B_feeder_0_channel data: %f, %f\n",_198,_199,_201[0],_201[1]);
     // block start
     bool _202 = _198 < (_JJ_-1);
     if (_202)
     {
      DPRINTF ("D(%d,%d): write_channel _B_feeder_0_channel data: %f, %f\n",_198,_199,_201[0],_201[1]);
      write_channel_intel(_B_feeder_0_channel[_199][_198], _201);
     } // if _202
        _rch_1 = _201;
     // block end
  

     float _231 = _D_0_ibuffer[_II_*_III_*_JJJ_*_KKK_-1];
     if (s.drain_signal)
     {
      DPRINTF ("D(%d,%d): write_channel _D_0_channel data: %f\n",_198,_199,_231);
      write_channel_intel(_D_0_channel[_198][_199], _231);
     } // if _217
 
 
     float sum = 0;

     #pragma unroll
     for (int lll = 0; lll < _LLL_; lll++) {
       sum += _rch_0[lll]*_rch_1[lll];
     }

     #pragma unroll
     for (int x = (_II_*_III_*_JJJ_*_KKK_-1); x >= 1; x--)
        _D_0_ibuffer[x] = _D_0_ibuffer[x-1];

     _D_0_ibuffer[0] = s.drain_signal ? sum : (_231 + sum); 
    
     DPRINTF("D(%d,%d): D.ibuffer: %f drain: %d\n",_198, _199, _D_0_ibuffer[0], s.drain_signal); 
  
  } // kernel kernel_D_channel
}

channel float _C_drainer_0_inter_channel[_KK_] __attribute__((depth(_III_*_JJJ_*_KKK_))) ;
channel float _C_drainer_0_channel[_JJ_-1][_KK_] __attribute__((depth(0))) ;

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(_JJ_, _KK_, 1)))
__kernel void kernel_C_drainer_channel(
)
{
 int _272 = get_compute_id(0);
 int _273 = get_compute_id(1);
 while (1)
 {
   int _274 = _JJ_ - _272;
   for (int _C_drainer_s0_t__gather_elem_ = 0; _C_drainer_s0_t__gather_elem_ < 0 + _274; _C_drainer_s0_t__gather_elem_++)
   {
    float _275;
    bool _276 = _C_drainer_s0_t__gather_elem_ == 0;
    if (_276)
    {
     float _277;
     DPRINTF ("C_drainer(%d,%d): To read_channel _D_0_channel\n",_272,_273);
     _277 = read_channel_intel(_D_0_channel[_272][_273]);
     DPRINTF ("C_drainer(%d,%d): read_channel _D_0_channel data: %f\n",_272,_273,_277);
     _275 = _277;
    } // if _276
    else
    {
     float _279;
     DPRINTF ("C_drainer(%d,%d): To read_channel _C_drainer_0_channel\n",_272,_273);
     _279 = read_channel_intel(_C_drainer_0_channel[_272][_273]);
     DPRINTF ("C_drainer(%d,%d): read_channel _C_drainer_0_channel data: %f\n",_272,_273,_279);
     _275 = _279;
    } // if _276 else
    DPRINTF ("C_drainer(%d,%d): write_channel _C_drainer_0_channel data: %f\n",_272,_273,_275);
    if (_272 == 0)
      write_channel_intel(_C_drainer_0_inter_channel[_273], _275);
    else
      write_channel_intel(_C_drainer_0_channel[_272-1][_273], _275);
   } // for _C_drainer_s0_t__gather_elem_
 } // while _C_drainer_s0_i_i_infinite
} // kernel kernel_C_drainer_channel

typedef struct _out_vec { float data[_KK_]; } outvec;

channel outvec _C_collector_0_inter_channel __attribute__((depth(_III_*_JJJ_*_KKK_))) ;
channel outvec _C_collector_0_channel[_KK_-1] __attribute__((depth(0))) ;
// Address spaces for kernel_C_collector_channel

__attribute__((max_global_work_dim(0)))__attribute__((autorun))

__attribute__((num_compute_units(_KK_, 1, 1)))
__kernel void kernel_C_collector_channel(
)
{
 int _280 = get_compute_id(0);
;
 while (1)
 {
   int _281 = _KK_ - _280;

     float _284;
     DPRINTF ("C_collector(%d): To read_channel _C_drainer_0_channel\n",_280);
     _284 = read_channel_intel(_C_drainer_0_inter_channel[_280]);
     DPRINTF ("C_collector(%d): read_channel _C_drainer_0_channel data: %f\n",_280,_284);

     outvec _286;
     if (_280 != (_KK_-1)) {
       DPRINTF ("C_collector(%d): To read_channel _C_collector_0_channel\n",_280);
       _286 = read_channel_intel(_C_collector_0_channel[_280]);
       DPRINTF ("C_collector(%d): read_channel _C_collector_0_channel data: %f\n",_280,_286);
     }

     outvec out;
     #pragma unroll
     for (int x = (_KK_-1); x > _280; x--)
        out.data[x] = _286.data[x];
     out.data[_280] = _284;

    DPRINTF ("C_collector(%d): write_channel _C_collector_0_channel data: %f\n",_280, out.data[0]);
    if (_280 == 0)
      write_channel_intel(_C_collector_0_inter_channel, out);
    else
      write_channel_intel(_C_collector_0_channel[_280-1], out);
 } // while _C_collector_s0_i_i_infinite
} // kernel kernel_C_collector_channel
// Address spaces for kernel_C_unloader
#define __address_space__C_unloader_mem_channel __global
__kernel void kernel_C_unloader(
 const int _C_deserializer_extent_0,
 const int _C_deserializer_extent_1,
 const int _C_deserializer_extent_2,
 __address_space__C_unloader_mem_channel outvec *_C_unloader_mem_channel,
 __address_space___shared int16* __shared)
{
 int addr = 0;
 while(addr!=(_C_deserializer_extent_0*_C_deserializer_extent_1*_C_deserializer_extent_2*_II_*_JJ_*_III_*_JJJ_*_KKK_)) {
     outvec in;
     DPRINTF ("C_unloader: To read_channel _C_collector_0_channel\n");
     in = read_channel_intel(_C_collector_0_inter_channel);
     DPRINTF ("C_unloader: read_channel _C_collector_0_channel data: %f\n",in.data[0]);
     _C_unloader_mem_channel[addr] = in;
     addr++;
 }
} // kernel kernel_C_unloader
#undef __address_space__C_unloader_mem_channel
