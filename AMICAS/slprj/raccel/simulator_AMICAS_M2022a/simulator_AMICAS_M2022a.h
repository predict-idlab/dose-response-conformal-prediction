#ifndef simulator_AMICAS_M2022a_h_
#define simulator_AMICAS_M2022a_h_
#ifndef simulator_AMICAS_M2022a_COMMON_INCLUDES_
#define simulator_AMICAS_M2022a_COMMON_INCLUDES_
#include <stdlib.h>
#include "sl_AsyncioQueue/AsyncioQueueCAPI.h"
#include "rtwtypes.h"
#include "sigstream_rtw.h"
#include "simtarget/slSimTgtSigstreamRTW.h"
#include "simtarget/slSimTgtSlioCoreRTW.h"
#include "simtarget/slSimTgtSlioClientsRTW.h"
#include "simtarget/slSimTgtSlioSdiRTW.h"
#include "simstruc.h"
#include "fixedpoint.h"
#include "raccel.h"
#include "slsv_diagnostic_codegen_c_api.h"
#include "rt_logging_simtarget.h"
#include "rt_nonfinite.h"
#include "math.h"
#include "dt_info.h"
#include "ext_work.h"
#endif
#include "simulator_AMICAS_M2022a_types.h"
#include "mwmathutil.h"
#include <stddef.h>
#include "rtw_modelmap_simtarget.h"
#include "rt_defines.h"
#include <string.h>
#define MODEL_NAME simulator_AMICAS_M2022a
#define NSAMPLE_TIMES (4) 
#define NINPUTS (0)       
#define NOUTPUTS (0)     
#define NBLOCKIO (20) 
#define NUM_ZC_EVENTS (0) 
#ifndef NCSTATES
#define NCSTATES (39)   
#elif NCSTATES != 39
#error Invalid specification of NCSTATES defined in compiler command
#endif
#ifndef rtmGetDataMapInfo
#define rtmGetDataMapInfo(rtm) (*rt_dataMapInfoPtr)
#endif
#ifndef rtmSetDataMapInfo
#define rtmSetDataMapInfo(rtm, val) (rt_dataMapInfoPtr = &val)
#endif
#ifndef IN_RACCEL_MAIN
#endif
typedef struct { real_T glp1tbzcj5 ; real_T do2ggtt1yj ; real_T hkfwlkn44z ;
real_T k2ovkquskq ; real_T ifrgwg0hqk ; real_T cllkm5kpwc ; real_T eu4yxwxe0f
; real_T lqothvc25a ; real_T j023pd3utu ; real_T hlohoxodx1 ; real_T
i5issijamp ; real_T hrox2xbs04 ; real_T j24ks1lf4w ; real_T gkhsfxl1z2 ;
real_T gg2ss2zf1k ; real_T duxrmorn55 ; real_T odngtpipys ; real_T pojf1hfliu
; real_T p4oo54b4y4 ; real_T hbqjo003ri ; } B ; typedef struct { struct {
void * AQHandles ; } kurh2j4r11 ; struct { void * AQHandles ; } mo0b2dlk3c ;
struct { void * AQHandles ; } hhjg0tswq3 ; struct { void * AQHandles ; }
jwlvqglcx5 ; struct { void * AQHandles ; } pexlwytvcb ; struct { void *
AQHandles ; } ks21q1oe0m ; struct { void * TimePtr ; void * DataPtr ; void *
RSimInfoPtr ; } kquv5edfh5 ; struct { void * TimePtr ; void * DataPtr ; void
* RSimInfoPtr ; } ck2c4rbpy4 ; struct { void * TimePtr ; void * DataPtr ;
void * RSimInfoPtr ; } jn41mxwh4n ; struct { void * TimePtr ; void * DataPtr
; void * RSimInfoPtr ; } l00cdg23mz ; struct { void * TimePtr ; void *
DataPtr ; void * RSimInfoPtr ; } ijhh2juliq ; struct { void * TimePtr ; void
* DataPtr ; void * RSimInfoPtr ; } kf2drr35ex ; struct { void * TimePtr ;
void * DataPtr ; void * RSimInfoPtr ; } kvwfqgfdli ; struct { void * TimePtr
; void * DataPtr ; void * RSimInfoPtr ; } pt4k0lry0r ; struct { void *
TimePtr ; void * DataPtr ; void * RSimInfoPtr ; } ccqit54xab ; struct { void
* TimePtr ; void * DataPtr ; void * RSimInfoPtr ; } oxs2h0xie1 ; struct {
void * TimePtr ; void * DataPtr ; void * RSimInfoPtr ; } pgpv2tqduz ; struct
{ void * TimePtr ; void * DataPtr ; void * RSimInfoPtr ; } igxx1lod55 ;
struct { void * TimePtr ; void * DataPtr ; void * RSimInfoPtr ; } fmgvpmawvu
; struct { void * TimePtr ; void * DataPtr ; void * RSimInfoPtr ; }
pfyaolnavt ; struct { void * TimePtr ; void * DataPtr ; void * RSimInfoPtr ;
} lzv3r0nmzz ; struct { void * TimePtr ; void * DataPtr ; void * RSimInfoPtr
; } defpspt1xf ; struct { void * TimePtr ; void * DataPtr ; void *
RSimInfoPtr ; } ayyzzi4kcp ; struct { void * TimePtr ; void * DataPtr ; void
* RSimInfoPtr ; } kccsld1twl ; struct { int_T PrevIndex ; } ludljk1mkt ;
struct { int_T PrevIndex ; } mimikdlvla ; struct { int_T PrevIndex ; }
azbmwzhuxe ; struct { int_T PrevIndex ; } j5a3pxorvd ; struct { int_T
PrevIndex ; } mpjornappt ; struct { int_T PrevIndex ; } h1xxcnwaw1 ; struct {
int_T PrevIndex ; } ayogcju54x ; struct { int_T PrevIndex ; } a2dupmilus ;
struct { int_T PrevIndex ; } pjtji2hhge ; struct { int_T PrevIndex ; }
iicsswm23u ; struct { int_T PrevIndex ; } cfr12oq5te ; struct { int_T
PrevIndex ; } m3yxmkxthp ; struct { int_T PrevIndex ; } oevdd11tc0 ; struct {
int_T PrevIndex ; } l5nxfgntsh ; struct { int_T PrevIndex ; } j3zy2npopo ;
struct { int_T PrevIndex ; } hfvzo3ediv ; struct { int_T PrevIndex ; }
lxcs24fexb ; struct { int_T PrevIndex ; } m1rn3czb2l ; } DW ; typedef struct
{ real_T g25adnvh4d ; real_T lwm5o4pybs ; real_T ne2xgqomhy ; real_T
c21hw23qxv ; real_T cdb2z14mer ; real_T cppalivsmr ; real_T can0xnfipc [ 5 ]
; real_T pqvalricfe [ 5 ] ; real_T bxwozln2gn ; real_T ewrbpvzi44 ; real_T
phzeerfedw [ 5 ] ; real_T eup3ihr1o0 [ 5 ] ; real_T j0dthfkwwn [ 3 ] ; real_T
ezeefipbde ; real_T hbapgsce04 [ 3 ] ; real_T odpdqdvgzq [ 3 ] ; real_T
kfks24suh2 ; } X ; typedef struct { real_T g25adnvh4d ; real_T lwm5o4pybs ;
real_T ne2xgqomhy ; real_T c21hw23qxv ; real_T cdb2z14mer ; real_T cppalivsmr
; real_T can0xnfipc [ 5 ] ; real_T pqvalricfe [ 5 ] ; real_T bxwozln2gn ;
real_T ewrbpvzi44 ; real_T phzeerfedw [ 5 ] ; real_T eup3ihr1o0 [ 5 ] ;
real_T j0dthfkwwn [ 3 ] ; real_T ezeefipbde ; real_T hbapgsce04 [ 3 ] ;
real_T odpdqdvgzq [ 3 ] ; real_T kfks24suh2 ; } XDot ; typedef struct {
boolean_T g25adnvh4d ; boolean_T lwm5o4pybs ; boolean_T ne2xgqomhy ;
boolean_T c21hw23qxv ; boolean_T cdb2z14mer ; boolean_T cppalivsmr ;
boolean_T can0xnfipc [ 5 ] ; boolean_T pqvalricfe [ 5 ] ; boolean_T
bxwozln2gn ; boolean_T ewrbpvzi44 ; boolean_T phzeerfedw [ 5 ] ; boolean_T
eup3ihr1o0 [ 5 ] ; boolean_T j0dthfkwwn [ 3 ] ; boolean_T ezeefipbde ;
boolean_T hbapgsce04 [ 3 ] ; boolean_T odpdqdvgzq [ 3 ] ; boolean_T
kfks24suh2 ; } XDis ; typedef struct { rtwCAPI_ModelMappingInfo mmi ; }
DataMapInfo ; struct P_ { real_T Anest_loop ; real_T Cobasis ; real_T
Dist_type ; real_T MAPbasis ; real_T RSM_type ; real_T alpha_G ; real_T
alpha_RG ; real_T Internal_A_pr ; real_T Internal_B_pr ; real_T Internal_C_pr
; real_T Internal_InitialCondition ; real_T TransferFcn_A ; real_T
TransferFcn_C ; real_T Internal_A_pr_bxw4oadisy ; real_T
Internal_B_pr_acd2u0mx1v ; real_T Internal_C_pr_cbzmhk4ru4 ; real_T
Internal_InitialCondition_flhq4j1c44 ; real_T Saturation_UpperSat ; real_T
Saturation_LowerSat ; real_T Internal_A_pr_oxglitxgdi ; real_T
Internal_B_pr_ixcvq1t1i0 ; real_T Internal_C_pr_a30ee2albx ; real_T
Internal_InitialCondition_plhlb04yjz ; real_T Saturation_UpperSat_kjur3awdxo
; real_T Saturation_LowerSat_i3sbtd4ihe ; real_T
Saturation_UpperSat_ce4kz1xeh2 ; real_T Saturation_LowerSat_j0oq522fxt ;
real_T TransferFcn_A_c31gaddzxu ; real_T TransferFcn_C_pwm5fjxov3 ; real_T
TransferFcn_A_cgq1qqeziw ; real_T TransferFcn_C_kg3euht54j ; real_T
Internal_A_pr_nkheacrqmt [ 9 ] ; real_T Internal_B_pr_l20xfqn4io ; real_T
Internal_C_pr_ihduuzxooi ; real_T Internal_InitialCondition_f013nqgro1 ;
real_T Gain1_Gain ; real_T Internal_A_pr_n4kfo5ljxc [ 9 ] ; real_T
Internal_B_pr_l0u03c0wog ; real_T Internal_C_pr_coj5mfz3fg ; real_T
Internal_InitialCondition_dascwzc0ku ; real_T Gain2_Gain ; real_T
Saturation7_UpperSat ; real_T Saturation7_LowerSat ; real_T
TransferFcn_A_eiu4to52sp ; real_T TransferFcn_C_ccp0mwtknu ; real_T
TransferFcn_A_fdwj0zhm3r ; real_T TransferFcn_C_pcze5nejeb ; real_T
Internal_A_pr_gtcc2nztln [ 9 ] ; real_T Internal_B_pr_e0vgcru5yv ; real_T
Internal_C_pr_nmldud5anh ; real_T Internal_InitialCondition_fyztitczqh ;
real_T Internal_A_pr_ojosqlrsev [ 9 ] ; real_T Internal_B_pr_isexhvxkqn ;
real_T Internal_C_pr_hhrpjg032d ; real_T Internal_InitialCondition_eoiqws2jvx
; real_T Saturation8_UpperSat ; real_T Saturation8_LowerSat ; real_T
Internal_A_pr_i2ltighkdc [ 5 ] ; real_T Internal_B_pr_mlvsl1pdjw ; real_T
Internal_C_pr_puxizhhqsn ; real_T Internal_InitialCondition_nizog25s2h ;
real_T Saturation9_UpperSat ; real_T Saturation9_LowerSat ; real_T
Internal_A_pr_m4zhmj0l23 ; real_T Internal_B_pr_aopwchlqyq ; real_T
Internal_C_pr_puewk12or5 ; real_T Internal_InitialCondition_oekgjvagkb ;
real_T Saturation6_UpperSat ; real_T Saturation6_LowerSat ; real_T
Atracuriuminput_Time0 [ 310 ] ; real_T Atracuriuminput_Data0 [ 310 ] ; real_T
Dopamineinput_Time0 [ 310 ] ; real_T Dopamineinput_Data0 [ 310 ] ; real_T
Propofolinput_Time0 [ 310 ] ; real_T Propofolinput_Data0 [ 310 ] ; real_T
Remifentanilinput_Time0 [ 310 ] ; real_T Remifentanilinput_Data0 [ 310 ] ;
real_T SNPinput_Time0 [ 310 ] ; real_T SNPinput_Data0 [ 310 ] ; real_T
Saturation2_UpperSat ; real_T Saturation2_LowerSat ; real_T _Gain ; real_T
FromWorkspace1_Time0 [ 1051 ] ; real_T FromWorkspace1_Data0 [ 1051 ] ; real_T
FromWorkspace_Time0 [ 1051 ] ; real_T FromWorkspace_Data0 [ 1051 ] ; real_T
Circumcision_Time0 [ 318 ] ; real_T Circumcision_Data0 [ 318 ] ; real_T
Nodisturbance_Time0 [ 500 ] ; real_T Nodisturbance_Data0 [ 500 ] ; real_T
KneeArthroscopy_Time0 [ 579 ] ; real_T KneeArthroscopy_Data0 [ 579 ] ; real_T
EpididymalCystectomy_Time0 [ 468 ] ; real_T EpididymalCystectomy_Data0 [ 468
] ; real_T ScrotalExploration_Time0 [ 756 ] ; real_T ScrotalExploration_Data0
[ 756 ] ; real_T Myomectomy_Time0 [ 1536 ] ; real_T Myomectomy_Data0 [ 1536 ]
; real_T Urethroplasty_Time0 [ 1824 ] ; real_T Urethroplasty_Data0 [ 1824 ] ;
real_T Mammoplasty_Time0 [ 1344 ] ; real_T Mammoplasty_Data0 [ 1344 ] ;
real_T OpenCholecysectomy_Time0 [ 1224 ] ; real_T OpenCholecysectomy_Data0 [
1224 ] ; real_T Vasovasostomy_Time0 [ 2160 ] ; real_T Vasovasostomy_Data0 [
2160 ] ; real_T Hysterectomy_Time0 [ 1428 ] ; real_T Hysterectomy_Data0 [
1428 ] ; real_T Internal_A_pr_kx5sallgve [ 7 ] ; real_T
Internal_B_pr_hfxj0kvrnc ; real_T Internal_C_pr_av2jlvpum0 ; real_T
Internal_InitialCondition_lpky2t20lr ; real_T Internal_A_pr_m25t4b112u [ 7 ]
; real_T Internal_B_pr_kda05yf4jy ; real_T Internal_C_pr_klda0set0f ; real_T
Internal_InitialCondition_lwwah2kv3v ; real_T Internal_A_pr_puchlpq5ne ;
real_T Internal_B_pr_d3jkwmc1kb ; real_T Internal_C_pr_cnq1if3xbe ; real_T
Internal_InitialCondition_cbt5hmxici ; real_T Saturation1_UpperSat ; real_T
Saturation1_LowerSat ; real_T Saturation3_UpperSat ; real_T
Saturation3_LowerSat ; real_T Saturation4_UpperSat ; real_T
Saturation4_LowerSat ; real_T patientweight_Gain ; real_T
Saturation5_UpperSat ; real_T Saturation5_LowerSat ; uint32_T Internal_A_ir ;
uint32_T Internal_A_jc [ 2 ] ; uint32_T Internal_B_ir ; uint32_T
Internal_B_jc [ 2 ] ; uint32_T Internal_C_ir ; uint32_T Internal_C_jc [ 2 ] ;
uint32_T Internal_A_ir_hf5tox2xk5 ; uint32_T Internal_A_jc_ogadgmqmoo [ 2 ] ;
uint32_T Internal_B_ir_oqjkicywhm ; uint32_T Internal_B_jc_bbjjxvfjvn [ 2 ] ;
uint32_T Internal_C_ir_akgglybrpk ; uint32_T Internal_C_jc_jfpd0pnvih [ 2 ] ;
uint32_T Internal_A_ir_dgoldd154g ; uint32_T Internal_A_jc_pge3o4iqct [ 2 ] ;
uint32_T Internal_B_ir_k0filihusw ; uint32_T Internal_B_jc_onaxgo32ml [ 2 ] ;
uint32_T Internal_C_ir_iyqg3glnsf ; uint32_T Internal_C_jc_llex4eusi2 [ 2 ] ;
uint32_T Internal_A_ir_kh1ahh4kw5 [ 9 ] ; uint32_T Internal_A_jc_cd0lveyter [
6 ] ; uint32_T Internal_B_ir_nrfq1m3t4y ; uint32_T Internal_B_jc_ac1tk1l1cm [
2 ] ; uint32_T Internal_C_ir_o11oddr1av ; uint32_T Internal_C_jc_d2vn1pgejl [
6 ] ; uint32_T Internal_A_ir_lq4j4xfz2n [ 9 ] ; uint32_T
Internal_A_jc_acfygnacds [ 6 ] ; uint32_T Internal_B_ir_ijyvpjmjsq ; uint32_T
Internal_B_jc_aajyzu2cah [ 2 ] ; uint32_T Internal_C_ir_ky5xf0i1px ; uint32_T
Internal_C_jc_galqf04ynt [ 6 ] ; uint32_T Internal_A_ir_aak1secaho [ 9 ] ;
uint32_T Internal_A_jc_kanybgxs0e [ 6 ] ; uint32_T Internal_B_ir_d0aaprxg0k ;
uint32_T Internal_B_jc_gbxmowzmqz [ 2 ] ; uint32_T Internal_C_ir_ibqb01xp5c ;
uint32_T Internal_C_jc_dwqmfgpi2b [ 6 ] ; uint32_T Internal_A_ir_dpjnaf3xwz [
9 ] ; uint32_T Internal_A_jc_n321zog0ex [ 6 ] ; uint32_T
Internal_B_ir_kkrwmhr25e ; uint32_T Internal_B_jc_bmh4dtrh0s [ 2 ] ; uint32_T
Internal_C_ir_mjslb5eyrb ; uint32_T Internal_C_jc_au0h0wd2zz [ 6 ] ; uint32_T
Internal_A_ir_deilu3523u [ 5 ] ; uint32_T Internal_A_jc_nnhrvh4uhs [ 4 ] ;
uint32_T Internal_B_ir_bf5h52djfe ; uint32_T Internal_B_jc_g51vo4a3vx [ 2 ] ;
uint32_T Internal_C_ir_j0bboxekne ; uint32_T Internal_C_jc_ap3qqcgpmi [ 4 ] ;
uint32_T Internal_A_ir_myylvo2i45 ; uint32_T Internal_A_jc_ig2cutmpb4 [ 2 ] ;
uint32_T Internal_B_ir_f5rfza45d4 ; uint32_T Internal_B_jc_lblmejayek [ 2 ] ;
uint32_T Internal_C_ir_os2whdiotd ; uint32_T Internal_C_jc_pwb3pumdzw [ 2 ] ;
uint32_T Internal_A_ir_ijulsdwdmi [ 7 ] ; uint32_T Internal_A_jc_p4ci0hu5qn [
4 ] ; uint32_T Internal_B_ir_jusrbow3ct ; uint32_T Internal_B_jc_ebzizneqnl [
2 ] ; uint32_T Internal_C_ir_hj1uxzp1ep ; uint32_T Internal_C_jc_cs0yy5eb3e [
4 ] ; uint32_T Internal_A_ir_oujtltlxgd [ 7 ] ; uint32_T
Internal_A_jc_it04zokaos [ 4 ] ; uint32_T Internal_B_ir_ks0p0yho3p ; uint32_T
Internal_B_jc_cecpce5hwt [ 2 ] ; uint32_T Internal_C_ir_d33tkca5cs ; uint32_T
Internal_C_jc_hhgn2nuvph [ 4 ] ; uint32_T Internal_A_ir_oik1appi1t ; uint32_T
Internal_A_jc_dt4g4ekvp3 [ 2 ] ; uint32_T Internal_B_ir_ov1s2pbumi ; uint32_T
Internal_B_jc_h12ogkfonh [ 2 ] ; uint32_T Internal_C_ir_pfmozql5e3 ; uint32_T
Internal_C_jc_f3vgqu5len [ 2 ] ; } ; extern const char_T *
RT_MEMORY_ALLOCATION_ERROR ; extern B rtB ; extern X rtX ; extern DW rtDW ;
extern P rtP ; extern mxArray * mr_simulator_AMICAS_M2022a_GetDWork ( ) ;
extern void mr_simulator_AMICAS_M2022a_SetDWork ( const mxArray * ssDW ) ;
extern mxArray * mr_simulator_AMICAS_M2022a_GetSimStateDisallowedBlocks ( ) ;
extern const rtwCAPI_ModelMappingStaticInfo *
simulator_AMICAS_M2022a_GetCAPIStaticMap ( void ) ; extern SimStruct * const
rtS ; extern DataMapInfo * rt_dataMapInfoPtr ; extern
rtwCAPI_ModelMappingInfo * rt_modelMapInfoPtr ; void MdlOutputs ( int_T tid )
; void MdlOutputsParameterSampleTime ( int_T tid ) ; void MdlUpdate ( int_T
tid ) ; void MdlTerminate ( void ) ; void MdlInitializeSizes ( void ) ; void
MdlInitializeSampleTimes ( void ) ; SimStruct * raccel_register_model ( ssExecutionInfo * executionInfo ) ;
#endif
