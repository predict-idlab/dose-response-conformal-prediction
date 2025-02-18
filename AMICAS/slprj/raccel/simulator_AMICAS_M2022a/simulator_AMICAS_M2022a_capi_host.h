#ifndef simulator_AMICAS_M2022a_cap_host_h__
#define simulator_AMICAS_M2022a_cap_host_h__
#ifdef HOST_CAPI_BUILD
#include "rtw_capi.h"
#include "rtw_modelmap_simtarget.h"
typedef struct { rtwCAPI_ModelMappingInfo mmi ; }
simulator_AMICAS_M2022a_host_DataMapInfo_T ;
#ifdef __cplusplus
extern "C" {
#endif
void simulator_AMICAS_M2022a_host_InitializeDataMapInfo ( simulator_AMICAS_M2022a_host_DataMapInfo_T * dataMap , const char * path ) ;
#ifdef __cplusplus
}
#endif
#endif
#endif
