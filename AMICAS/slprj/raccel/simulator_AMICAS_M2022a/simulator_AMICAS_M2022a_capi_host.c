#include "simulator_AMICAS_M2022a_capi_host.h"
static simulator_AMICAS_M2022a_host_DataMapInfo_T root;
static int initialized = 0;
__declspec( dllexport ) rtwCAPI_ModelMappingInfo *getRootMappingInfo()
{
    if (initialized == 0) {
        initialized = 1;
        simulator_AMICAS_M2022a_host_InitializeDataMapInfo(&(root), "simulator_AMICAS_M2022a");
    }
    return &root.mmi;
}

rtwCAPI_ModelMappingInfo *mexFunction(){return(getRootMappingInfo());}
