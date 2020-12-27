/* 
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */

/* 
 * This expermental software is provided AS IS.
 * Feel free to use/modify/distribute,
 * If used, please retain this disclaimer and cite
 * "GPUfs: Integrating a file system with GPUs",
 * M Silberstein,B Ford,I Keidar,E Witchel
 * ASPLOS13, March 2013, Houston,USA
 */

#ifndef FS_CONSTANTS
#define FS_CONSTANTS

// needed for mmap return codes
#include <sys/mman.h>

//** CPU IPC R/W TABLE
//#define RW_IPC_SIZE 1024

#define MAX_BLOCK 900


#define IPC_MGR_EMPTY 0
#define IPC_MGR_BUSY 1

//** CPU IPC 
#define CPU_IPC_EMPTY 0
#define CPU_IPC_PENDING 1
#define CPU_IPC_IN_PROCESS 2
#define CPU_IPC_COMPLETE 3
//#define CPU_IPC_HOST 4
//#define CPU_IPC_IN_PROCESS 2
//#define CPU_IPC_READY 3
#define MODE_BLOCK 0
#define MODE_WARP 1
#define MODE_THREAD 2

//** ERROR CODES **//
#define E_FSTABLE_FULL -1
#define E_IPC_OPEN_ERROR -2

#define RW_IPC_GET 2
#define RW_IPC_PUT 1

//api
#define GET 0x02
#define PUT 0x01
//#define GET 0xA1
//#define PUT 0xA0




typedef unsigned char uchar;

#endif
