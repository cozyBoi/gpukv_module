#ifndef __KEY_P2P_H
#define __KEY_P2P_H

#include <linux/types.h>


#define KEY_IOCTL	0xBA

struct I_KEY_PIN_PARAM{
	__u64 addr;
	__u64 size;
	__u64 start_addr;
	__u64 pin_size;
	__u32 id;
};
struct I_KEY_UNPIN_PARAM{
	__u32 id;
};
struct I_KEY_P2P_GET_PARAM{
	__u32 id;
	__u64 bar_addr;
	__u64 size;
	__u64 data_offset;
};
struct I_KEY_P2P_PARAM{
	__u32 id;
	__u64 bar_addr;
	__u64 size;
	__u64 data_offset;
};

#define I_KEY_PIN_BUF _IOWR(KEY_IOCTL, 1, struct I_KEY_PIN_PARAM *)
#define I_KEY_UNPIN_BUF _IOWR(KEY_IOCTL, 2, struct I_KEY_UNPIN_PARAM *)
#define I_KEY_P2P_GET _IOWR(KEY_IOCTL, 3, struct I_KEY_P2P_PARAM *)
#define I_KEY_P2P_PUT _IOWR(KEY_IOCTL, 4, struct I_KIN_P2P_PARAM *)
#define I_KEY_ASYNC _IOWR(KEY_IOCTL, 5, unsigned long )
#define I_KEY_ASYNC_CLOSE _IOWR(KEY_IOCTL, 6, unsigned long )


#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    ((__u64)1 << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

#endif


