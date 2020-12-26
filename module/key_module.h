#ifndef __KEY_MODULE_H
#define __KEY_MODULE_H

#define NAME "key_value_module"
#define NAME_CLASS "key_value"
#define MASK_DU (((u64)1<<32)-1)
#define MAX_FILENAME 100

#define LBA_SHIFT 9

#define GET 0x02
#define PUT 0x01
//#define GET 0xA1
//#define PUT 0xA0

#define MODE_KEY 0
#define MODE_P2P 1

// for boundary alignment requirement
#define GPU_BOUND_SHIFT   16
#define GPU_BOUND_SIZE    ((u64)1 << GPU_BOUND_SHIFT)
#define GPU_BOUND_OFFSET  (GPU_BOUND_SIZE-1)
#define GPU_BOUND_MASK    (~GPU_BOUND_OFFSET)

#define ID_MASK ((1UL<<32)-1)
#ifndef random_get_entropy
#define random_get_entropy()	get_cycles()
#endif

#if defined(CONFIG_X86_64) || defined(CONFIG_X86_32)
#define get_tsc_khz() cpu_khz // tsc_khz
#elif defined(CONFIG_PPC64)
#define get_tsc_khz() (get_cycles()/1000)
#else
#error "X86_64/32 or PPC64 is required"
#endif
#define MIN(a,b)((a<b)? a:b )
#define MAX(a,b)((a>b)? a:b )
#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

//ifndef RW_IPC_SIZE
//#define RW_IPC_SIZE 1024

  /////////////////////////////GPU structure
typedef struct _key_page_table {
	//struct list_head node;
	u32 id;
	u64 offset;
	u64 length;
	u64 start_addr;
	u64 pin_size;
	u64 p2p_token;
	u32 va_space;
	u32 page_size;
	nvidia_p2p_page_table_t* page_table;
	int call_flag;
}key_page_table;

typedef struct key_value_operations {
	long (*open) (const char __user*);
	long (*close) (const char __user*);
    long (*key_op) (int , unsigned long , char __user *, unsigned int );
	long (*key_p2p_op) (int, unsigned long, unsigned long, unsigned int);
	long(*key_batch_op) (unsigned long, int,int);
	long (*map_p2p_pages)(unsigned long, unsigned long,unsigned long ,unsigned );
}KEY_OPERATIONS;


typedef struct _complete_list {
//	__u32 result;
	volatile int status;
	volatile unsigned long result_offset;
	volatile unsigned long complete_offset;
	volatile unsigned int length;
	volatile unsigned int mode;
//	int num;
}Complete_list;

typedef struct __batch_list{
	spinlock_t lock;
	int cnt,end;
}Batch_list;
typedef struct _process_list{
	spinlock_t lock;
	int f,r,size;
	int *node;
}Process_list;

struct batch_cmd_info {
	struct request *req;
	Complete_list *res;
	Process_list *poll;
	int num;
	///////////////////////////////////////////
	struct task_struct *task;
	Batch_list *end;
};
typedef struct _command_list {
	struct nvme_command c;
	struct batch_cmd_info cmdinfo;
	struct nvme_iod *iod;
	unsigned long addr;
	unsigned int length;
	unsigned timeout;
	int num;
}Command_list;

typedef struct _user_list {
	unsigned long key;
	unsigned long offset;
	unsigned int size;
	unsigned int mode;
	int num;
	unsigned long result_offset;
	unsigned long complete_offset;
}User_list;

#define BUF_NEXT(a,num) ((a+1)%num)

/////////////////////////////////////////
long Key_value_open(const char __user* filename);
long Key_value_close(const char __user* filename);
void print_error(int x);

static int inter_open(struct inode *, struct file *);
static int inter_release(struct inode *, struct file *);
static long inter_ioctl(struct file *inode, unsigned int cmd, unsigned long arg);

static int submit_key_cmd(struct nvme_passthru_cmd *cmd,int mode);


long key_op(int mode, unsigned long key, char __user *buf, unsigned int size);
long key_p2p_op(int mode, unsigned long key, unsigned long offset,unsigned int size);
int Complete_Process(void *data);
long key_batch_command(unsigned long _buf, int len,int mode);

////////GPU function
long Key_map_p2p_pages(unsigned long, unsigned long, unsigned long, unsigned);
int key_pin_buf(void __user *_param);
int key_unpin_buf(void __user *_param);
void get_p2p_free_callback(void *data);
////////////////////////////////////////////////////




#endif


