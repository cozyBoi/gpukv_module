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

//added
struct nvme_dev {
	struct list_head node;
	struct nvme_queue **queues;
	struct request_queue *admin_q;
	struct blk_mq_tag_set tagset;
	struct blk_mq_tag_set admin_tagset;
	u32 __iomem *dbs;
	struct device *dev;
	struct pci_dev *pci_dev;
	struct dma_pool *prp_page_pool;
	struct dma_pool *prp_small_pool;
	int instance;
	unsigned queue_count;
	unsigned online_queues;
	unsigned max_qid;
	int q_depth;
	u32 db_stride;
	void __iomem *bar;
	u32 ctrl_config;
	struct msix_entry *entry;
	//struct nvme_bar __iomem *bar;
	struct list_head namespaces;
	struct kref kref;
	struct device *device;
	work_func_t reset_workfn;
	struct work_struct reset_work;
	struct work_struct remove_work;
	struct work_struct probe_work;
	struct timer_list watchdog_timer;
	struct mutex shutdown_lock;
	bool subsystem;
	void __iomem *cmb;
	pci_bus_addr_t cmb_bus_addr;
	u64 cmb_size;
	u32 cmbsz;
	u32 cmbloc;
	struct nvme_ctrl ctrl;
	struct completion ioq_wait;
	char name[12];
	char serial[20];
	char model[40];
	char firmware_rev[8];
	u32 max_hw_sectors;
	u32 stripe_size;
	u32 page_size;
	u16 oncs;
	u16 abort_limit;
	u8 event_limit;
	u8 vwc;
};

struct nvme_iod {
	struct nvme_queue *nvmeq;
	int aborted;
	int npages;		/* In the PRP list. 0 means small pool in use */
	int nents;		/* Used in scatterlist */
	int length;		/* Of data, in bytes */
	dma_addr_t first_dma;
	struct scatterlist meta_sg; /* metadata requires single contiguous buffer */
	struct scatterlist *sg;
	struct scatterlist inline_sg[0];
};


struct nvme_ns {
	struct list_head list;

	struct nvme_ctrl *ctrl;
	struct request_queue *queue;
	struct gendisk *disk;
	struct nvm_dev *ndev;
	struct kref kref;
	int instance;

	u8 eui[8];
	u8 uuid[16];

	unsigned ns_id;
	int lba_shift;
	u16 ms;
	bool ext;
	u8 pi_type;
	unsigned long flags;

#define NVME_NS_REMOVING 0
#define NVME_NS_DEAD     1

	u64 mode_select_num_blocks;
	u32 mode_select_block_len;

	struct nvme_dev *dev;
};

#endif


