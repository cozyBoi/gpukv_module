//for linux version 4.17.6

#include <linux/module.h>  //needed by all modules
#include <linux/kernel.h> //needed for KERN_INFO
#include <linux/spinlock.h>
#include <linux/bio.h>
#include <linux/vmalloc.h>
#include <linux/string.h> 
#include <linux/uaccess.h>
#include <linux/device.h>
#include <linux/fcntl.h>
#include <linux/timekeeping.h>
#include <linux/mutex.h>
#include <linux/kthread.h> //for kernel thread 
#include <linux/wait.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/smp.h>
//////////////////////////////
#include <linux/blkdev.h>
//#include <linux/nvme_ioctl.h>
#include <linux/nvme.h>
#include <linux/fs.h>
#include <linux/errno.h> 

#include <linux/io.h>
#include<linux/timex.h>
#include<linux/timer.h>
#include <linux/types.h>
#include<linux/string.h>
//#include<linux/jffies.h>
#include<linux/wait.h>
#include<linux/nvme_ioctl.h>
//needed page_mask header file
/////////////////////////
#include "nv-p2p.h"
#include "key-p2p.h"
#include "key_module.h"
#include "nvme.h"

/*
just do it
*/
#include <linux/aer.h>
#include <linux/bitops.h>
#include <linux/blkdev.h>
#include <linux/blk-mq.h>
#include <linux/blk-mq-pci.h>
#include <linux/cpu.h>
#include <linux/delay.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/genhd.h>
#include <linux/hdreg.h>
#include <linux/idr.h>
#include <linux/init.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/kdev_t.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/poison.h>
#include <linux/ptrace.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/t10-pi.h>
#include <linux/timer.h>
#include <linux/types.h>
#include <linux/io-64-nonatomic-lo-hi.h>
#include <asm/unaligned.h>
#include <linux/nvme_ioctl.h>
#include "nvme.h"

#define NVME_Q_DEPTH		1024
#define NVME_AQ_DEPTH		256
#define SQ_SIZE(depth)		(depth * sizeof(struct nvme_command))
#define CQ_SIZE(depth)		(depth * sizeof(struct nvme_completion))
		
/*
 * We handle AEN commands ourselves and don't even let the
 * block layer know about them.
 */
#define NVME_AQ_BLKMQ_DEPTH	(NVME_AQ_DEPTH - NVME_NR_AERS)

static int use_threaded_interrupts;
module_param(use_threaded_interrupts, int, 0);

static bool use_cmb_sqes = true;
module_param(use_cmb_sqes, bool, 0644);
MODULE_PARM_DESC(use_cmb_sqes, "use controller's memory buffer for I/O SQes");

static struct workqueue_struct *nvme_workq;

struct nvme_dev;
struct nvme_queue;

static int nvme_reset(struct nvme_dev *dev);
static void nvme_process_cq(struct nvme_queue *nvmeq);
static void nvme_dev_disable(struct nvme_dev *dev, bool shutdown);
//


//#include "fs_initializer.cu.h"
//
/*
extern nvme_iod*nvme_map_user_pages(struct nvme_dev *dev, int write, unsigned long addr, unsigned length);
extern bool nvme_setup_prps(struct nvme_dev *dev, struct request *req, int total_len);
extern int nvme_key_batch_command(struct block_device *bdev, unsigned long arg, int len,int mode);
extern void nvme_unmap_user_pages(struct nvme_dev *dev, int write, struct nvme_iod *iod);
extern void nvme_free_iod(struct nvme_dev *dev, struct request *req);*/
///////for key_value_open
static char* nvme_name = NULL;
static struct file* nvme_dev_f = NULL;
static unsigned int key_cmd_limit;
static struct block_device* nvme_bdev;
static struct nvme_dev *ns_dev;
static struct nvme_ns *ns;
static int shift_value;

///////ioctl
static struct file_operations inter_fops =
{
    .open = inter_open,
    .unlocked_ioctl = inter_ioctl,
    //  .locked_ioctl = inter_ioctl,
    .release = inter_release,
};
static struct class* dclass = NULL;
static struct device *device = NULL;
static int inter_major = 0;//, inter_minor=0;
///////////
static KEY_OPERATIONS key_ops =
{
    .open = Key_value_open,
    .close = Key_value_close,
    .key_op = key_op,
    .key_p2p_op = key_p2p_op,
    .key_batch_op=key_batch_command,
    .map_p2p_pages = Key_map_p2p_pages,
};
extern KEY_OPERATIONS* Key_op;
extern int Key_value_module;

///////////////
static key_page_table* key_table = NULL;

//////////////////////////////////////////
//for p2p mapping 
long Key_map_p2p_pages(unsigned long _iod,unsigned long _dev, unsigned long _off,unsigned length){
    // iod는 data in I/O를 의미!
    int i,of;
    struct nvme_iod *iod;
    struct scatterlist *sg;
    uint32_t page_size;
    __u64 offset=_off;
    int count,start,end;
    struct nvme_dev *dev=(struct nvme_dev *)_dev;
    nvidia_p2p_page_table_t* page_table=key_table->page_table;
    unsigned nprps;

//  if (addr & 3)
//      return ERR_PTR(-EINVAL);
    if (!length || length > INT_MAX - PAGE_SIZE)
    {
        printk("Key_map_p2p_pages | EINVAL! length: %u\n", length);
        return -EINVAL;
    }

    page_size=key_table->page_size;

    start=offset/page_size;
    end=(offset+length-1)/page_size;
    count=end-start+1;

    nprps=DIV_ROUND_UP(length + dev->page_size, dev->page_size);

    iod=kmalloc(sizeof(struct nvme_iod) + sizeof(__le64 *) * (DIV_ROUND_UP(8*nprps,PAGE_SIZE-8) )+sizeof(struct scatterlist)*count,GFP_KERNEL);

    if(iod){
        iod->private=(unsigned long)0;
        iod->offset=offsetof(struct nvme_iod,sg[count]);
        iod->npages=-1;
        iod->length=length;
        iod->nents=0;
        iod->first_dma=0;
    }
    //  iod_init(iod,length,count,(unsigned long)dev);
//  iod=__nvme_alloc_iod(count,length,dev,0,GFP_KERNEL);
    else{//if(!iod)
        printk("Key_map_p2p_pages | ENOMEM! iod\n");
        return -ENOMEM;
    }

    sg=iod->sg;
    sg_init_table(sg,count);

//  printk("map_p2p count:%d, page_size:%dKB, offset:%lx\n",count,page_size,offset);
    offset-=start*page_size;
    for(i=0;i<count;i++){
        if(!length) break;
        sg[i].page_link=0;
        sg[i].dma_address= page_table->pages[start+i]->physical_address+offset;
        sg[i].length=min_t(unsigned, length, page_size-offset);
        sg[i].dma_length=sg[i].length;
        sg[i].offset=0;
        offset=0;
        length-=sg[i].length;
//      printk("dma_addr[%d]:%lx,length:%lu\n",i,sg[i].dma_address,sg[i].length);
    }
    sg_mark_end(&sg[i - 1]);
    iod->nents=count;

    *((struct nvme_iod**)_iod)=iod;

    return 0;
}
void get_p2p_free_callback(void *data){
    key_page_table *p=data;
    nvidia_p2p_page_table_t *page_table=NULL;
    printk("free callbakc!\n");

    ACCESS_ONCE(p->call_flag)=1;
    wmb();
    page_table=xchg(&p->page_table,NULL);
    if(page_table){
        nvidia_p2p_free_page_table(page_table);
        barrier();
    }
    else{
        printk("callback error! nULL pagetable\n");
    }
}
int key_pin_buf(void __user *_param){
    struct I_KEY_PIN_PARAM param={0};
    int ret=0;
    struct nvidia_p2p_page_table *page_table=NULL;
    u64 page_virt_start;
    u64 page_virt_end;
    size_t rounded_size;
    key_page_table* p=NULL;
    cycles_t ts,te;

    if(copy_from_user(&param,_param,sizeof(param))){
        printk("copy_from_user fail!\n");
        ret=-EFAULT;
        goto pin_out;
    }
    
    p=kmalloc(sizeof(key_page_table),GFP_KERNEL);
    if(!p){
        printk("kmalloc fail!\n");
        ret=-ENOMEM;
        goto pin_out;
    }
    memset(p,0,sizeof(*p));

    ///paramter fill
    p->start_addr=param.addr & GPU_BOUND_MASK; //virtual address start
    p->pin_size=param.addr+param.size-p->start_addr;

    p->offset=param.addr&GPU_BOUND_OFFSET;
    p->length=param.size;
    p->p2p_token=0;
    p->va_space=0;
    p->id=random_get_entropy()&ID_MASK;
    p->call_flag=0;

    printk("nvidia_p2p_get_pages(va=0x%llx len=%lld)\n",p->start_addr,p->pin_size);
    
    ts=get_cycles();
    ret=nvidia_p2p_get_pages(p->p2p_token,p->va_space,p->start_addr,p->pin_size,&page_table,get_p2p_free_callback,p);
    // nvidia_p2p_get_pages(0, 0, p->start_addr, p->pin_size, &page_table, callback, p);
    // callback(p)
    te=get_cycles();
    if(ret<0){
        printk("failed nvidia_p2p_get_pages! ret=%d\n",ret);
        goto pin_out;
    }
    p->page_table=page_table;
    printk("cycles=%llu, tsc_khz=%u\n",te-ts,get_tsc_khz());

    switch(page_table->page_size){
        case NVIDIA_P2P_PAGE_SIZE_4KB:
            p->page_size=4*1024;
            break;
        case NVIDIA_P2P_PAGE_SIZE_64KB:
            p->page_size=64*1024;
            break;
        case NVIDIA_P2P_PAGE_SIZE_128KB:
            p->page_size=128*1024;
            break;
        default:
            printk("page_size error!\n");
            ret=-EINVAL;
            goto pin_out;
    }
    if(page_table->page_size !=NVIDIA_P2P_PAGE_SIZE_64KB){ //only 64kb enabled
        printk("nvidia_p2p_page size is not 64KB ! size_id=%d\n",page_table->page_size);
        ret=-EINVAL;
        goto pin_out;
    }
    {//test print
        int i;
        printk("page table entries: %d\n", page_table->entries);
        for (i=0; i<MIN(20,page_table->entries); ++i) {
            printk("page[%d]=0x%016llx\n", i, page_table->pages[i]->physical_address);
        }
    }
    param.id=p->id;
    param.start_addr=p->start_addr;
    param.pin_size=p->pin_size;
    key_table=p;
    
pin_out:
    if(ret && p && p->page_table){
        printk("error, call p2p_put_pages\n");
        nvidia_p2p_put_pages(p->p2p_token,p->va_space,p->start_addr,p->page_table);
        page_table=NULL; p->page_table=NULL;
    }
    if(ret && p){
        memset(p,0,sizeof(*p));
        kfree(p);
        p=NULL;
    }
    if(!ret && copy_to_user(_param,&param,sizeof(param))){
        printk("failed copy_to_user on pin_buffer\n");
        ret=-EFAULT;
    }
    return ret;
}
int key_unpin_buf(void __user *_param){
    int ret=0;
    key_page_table *p;
    ///first, simply implementation for working
    printk("unpin_buf!\n");
    p=key_table;
    if(!p->call_flag){
        ret=nvidia_p2p_put_pages(p->p2p_token,p->va_space,p->start_addr,p->page_table);
        if(ret)
            printk("nvidia_p2p_put_page error %d\n",ret);
    }
    else printk("unpinbuf called while callback has already been fired\n");
    kfree(p);
    key_table=NULL;
    return ret;
}
/////////////////////////////////////////////////////
Command_list cmd_list[RW_IPC_SIZE];
User_list buf_list[RW_IPC_SIZE];
Command_list *list[RW_IPC_SIZE];
static struct task_struct *complete_thread;
Complete_list complete_list[RW_IPC_SIZE]; 
Process_list p_list;
//static int async_mode=0;
#define STATUS_INIT -1
static volatile int async_done=0;
//static wait_queue_head_t complete_wait;
//static unsigned long wait_time;
static struct task_struct *complete_task=NULL;

////for batch method
long key_batch_command(unsigned long _buf, int len,int mode) {///////mode0 : batch_sync, 1: batch_async
    int ret = 0,i,num;
    unsigned int length;
    Batch_list batch_list;
    __u32 nblocks;
    if (copy_from_user(buf_list, _buf, sizeof(User_list)*len)) {
        printk("copy_from_user fail!\n");
        ret = -EFAULT;
        goto batch_op_out;
    }
    //printk("KEY_BATCH | num : %d, mode :%d\n", len,mode);
    if(mode==0){ //batch_sync
        batch_list.cnt=0;
        batch_list.end=len;
        spin_lock_init(&(batch_list.lock));
    }
    for (i = 0; i < len; i ++) {
        // num은 push_list의 idx
        num = buf_list[i].num;
        memset(&(cmd_list[num]), 0, sizeof(Command_list));
        nblocks = (long)(buf_list[i].size >> shift_value) - 1 < 0 ? 0 : (buf_list[i].size >> shift_value) - 1;  //maybe start 0 values!!!

        list[i] = &cmd_list[num];
        list[i]->c.common.opcode = (__u8)buf_list[i].mode;
        //list[i]->c.common.flags = (__u8)0x00;
        list[i]->c.common.nsid = cpu_to_le32(ns->ns_id);
        //c.common.cdw2[0] = cpu_to_le32(cmd->cdw2);
        //c.common.cdw2[1] = cpu_to_le32(cmd->cdw3);
        list[i]->c.common.cdw10[0] = cpu_to_le32( (__u32)(buf_list[i].key&MASK_DU) );
        list[i]->c.common.cdw10[1] = cpu_to_le32( (__u32)(buf_list[i].key>>32) );
        list[i]->c.common.cdw10[2] = cpu_to_le32( (__u32) nblocks );
        list[i]->c.common.cdw10[3] = cpu_to_le32( (__u32)(buf_list[i].size));
        //c.common.cdw10[3] = cpu_to_le32(cmd->cdw13);
        //c.common.cdw10[4] = cpu_to_le32(cmd->cdw14);
        //c.common.cdw10[5] = cpu_to_le32(cmd->cdw15);
//      list[i]->complete_num = num;
        list[i]->addr = buf_list[i].offset;
        list[i]->length = buf_list[i].size;
        
        if(buf_list[i].mode==GET){ //get
            if(Key_map_p2p_pages((unsigned long)(&(list[i]->iod)), (unsigned long)ns_dev, list[i]->addr, list[i]->length)<0){
                printk("key_batch | key_map_p2p_pages_fail!\n"); ret=-1; goto batch_op_out;
            }
        }
        else{ //put
            list[i]->iod=nvme_map_user_pages(ns_dev, buf_list[i].mode & 1, list[i]->addr,list[i]->length);
            if(IS_ERR(list[i]->iod)){
                printk("key_batch | nvme_map_user_pages fail!\n"); ret=-1; goto batch_op_out;
            }
        }
        length = nvme_setup_prps(ns_dev, list[i]->iod, list[i]->length, GFP_KERNEL);
        list[i]->c.common.dptr.prp1 = cpu_to_le64(sg_dma_address(list[i]->iod->sg));
        list[i]->c.common.dptr.prp2 = cpu_to_le64(list[i]->iod->first_dma);

        //list[i]->timout = ADMIN_TIMOUT;
        list[i]->cmdinfo.res = &(complete_list[num]);
        if(mode==0){ //batch_sync
            list[i]->cmdinfo.end=&(batch_list); ///only mode 0 is used
            list[i]->num=num;
        }
        else{  //batch_async
            ///test
//  int result_value=4096,err;
            list[i]->cmdinfo.task=complete_task;
            list[i]->cmdinfo.poll=&(p_list);
            list[i]->cmdinfo.num=num;
            complete_list[num].result_offset = buf_list[i].result_offset;
            complete_list[num].complete_offset = buf_list[i].complete_offset;
            complete_list[num].length = buf_list[i].size;
            complete_list[num].mode=buf_list[i].mode;
            /*
            if(err=copy_to_user((int __user *)complete_list[num].result_offset,&result_value,sizeof(int))){
                printk("copy_user_error|result_value %d!\n",err);
            }
            else printk("copy_success ,num %d!\n",num);
            */
        }
        if (length != list[i]->length) {
            printk("COMMAND CREATE FAIL!!!\n"); ret = -1; goto batch_op_out;
        }
    }
    ret=nvme_key_batch_command(nvme_bdev, (unsigned long)list, len,mode);
//  printk("END : KEY_BATCH | num : %d, mode :%d\n", len,mode);
    if(mode==0){ ///batch_sync
        for(i=0;i<len;i++){
            int t;
            if(list[i]->c.common.opcode==PUT){
                nvme_unmap_user_pages(ns_dev, list[i]->c.common.opcode & 1, list[i]->iod);
            }
            nvme_free_iod(ns_dev,list[i]->iod);
            t=complete_list[list[i]->num].status;
            if(t!=0){
                if(t==STATUS_INIT) printk("key_batch_sync | WHY STATUS_INIT_VALUE!!!\n");
                else{printk("key_batch_sync completion fail!\n"); ret=-1;}
            }
            complete_list[list[i]->num].status=STATUS_INIT;
        }
    }
batch_op_out:
    return ret;
}
int Complete_Process(void *data) {
    unsigned long flags;
    int i,result_value;
    int f=0,r=0;
    char success = 3; //put_success=4;
    int err;
    printk("complete_thread!\n");
    complete_task=current;
    while (!async_done) {
        set_current_state(TASK_INTERRUPTIBLE);

        spin_lock_irqsave(&p_list.lock,flags);
        // If NVMe transfer completed, batch_completion set r to (r+1)%num
        r=ACCESS_ONCE(p_list.r); p_list.f=f;
        if(f==r){
            spin_unlock_irqrestore(&p_list.lock,flags); 
            schedule();  //request 없을 때 sleep
            spin_lock_irqsave(&p_list.lock,flags);
            r=ACCESS_ONCE(p_list.r);
        }
        set_current_state(TASK_RUNNING);
        spin_unlock_irqrestore(&p_list.lock,flags);
//          wait_event_interruptible_timeout(complete_wait,  ,wait_time);
//      if(f==r) schedule();
        while(f!=r){
            i=p_list.node[f];
            // STATUS_INIT == -1. If we call key_batch_command, complete_list[i] is initialized as 0 
            // check STATUS_INIT for batch_sync(for check only one thread)
            if (complete_list[i].status!=STATUS_INIT) { //무조건 들어가야지.
//              printk("request_complete : %d\n",i);
                if(complete_list[i].mode==PUT){
                    nvme_unmap_user_pages(ns_dev,cmd_list[i].c.common.opcode & 1, cmd_list[i].iod);
                }
                nvme_free_iod(ns_dev,cmd_list[i].iod);
                if (complete_list[i].status == 0) result_value = complete_list[i].length;
                else result_value = -1;
                complete_list[i].status = STATUS_INIT;
        //      if(err=put_user(result_value,(int __user *)complete_list[i].result_offset)){
                if(err=copy_to_user((int __user *)complete_list[i].result_offset,&result_value,sizeof(int))){
                    printk("copy_user_error|result_value %d!\n",err);
                }
                //wmb();
                //              if(complete_list[i].mode==GET)
                //if(err=put_user(success, (char __user *)complete_list[i].complete_offset)){
                if(err=copy_to_user((char __user *)complete_list[i].complete_offset,&success,sizeof(char))){
                    printk("copy_user_error|success %d!\n",err);
                }
                //printk("[DEBUG:key_module/complete process]Request completed!\n");
                wmb();
                //              else
                //                  put_user(put_success, (char*)complete_list[i].complete_offset);
                f=BUF_NEXT(f,p_list.size);
            }
        }
    }
    return 0;
}
/////////////////////////////////////

//just one p2p operation
long key_p2p_op(int mode, unsigned long key, unsigned long offset,unsigned int size){

    int ret=0;
        
    __u32 nblocks;

    nblocks = (long)(size >> shift_value) - 1 < 0 ? 0 : (size >> shift_value) - 1;  //maybe start 0 values!!!

    //printk("p2p op=mode:%d, key=0x%llx, nblocks=%u, bar_addr:0x%llx, data_length=%u\n",mode,key,nblocks,offset,size);

    struct nvme_passthru_cmd cmd={
        .opcode=(__u8)mode,         //new opcode
        .flags=(__u8)0x00,              //PRP use + normal operation
        .rsvd1=(__u16)0x0000,           //not used in code
        .nsid=(__u32)(ns->ns_id),       //namespaceID is 0 -> maybe changed
        .cdw2=(__u32)0x00000000,        //not used
        .cdw3=(__u32)0x00000000,        //not used
        .metadata =(__u64)(uintptr_t)0,
        .addr=(__u64)(uintptr_t)offset,
        .metadata_len=(__u32)0,
        .data_len=(__u32)size, //혹시라도 align되야 되면, issue할 때 그렇게 안넣으면 되는거고.
        .cdw10=(__u32)(key & MASK_DU),
        .cdw11=(__u32)(key >> 32),
        .cdw12=(__u32)nblocks,
        .cdw13=(__u32)size,
        .cdw14=(__u32)0,
        .cdw15=(__u32)0,
        .timeout_ms=(__u32)0,
        .result=(__u32)0, //will be pointed complete command dword0(result)
    };

    ret=submit_key_cmd(&cmd,MODE_P2P);
//  printk("result : %d\n",ret);
    if(ret==0) return size;
    else return -1;

//p_op_out:
//  return ret;
}

////////////////////////////////////////////////

static inline unsigned int page_size_fit(size_t length){
    return (unsigned int)((length+PAGE_SIZE-1) & PAGE_MASK);
}
static int submit_key_cmd(struct nvme_passthru_cmd *cmd,int mode){
    int ret;
    ret=nvme_key_command(nvme_bdev,(unsigned long)cmd,mode);
    return ret;
}


//direct IO
long key_op(int mode, unsigned long key, char __user *buf, unsigned int size){

    int ret = 0;

    __u32 nblocks;

    nblocks = (long)(size >> shift_value) - 1 < 0 ? 0 : (size >> shift_value) - 1;  //maybe start 0 values!!!
    //nblocks = ((long)size - 1 >> shift_value) + 1;

    //printk("key op=mode:%d, key=0x%llx, nblocks=%u, data_length=%u\n", mode, key, nblocks, size);

    struct nvme_passthru_cmd cmd = {
        .opcode = (__u8)mode,           //new opcode
        .flags = (__u8)0x00,                //PRP use + normal operation
        .rsvd1 = (__u16)0x0000,         //not used in code
        .nsid = (__u32)(ns->ns_id),     //namespaceID is 0 -> maybe changed
        .cdw2 = (__u32)0x00000000,      //not used
        .cdw3 = (__u32)0x00000000,      //not used
        .metadata = (__u64)(uintptr_t)0,
        .addr = (__u64)(uintptr_t)buf,
        .metadata_len = (__u32)0,
        .data_len = (__u32)size, //혹시라도 align되야 되면, issue할 때 그렇게 안넣으면 되는거고.
        .cdw10 = (__u32)(key & MASK_DU),
        .cdw11 = (__u32)(key >> 32),
        .cdw12 = (__u32)nblocks,
        .cdw13 = (__u32)size,
        .cdw14 = (__u32)0,
        .cdw15 = (__u32)0,
        .timeout_ms = (__u32)0,
        .result = (__u32)0, //will be pointed complete command dword0(result)
    };

    ret = submit_key_cmd(&cmd,MODE_KEY);
//  printk("result : %d\n", ret);
    if(ret==0){
        return size;
    }
    else {
        printk("key_module | Fail! ret=%d\n", ret);
        return -1;
    }
}

void print_error(int x){
    return;
}
////////open,close
long Key_value_open(const char __user* filename) {
    long len;
    int err,i;
    //struct nvme_ns *ns; //just 4.1!!!!
    printk("key_value_open!\n");
    if (nvme_name != NULL) return 1;
    len = strnlen_user(filename, MAX_FILENAME); // Is len including NULL?
    nvme_name = kmalloc(len, GFP_ATOMIC);
    if (nvme_name == NULL) { print_error(0); return -1; }
    if ((err = copy_from_user(nvme_name, filename, len)<0)) { print_error(1); return err; }
    nvme_dev_f = filp_open(nvme_name, O_RDONLY, 0666);
    if (IS_ERR(nvme_dev_f)) { printk("nvme open error!\n"); print_error(2); return -1; }
    nvme_bdev = nvme_dev_f->f_inode->i_bdev;

    ns = nvme_bdev->bd_disk->private_data; //just 4.1!!!!!
    ns_dev=ns->dev;
    key_cmd_limit = nvme_key_max_segments(nvme_bdev) << LBA_SHIFT;
    printk("key_cmd_limit %u,  ns->lba_shift %d\n", key_cmd_limit, ns->lba_shift); //just 4.1!!!!
    shift_value = ns->lba_shift; ////cosmos is 4KB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!shibal!!!!!!!
            //nvme-cli에서 ioctl(fd,dBLKPBSZGET,&phys_sector_size)를 써서 괜히 phys_sector_size를 가져오는 것이 아니었구나!
    for (i = 0; i < RW_IPC_SIZE; i++) {
        complete_list[i].status = STATUS_INIT;
    }
    //for ASYNC

    return key_cmd_limit;
}
long Key_value_close(const char __user* filename) {
    kfree(nvme_name); nvme_name = NULL; //assume we hanve only one key_value_ssd and one namespace
    filp_close(nvme_dev_f, NULL); nvme_dev_f = NULL;

//  if(async_mode==1){
//      kthread_stop(complete_thread);
//  }
    return 1;
}
//driver init
static int __init key_value_init(void) {
    int result;
    
    Key_value_module=1;
    Key_op=&key_ops;
    //////////////////////////////////////////////////////////////
    result=register_chrdev(0,NAME,&inter_fops); //dynamically allocate a major number for the device
    if(result<0){
        printk(KERN_ALERT "failed to register a major number\n");
        return result;
    }
    inter_major=result;
    dclass=class_create(THIS_MODULE,NAME_CLASS);
    if(IS_ERR(dclass)){
        unregister_chrdev(inter_major,NAME);
        printk(KERN_ALERT "failed to register device class\n");
        return PTR_ERR(dclass);
    }
    device=device_create(dclass,NULL,MKDEV(inter_major,0),NULL,NAME);
    if(IS_ERR(device)){
        class_destroy(dclass);
        unregister_chrdev(inter_major,NAME);
        printk(KERN_ALERT "failed to create the device\n");
        return PTR_ERR(device);
    }
    printk(KERN_INFO "Init Module Success \n");
    printk(KERN_INFO "Device : /dev/key_value_driver, Major Num : %d \n",inter_major);
    //for ASYNC
    spin_lock_init(&p_list.lock);
    p_list.node=kmalloc(sizeof(int)*RW_IPC_SIZE+2,GFP_KERNEL);
    p_list.size=RW_IPC_SIZE+2;
//  wait_time=msecs_to_jiffies(1);
//  init_waitqueue_head(&complete_wait);

    printk("key_value_module init!\n");
    return 0;
}
//exit
static void __exit key_value_exit(void) {
    Key_value_module=0;
    Key_op=NULL;
    key_page_table *p=NULL;
    int ret;
    kfree(p_list.node);

    if(key_table!=NULL){
        p=key_table;
        if(!ACCESS_ONCE(p->call_flag)){
            ret=nvidia_p2p_put_pages(p->p2p_token,p->va_space,p->start_addr,p->page_table);
            if(ret)
                printk("nvidia_p2p_put_page error %d\n",ret);
        }
        kfree(p);
        key_table=NULL;
        printk("free pin_pages!\n");
    }
    device_destroy(dclass,MKDEV(inter_major,0));
    class_unregister(dclass);
    unregister_chrdev(inter_major,NAME);
    printk("key_value_module exit!\n");
}


//driver open
static long inter_ioctl(struct file *inode,unsigned int cmd, unsigned long arg){  //unlocked_ioctl은 3 parameter!!!!!!,옛날엔 4개
    int ret=0;
    //printk("ioctl!!!!!!!!!!! cmd:%x, %x,%x,%x,%x\n",cmd,I_KEY_PIN_BUF,I_KEY_UNPIN_BUF,I_KEY_P2P_GET,I_KEY_P2P_PUT);
    switch(cmd){
        case I_KEY_PIN_BUF:
            ret=key_pin_buf(arg);
            break;
        case I_KEY_UNPIN_BUF:
            ret=key_unpin_buf(arg);
            break;
        case I_KEY_ASYNC:
            printk("start ASYNC\n");
            //async_mode=1; 
            async_done=0;
            //p_list.size=RW_IPC_SIZE+2;
            p_list.f=p_list.r=0;
            Complete_Process((void*)NULL); //complete process
            ret=0;
            break;
        case I_KEY_ASYNC_CLOSE:
            async_done=1;
            wake_up_process(complete_task);

            break;
//      case I_KEY_P2P_GET:
//          ret=key_p2p_op(0,arg);
//          break;
//      case I_KEY_P2P_PUT:
//          ret=key_p2p_op(1,arg);
//          break;
        default:
            ret=-1;
    }
    return ret;
}
static int inter_open(struct inode *minode, struct file *mfile){

    printk("Open Module\n");

    return 0;
}
//close
static int inter_release(struct inode *minode, struct file *mfile){


    printk( "Release Module\n");
    return 0;
}

/*
pci.c
*/
/*
 * NVM Express device driver
 * Copyright (c) 2011-2014, Intel Corporation.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 */

#include <linux/aer.h>
#include <linux/bitops.h>
#include <linux/blkdev.h>
#include <linux/blk-mq.h>
#include <linux/blk-mq-pci.h>
#include <linux/cpu.h>
#include <linux/delay.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/genhd.h>
#include <linux/hdreg.h>
#include <linux/idr.h>
#include <linux/init.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/kdev_t.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/mutex.h>
#include <linux/pci.h>
#include <linux/poison.h>
#include <linux/ptrace.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/t10-pi.h>
#include <linux/timer.h>
#include <linux/types.h>
#include <linux/io-64-nonatomic-lo-hi.h>
#include <asm/unaligned.h>
#include <linux/nvme_ioctl.h>
#include "nvme.h"

#define NVME_Q_DEPTH		1024
#define NVME_AQ_DEPTH		256
#define SQ_SIZE(depth)		(depth * sizeof(struct nvme_command))
#define CQ_SIZE(depth)		(depth * sizeof(struct nvme_completion))
		
/*
 * We handle AEN commands ourselves and don't even let the
 * block layer know about them.
 */
#define NVME_AQ_BLKMQ_DEPTH	(NVME_AQ_DEPTH - NVME_NR_AERS)

static int use_threaded_interrupts;
module_param(use_threaded_interrupts, int, 0);

static bool use_cmb_sqes = true;
module_param(use_cmb_sqes, bool, 0644);
MODULE_PARM_DESC(use_cmb_sqes, "use controller's memory buffer for I/O SQes");

static struct workqueue_struct *nvme_workq;

struct nvme_dev;
struct nvme_queue;

static int nvme_reset(struct nvme_dev *dev);
static void nvme_process_cq(struct nvme_queue *nvmeq);
static void nvme_dev_disable(struct nvme_dev *dev, bool shutdown);

/*
 * Represents an NVM Express device.  Each nvme_dev is a PCI function.
 */
/*
struct nvme_dev {
	struct nvme_queue **queues;
	struct blk_mq_tag_set tagset;
	struct blk_mq_tag_set admin_tagset;
	u32 __iomem *dbs;
	struct device *dev;
	struct dma_pool *prp_page_pool;
	struct dma_pool *prp_small_pool;
	unsigned queue_count;
	unsigned online_queues;
	unsigned max_qid;
	int q_depth;
	u32 db_stride;
	void __iomem *bar;
	struct work_struct reset_work;
	struct work_struct remove_work;
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
};
*/

static inline struct nvme_dev *to_nvme_dev(struct nvme_ctrl *ctrl)
{
	return container_of(ctrl, struct nvme_dev, ctrl);
}

/*
 * An NVM Express queue.  Each device has at least two (one for admin
 * commands and one for I/O commands).
 */
struct nvme_queue {
	struct device *q_dmadev;
	struct nvme_dev *dev;
	char irqname[24];	/* nvme4294967295-65535\0 */
	spinlock_t q_lock;
	struct nvme_command *sq_cmds;
	struct nvme_command __iomem *sq_cmds_io;
	volatile struct nvme_completion *cqes;
	struct blk_mq_tags **tags;
	dma_addr_t sq_dma_addr;
	dma_addr_t cq_dma_addr;
	u32 __iomem *q_db;
	u16 q_depth;
	s16 cq_vector;
	u16 sq_tail;
	u16 cq_head;
	u16 qid;
	u8 cq_phase;
	u8 cqe_seen;
//	struct async_cmd_info cmdinfo;
	struct blk_mq_hw_ctx *hctx;
};

/*
 * The nvme_iod describes the data in an I/O, including the list of PRP
 * entries.  You can't see it in this data structure because C doesn't let
 * me express that.  Use nvme_init_iod to ensure there's enough space
 * allocated to store the PRP list.
 */
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

/*
 * Check we didin't inadvertently grow the command struct
 */
static inline void _nvme_check_size(void)
{
	BUILD_BUG_ON(sizeof(struct nvme_rw_command) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_create_cq) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_create_sq) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_delete_queue) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_features) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_format_cmd) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_abort_cmd) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_command) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_id_ctrl) != 4096);
	BUILD_BUG_ON(sizeof(struct nvme_id_ns) != 4096);
	BUILD_BUG_ON(sizeof(struct nvme_lba_range_type) != 64);
	BUILD_BUG_ON(sizeof(struct nvme_smart_log) != 512);
}

/*
 * Max size of iod being embedded in the request payload
 */
#define NVME_INT_PAGES		2
#define NVME_INT_BYTES(dev)	(NVME_INT_PAGES * (dev)->ctrl.page_size)

/*
 * Will slightly overestimate the number of pages needed.  This is OK
 * as it only leads to a small amount of wasted memory for the lifetime of
 * the I/O.
 */
static int nvme_npages(unsigned size, struct nvme_dev *dev)
{
	unsigned nprps = DIV_ROUND_UP(size + dev->ctrl.page_size,
				      dev->ctrl.page_size);
	return DIV_ROUND_UP(8 * nprps, PAGE_SIZE - 8);
}

static unsigned int nvme_iod_alloc_size(struct nvme_dev *dev,
		unsigned int size, unsigned int nseg)
{
	return sizeof(__le64 *) * nvme_npages(size, dev) +
			sizeof(struct scatterlist) * nseg;
}

static unsigned int nvme_cmd_size(struct nvme_dev *dev)
{
	return sizeof(struct nvme_iod) +
		nvme_iod_alloc_size(dev, NVME_INT_BYTES(dev), NVME_INT_PAGES);
}

static int nvmeq_irq(struct nvme_queue *nvmeq)
{
	return pci_irq_vector(to_pci_dev(nvmeq->dev->dev), nvmeq->cq_vector);
}

static int nvme_admin_init_hctx(struct blk_mq_hw_ctx *hctx, void *data,
				unsigned int hctx_idx)
{
	struct nvme_dev *dev = data;
	struct nvme_queue *nvmeq = dev->queues[0];

	WARN_ON(hctx_idx != 0);
	WARN_ON(dev->admin_tagset.tags[0] != hctx->tags);
	WARN_ON(nvmeq->tags);

	hctx->driver_data = nvmeq;
	nvmeq->tags = &dev->admin_tagset.tags[0];
	return 0;
}

static void nvme_admin_exit_hctx(struct blk_mq_hw_ctx *hctx, unsigned int hctx_idx)
{
	struct nvme_queue *nvmeq = hctx->driver_data;

	nvmeq->tags = NULL;
}

static int nvme_admin_init_request(void *data, struct request *req,
				unsigned int hctx_idx, unsigned int rq_idx,
				unsigned int numa_node)
{
	struct nvme_dev *dev = data;
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_queue *nvmeq = dev->queues[0];

	BUG_ON(!nvmeq);
	iod->nvmeq = nvmeq;
	return 0;
}

static int nvme_init_hctx(struct blk_mq_hw_ctx *hctx, void *data,
			  unsigned int hctx_idx)
{
	struct nvme_dev *dev = data;
	struct nvme_queue *nvmeq = dev->queues[hctx_idx + 1];

	if (!nvmeq->tags)
		nvmeq->tags = &dev->tagset.tags[hctx_idx];

	WARN_ON(dev->tagset.tags[hctx_idx] != hctx->tags);
	hctx->driver_data = nvmeq;
	return 0;
}

static int nvme_init_request(void *data, struct request *req,
				unsigned int hctx_idx, unsigned int rq_idx,
				unsigned int numa_node)
{
	struct nvme_dev *dev = data;
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_queue *nvmeq = dev->queues[hctx_idx + 1];

	BUG_ON(!nvmeq);
	iod->nvmeq = nvmeq;
	return 0;
}

static int nvme_pci_map_queues(struct blk_mq_tag_set *set)
{
	struct nvme_dev *dev = set->driver_data;

	return blk_mq_pci_map_queues(set, to_pci_dev(dev->dev));
}

/**
 * __nvme_submit_cmd() - Copy a command into a queue and ring the doorbell
 * @nvmeq: The queue to use
 * @cmd: The command to send
 *
 * Safe to use from interrupt context
 */
static void __nvme_submit_cmd(struct nvme_queue *nvmeq,
						struct nvme_command *cmd)
{
	u16 tail = nvmeq->sq_tail;

	if (nvmeq->sq_cmds_io)
		memcpy_toio(&nvmeq->sq_cmds_io[tail], cmd, sizeof(*cmd));
	else
		memcpy(&nvmeq->sq_cmds[tail], cmd, sizeof(*cmd));

	if (++tail == nvmeq->q_depth)
		tail = 0;
	writel(tail, nvmeq->q_db);
	nvmeq->sq_tail = tail;
}

static __le64 **iod_list(struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	return (__le64 **)(iod->sg + req->nr_phys_segments);
}

static int nvme_init_iod(struct request *rq, unsigned size,
		struct nvme_dev *dev)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(rq);
	int nseg = rq->nr_phys_segments;

	if (nseg > NVME_INT_PAGES || size > NVME_INT_BYTES(dev)) {
		iod->sg = kmalloc(nvme_iod_alloc_size(dev, size, nseg), GFP_ATOMIC);
		if (!iod->sg)
			return BLK_MQ_RQ_QUEUE_BUSY;
	} else {
		iod->sg = iod->inline_sg;
	}

	iod->aborted = 0;
	iod->npages = -1;
	iod->nents = 0;
	iod->length = size;

	if (!(rq->cmd_flags & REQ_DONTPREP)) {
		rq->retries = 0;
		rq->cmd_flags |= REQ_DONTPREP;
	}
	return 0;
}

void nvme_free_iod(struct nvme_dev *dev, struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	const int last_prp = dev->ctrl.page_size / 8 - 1;
	int i;
	__le64 **list = iod_list(req);
	dma_addr_t prp_dma = iod->first_dma;

	nvme_cleanup_cmd(req);

	if (iod->npages == 0)
		dma_pool_free(dev->prp_small_pool, list[0], prp_dma);
	for (i = 0; i < iod->npages; i++) {
		__le64 *prp_list = list[i];
		dma_addr_t next_prp_dma = le64_to_cpu(prp_list[last_prp]);
		dma_pool_free(dev->prp_page_pool, prp_list, prp_dma);
		prp_dma = next_prp_dma;
	}

	if (iod->sg != iod->inline_sg)
		kfree(iod->sg);
}
EXPORT_SYMBOL_GPL(nvme_free_iod);

#ifdef CONFIG_BLK_DEV_INTEGRITY
static void nvme_dif_prep(u32 p, u32 v, struct t10_pi_tuple *pi)
{
	if (be32_to_cpu(pi->ref_tag) == v)
		pi->ref_tag = cpu_to_be32(p);
}

static void nvme_dif_complete(u32 p, u32 v, struct t10_pi_tuple *pi)
{
	if (be32_to_cpu(pi->ref_tag) == p)
		pi->ref_tag = cpu_to_be32(v);
}

/**
 * nvme_dif_remap - remaps ref tags to bip seed and physical lba
 *
 * The virtual start sector is the one that was originally submitted by the
 * block layer.	Due to partitioning, MD/DM cloning, etc. the actual physical
 * start sector may be different. Remap protection information to match the
 * physical LBA on writes, and back to the original seed on reads.
 *
 * Type 0 and 3 do not have a ref tag, so no remapping required.
 */
static void nvme_dif_remap(struct request *req,
			void (*dif_swap)(u32 p, u32 v, struct t10_pi_tuple *pi))
{
	struct nvme_ns *ns = req->rq_disk->private_data;
	struct bio_integrity_payload *bip;
	struct t10_pi_tuple *pi;
	void *p, *pmap;
	u32 i, nlb, ts, phys, virt;

	if (!ns->pi_type || ns->pi_type == NVME_NS_DPS_PI_TYPE3)
		return;

	bip = bio_integrity(req->bio);
	if (!bip)
		return;

	pmap = kmap_atomic(bip->bip_vec->bv_page) + bip->bip_vec->bv_offset;

	p = pmap;
	virt = bip_get_seed(bip);
	phys = nvme_block_nr(ns, blk_rq_pos(req));
	nlb = (blk_rq_bytes(req) >> ns->lba_shift);
	ts = ns->disk->queue->integrity.tuple_size;

	for (i = 0; i < nlb; i++, virt++, phys++) {
		pi = (struct t10_pi_tuple *)p;
		dif_swap(phys, virt, pi);
		p += ts;
	}
	kunmap_atomic(pmap);
}
#else /* CONFIG_BLK_DEV_INTEGRITY */
static void nvme_dif_remap(struct request *req,
			void (*dif_swap)(u32 p, u32 v, struct t10_pi_tuple *pi))
{
}
static void nvme_dif_prep(u32 p, u32 v, struct t10_pi_tuple *pi)
{
}
static void nvme_dif_complete(u32 p, u32 v, struct t10_pi_tuple *pi)
{
}
#endif

bool nvme_setup_prps(struct nvme_dev *dev, struct request *req,
		int total_len)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct dma_pool *pool;
	int length = total_len;
	struct scatterlist *sg = iod->sg;
	int dma_len = sg_dma_len(sg);
	u64 dma_addr = sg_dma_address(sg);
	u32 page_size = dev->ctrl.page_size;
	int offset = dma_addr & (page_size - 1);
	__le64 *prp_list;
	__le64 **list = iod_list(req);
	dma_addr_t prp_dma;
	int nprps, i;

	length -= (page_size - offset);
	if (length <= 0)
		return true;

	dma_len -= (page_size - offset);
	if (dma_len) {
		dma_addr += (page_size - offset);
	} else {
		sg = sg_next(sg);
		dma_addr = sg_dma_address(sg);
		dma_len = sg_dma_len(sg);
	}

	if (length <= page_size) {
		iod->first_dma = dma_addr;
		return true;
	}

	nprps = DIV_ROUND_UP(length, page_size);
	if (nprps <= (256 / 8)) {
		pool = dev->prp_small_pool;
		iod->npages = 0;
	} else {
		pool = dev->prp_page_pool;
		iod->npages = 1;
	}

	prp_list = dma_pool_alloc(pool, GFP_ATOMIC, &prp_dma);
	if (!prp_list) {
		iod->first_dma = dma_addr;
		iod->npages = -1;
		return false;
	}
	list[0] = prp_list;
	iod->first_dma = prp_dma;
	i = 0;
	for (;;) {
		if (i == page_size >> 3) {
			__le64 *old_prp_list = prp_list;
			prp_list = dma_pool_alloc(pool, GFP_ATOMIC, &prp_dma);
			if (!prp_list)
				return false;
			list[iod->npages++] = prp_list;
			prp_list[0] = old_prp_list[i - 1];
			old_prp_list[i - 1] = cpu_to_le64(prp_dma);
			i = 1;
		}
		prp_list[i++] = cpu_to_le64(dma_addr);
		dma_len -= page_size;
		dma_addr += page_size;
		length -= page_size;
		if (length <= 0)
			break;
		if (dma_len > 0)
			continue;
		BUG_ON(dma_len < 0);
		sg = sg_next(sg);
		dma_addr = sg_dma_address(sg);
		dma_len = sg_dma_len(sg);
	}

	return true;
}
EXPORT_SYMBOL_GPL(nvme_setup_prps);

static int nvme_map_data(struct nvme_dev *dev, struct request *req,
		unsigned size, struct nvme_command *cmnd)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct request_queue *q = req->q;
	enum dma_data_direction dma_dir = rq_data_dir(req) ?
			DMA_TO_DEVICE : DMA_FROM_DEVICE;
	int ret = BLK_MQ_RQ_QUEUE_ERROR;

	sg_init_table(iod->sg, req->nr_phys_segments);
	iod->nents = blk_rq_map_sg(q, req, iod->sg);
	if (!iod->nents)
		goto out;

	ret = BLK_MQ_RQ_QUEUE_BUSY;
	if (!dma_map_sg_attrs(dev->dev, iod->sg, iod->nents, dma_dir,
				DMA_ATTR_NO_WARN))
		goto out;

	if (!nvme_setup_prps(dev, req, size))
		goto out_unmap;

	ret = BLK_MQ_RQ_QUEUE_ERROR;
	if (blk_integrity_rq(req)) {
		if (blk_rq_count_integrity_sg(q, req->bio) != 1)
			goto out_unmap;

		sg_init_table(&iod->meta_sg, 1);
		if (blk_rq_map_integrity_sg(q, req->bio, &iod->meta_sg) != 1)
			goto out_unmap;

		if (rq_data_dir(req))
			nvme_dif_remap(req, nvme_dif_prep);

		if (!dma_map_sg(dev->dev, &iod->meta_sg, 1, dma_dir))
			goto out_unmap;
	}

	cmnd->rw.dptr.prp1 = cpu_to_le64(sg_dma_address(iod->sg));
	cmnd->rw.dptr.prp2 = cpu_to_le64(iod->first_dma);
	if (blk_integrity_rq(req))
		cmnd->rw.metadata = cpu_to_le64(sg_dma_address(&iod->meta_sg));
	return BLK_MQ_RQ_QUEUE_OK;

out_unmap:
	dma_unmap_sg(dev->dev, iod->sg, iod->nents, dma_dir);
out:
	return ret;
}

static void nvme_unmap_data(struct nvme_dev *dev, struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	enum dma_data_direction dma_dir = rq_data_dir(req) ?
			DMA_TO_DEVICE : DMA_FROM_DEVICE;

	if (iod->nents) {
		dma_unmap_sg(dev->dev, iod->sg, iod->nents, dma_dir);
		if (blk_integrity_rq(req)) {
			if (!rq_data_dir(req))
				nvme_dif_remap(req, nvme_dif_complete);
			dma_unmap_sg(dev->dev, &iod->meta_sg, 1, dma_dir);
		}
	}

	nvme_free_iod(dev, req);
}

/*
 * NOTE: ns is NULL when called on the admin queue.
 */
static int nvme_queue_rq(struct blk_mq_hw_ctx *hctx,
			 const struct blk_mq_queue_data *bd)
{
	struct nvme_ns *ns = hctx->queue->queuedata;
	struct nvme_queue *nvmeq = hctx->driver_data;
	struct nvme_dev *dev = nvmeq->dev;
	struct request *req = bd->rq;
	struct nvme_command cmnd;
	unsigned map_len;
	int ret = BLK_MQ_RQ_QUEUE_OK;

	/*
	 * If formated with metadata, require the block layer provide a buffer
	 * unless this namespace is formated such that the metadata can be
	 * stripped/generated by the controller with PRACT=1.
	 */
	if (ns && ns->ms && !blk_integrity_rq(req)) {
		if (!(ns->pi_type && ns->ms == 8) &&
					req->cmd_type != REQ_TYPE_DRV_PRIV) {
			blk_mq_end_request(req, -EFAULT);
			return BLK_MQ_RQ_QUEUE_OK;
		}
	}

	map_len = nvme_map_len(req);
	ret = nvme_init_iod(req, map_len, dev);
	if (ret)
		return ret;

	ret = nvme_setup_cmd(ns, req, &cmnd);
	if (ret)
		goto out;

	if (req->nr_phys_segments)
		ret = nvme_map_data(dev, req, map_len, &cmnd);

	if (ret)
		goto out;

	cmnd.common.command_id = req->tag;
	blk_mq_start_request(req);

	spin_lock_irq(&nvmeq->q_lock);
	if (unlikely(nvmeq->cq_vector < 0)) {
		if (ns && !test_bit(NVME_NS_DEAD, &ns->flags))
			ret = BLK_MQ_RQ_QUEUE_BUSY;
		else
			ret = BLK_MQ_RQ_QUEUE_ERROR;
		spin_unlock_irq(&nvmeq->q_lock);
		goto out;
	}
	__nvme_submit_cmd(nvmeq, &cmnd);
	nvme_process_cq(nvmeq);
	spin_unlock_irq(&nvmeq->q_lock);
	return BLK_MQ_RQ_QUEUE_OK;
out:
	nvme_free_iod(dev, req);
	return ret;
}

static void nvme_complete_rq(struct request *req)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_dev *dev = iod->nvmeq->dev;
	int error = 0;

	nvme_unmap_data(dev, req);

	if (unlikely(req->errors)) {
		if (nvme_req_needs_retry(req, req->errors)) {
			req->retries++;
			nvme_requeue_req(req);
			return;
		}

		if (req->cmd_type == REQ_TYPE_DRV_PRIV)
			error = req->errors;
		else
			error = nvme_error_status(req->errors);
	}

	if (unlikely(iod->aborted)) {
		dev_warn(dev->ctrl.device,
			"completing aborted command with status: %04x\n",
			req->errors);
	}

	blk_mq_end_request(req, error);
}

/* We read the CQE phase first to check if the rest of the entry is valid */
static inline bool nvme_cqe_valid(struct nvme_queue *nvmeq, u16 head,
		u16 phase)
{
	return (le16_to_cpu(nvmeq->cqes[head].status) & 1) == phase;
}

static void __nvme_process_cq(struct nvme_queue *nvmeq, unsigned int *tag)
{
	u16 head, phase;

	head = nvmeq->cq_head;
	phase = nvmeq->cq_phase;

	while (nvme_cqe_valid(nvmeq, head, phase)) {
		struct nvme_completion cqe = nvmeq->cqes[head];
		struct request *req;

		if (++head == nvmeq->q_depth) {
			head = 0;
			phase = !phase;
		}

		if (tag && *tag == cqe.command_id)
			*tag = -1;

		if (unlikely(cqe.command_id >= nvmeq->q_depth)) {
			dev_warn(nvmeq->dev->ctrl.device,
				"invalid id %d completed on queue %d\n",
				cqe.command_id, le16_to_cpu(cqe.sq_id));
			continue;
		}

		/*
		 * AEN requests are special as they don't time out and can
		 * survive any kind of queue freeze and often don't respond to
		 * aborts.  We don't even bother to allocate a struct request
		 * for them but rather special case them here.
		 */
		if (unlikely(nvmeq->qid == 0 &&
				cqe.command_id >= NVME_AQ_BLKMQ_DEPTH)) {
			nvme_complete_async_event(&nvmeq->dev->ctrl, &cqe);
			continue;
		}

		req = blk_mq_tag_to_rq(*nvmeq->tags, cqe.command_id);
		if (req->cmd_type == REQ_TYPE_DRV_PRIV && req->special)
			memcpy(req->special, &cqe, sizeof(cqe));
		blk_mq_complete_request(req, le16_to_cpu(cqe.status) >> 1);

	}

	/* If the controller ignores the cq head doorbell and continuously
	 * writes to the queue, it is theoretically possible to wrap around
	 * the queue twice and mistakenly return IRQ_NONE.  Linux only
	 * requires that 0.1% of your interrupts are handled, so this isn't
	 * a big problem.
	 */
	if (head == nvmeq->cq_head && phase == nvmeq->cq_phase)
		return;

	if (likely(nvmeq->cq_vector >= 0))
		writel(head, nvmeq->q_db + nvmeq->dev->db_stride);
	nvmeq->cq_head = head;
	nvmeq->cq_phase = phase;

	nvmeq->cqe_seen = 1;
}

static void nvme_process_cq(struct nvme_queue *nvmeq)
{
	__nvme_process_cq(nvmeq, NULL);
}

static irqreturn_t nvme_irq(int irq, void *data)
{
	irqreturn_t result;
	struct nvme_queue *nvmeq = data;
	spin_lock(&nvmeq->q_lock);
	nvme_process_cq(nvmeq);
	result = nvmeq->cqe_seen ? IRQ_HANDLED : IRQ_NONE;
	nvmeq->cqe_seen = 0;
	spin_unlock(&nvmeq->q_lock);
	return result;
}

static irqreturn_t nvme_irq_check(int irq, void *data)
{
	struct nvme_queue *nvmeq = data;
	if (nvme_cqe_valid(nvmeq, nvmeq->cq_head, nvmeq->cq_phase))
		return IRQ_WAKE_THREAD;
	return IRQ_NONE;
}

static int nvme_poll(struct blk_mq_hw_ctx *hctx, unsigned int tag)
{
	struct nvme_queue *nvmeq = hctx->driver_data;

	if (nvme_cqe_valid(nvmeq, nvmeq->cq_head, nvmeq->cq_phase)) {
		spin_lock_irq(&nvmeq->q_lock);
		__nvme_process_cq(nvmeq, &tag);
		spin_unlock_irq(&nvmeq->q_lock);

		if (tag == -1)
			return 1;
	}

	return 0;
}

static void nvme_pci_submit_async_event(struct nvme_ctrl *ctrl, int aer_idx)
{
	struct nvme_dev *dev = to_nvme_dev(ctrl);
	struct nvme_queue *nvmeq = dev->queues[0];
	struct nvme_command c;

	memset(&c, 0, sizeof(c));
	c.common.opcode = nvme_admin_async_event;
	c.common.command_id = NVME_AQ_BLKMQ_DEPTH + aer_idx;

	spin_lock_irq(&nvmeq->q_lock);
	__nvme_submit_cmd(nvmeq, &c);
	spin_unlock_irq(&nvmeq->q_lock);
}

static int adapter_delete_queue(struct nvme_dev *dev, u8 opcode, u16 id)
{
	struct nvme_command c;

	memset(&c, 0, sizeof(c));
	c.delete_queue.opcode = opcode;
	c.delete_queue.qid = cpu_to_le16(id);

	return nvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

static int adapter_alloc_cq(struct nvme_dev *dev, u16 qid,
						struct nvme_queue *nvmeq)
{
	struct nvme_command c;
	int flags = NVME_QUEUE_PHYS_CONTIG | NVME_CQ_IRQ_ENABLED;

	/*
	 * Note: we (ab)use the fact the the prp fields survive if no data
	 * is attached to the request.
	 */
	memset(&c, 0, sizeof(c));
	c.create_cq.opcode = nvme_admin_create_cq;
	c.create_cq.prp1 = cpu_to_le64(nvmeq->cq_dma_addr);
	c.create_cq.cqid = cpu_to_le16(qid);
	c.create_cq.qsize = cpu_to_le16(nvmeq->q_depth - 1);
	c.create_cq.cq_flags = cpu_to_le16(flags);
	c.create_cq.irq_vector = cpu_to_le16(nvmeq->cq_vector);

	return nvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

static int adapter_alloc_sq(struct nvme_dev *dev, u16 qid,
						struct nvme_queue *nvmeq)
{
	struct nvme_command c;
	int flags = NVME_QUEUE_PHYS_CONTIG | NVME_SQ_PRIO_MEDIUM;

	/*
	 * Note: we (ab)use the fact the the prp fields survive if no data
	 * is attached to the request.
	 */
	memset(&c, 0, sizeof(c));
	c.create_sq.opcode = nvme_admin_create_sq;
	c.create_sq.prp1 = cpu_to_le64(nvmeq->sq_dma_addr);
	c.create_sq.sqid = cpu_to_le16(qid);
	c.create_sq.qsize = cpu_to_le16(nvmeq->q_depth - 1);
	c.create_sq.sq_flags = cpu_to_le16(flags);
	c.create_sq.cqid = cpu_to_le16(qid);

	return nvme_submit_sync_cmd(dev->ctrl.admin_q, &c, NULL, 0);
}

static int adapter_delete_cq(struct nvme_dev *dev, u16 cqid)
{
	return adapter_delete_queue(dev, nvme_admin_delete_cq, cqid);
}

static int adapter_delete_sq(struct nvme_dev *dev, u16 sqid)
{
	return adapter_delete_queue(dev, nvme_admin_delete_sq, sqid);
}

static void abort_endio(struct request *req, int error)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_queue *nvmeq = iod->nvmeq;
	u16 status = req->errors;

	dev_warn(nvmeq->dev->ctrl.device, "Abort status: 0x%x", status);
	atomic_inc(&nvmeq->dev->ctrl.abort_limit);
	blk_mq_free_request(req);
}

static enum blk_eh_timer_return nvme_timeout(struct request *req, bool reserved)
{
	struct nvme_iod *iod = blk_mq_rq_to_pdu(req);
	struct nvme_queue *nvmeq = iod->nvmeq;
	struct nvme_dev *dev = nvmeq->dev;
	struct request *abort_req;
	struct nvme_command cmd;

	/*
	 * Shutdown immediately if controller times out while starting. The
	 * reset work will see the pci device disabled when it gets the forced
	 * cancellation error. All outstanding requests are completed on
	 * shutdown, so we return BLK_EH_HANDLED.
	 */
	if (dev->ctrl.state == NVME_CTRL_RESETTING) {
		dev_warn(dev->ctrl.device,
			 "I/O %d QID %d timeout, disable controller\n",
			 req->tag, nvmeq->qid);
		nvme_dev_disable(dev, false);
		req->errors = NVME_SC_CANCELLED;
		return BLK_EH_HANDLED;
	}

	/*
 	 * Shutdown the controller immediately and schedule a reset if the
 	 * command was already aborted once before and still hasn't been
 	 * returned to the driver, or if this is the admin queue.
	 */
	if (!nvmeq->qid || iod->aborted) {
		dev_warn(dev->ctrl.device,
			 "I/O %d QID %d timeout, reset controller\n",
			 req->tag, nvmeq->qid);
		nvme_dev_disable(dev, false);
		nvme_reset(dev);

		/*
		 * Mark the request as handled, since the inline shutdown
		 * forces all outstanding requests to complete.
		 */
		req->errors = NVME_SC_CANCELLED;
		return BLK_EH_HANDLED;
	}

	iod->aborted = 1;

	if (atomic_dec_return(&dev->ctrl.abort_limit) < 0) {
		atomic_inc(&dev->ctrl.abort_limit);
		return BLK_EH_RESET_TIMER;
	}

	memset(&cmd, 0, sizeof(cmd));
	cmd.abort.opcode = nvme_admin_abort_cmd;
	cmd.abort.cid = req->tag;
	cmd.abort.sqid = cpu_to_le16(nvmeq->qid);

	dev_warn(nvmeq->dev->ctrl.device,
		"I/O %d QID %d timeout, aborting\n",
		 req->tag, nvmeq->qid);

	abort_req = nvme_alloc_request(dev->ctrl.admin_q, &cmd,
			BLK_MQ_REQ_NOWAIT, NVME_QID_ANY);
	if (IS_ERR(abort_req)) {
		atomic_inc(&dev->ctrl.abort_limit);
		return BLK_EH_RESET_TIMER;
	}

	abort_req->timeout = ADMIN_TIMEOUT;
	abort_req->end_io_data = NULL;
	blk_execute_rq_nowait(abort_req->q, NULL, abort_req, 0, abort_endio);

	/*
	 * The aborted req will be completed on receiving the abort req.
	 * We enable the timer again. If hit twice, it'll cause a device reset,
	 * as the device then is in a faulty state.
	 */
	return BLK_EH_RESET_TIMER;
}

static void nvme_free_queue(struct nvme_queue *nvmeq)
{
	dma_free_coherent(nvmeq->q_dmadev, CQ_SIZE(nvmeq->q_depth),
				(void *)nvmeq->cqes, nvmeq->cq_dma_addr);
	if (nvmeq->sq_cmds)
		dma_free_coherent(nvmeq->q_dmadev, SQ_SIZE(nvmeq->q_depth),
					nvmeq->sq_cmds, nvmeq->sq_dma_addr);
	kfree(nvmeq);
}

static void nvme_free_queues(struct nvme_dev *dev, int lowest)
{
	int i;

	for (i = dev->queue_count - 1; i >= lowest; i--) {
		struct nvme_queue *nvmeq = dev->queues[i];
		dev->queue_count--;
		dev->queues[i] = NULL;
		nvme_free_queue(nvmeq);
	}
}

/**
 * nvme_suspend_queue - put queue into suspended state
 * @nvmeq - queue to suspend
 */
static int nvme_suspend_queue(struct nvme_queue *nvmeq)
{
	int vector;

	spin_lock_irq(&nvmeq->q_lock);
	if (nvmeq->cq_vector == -1) {
		spin_unlock_irq(&nvmeq->q_lock);
		return 1;
	}
	vector = nvmeq_irq(nvmeq);
	nvmeq->dev->online_queues--;
	nvmeq->cq_vector = -1;
	spin_unlock_irq(&nvmeq->q_lock);

	if (!nvmeq->qid && nvmeq->dev->ctrl.admin_q)
		blk_mq_stop_hw_queues(nvmeq->dev->ctrl.admin_q);

	free_irq(vector, nvmeq);

	return 0;
}

static void nvme_disable_admin_queue(struct nvme_dev *dev, bool shutdown)
{
	struct nvme_queue *nvmeq = dev->queues[0];

	if (!nvmeq)
		return;
	if (nvme_suspend_queue(nvmeq))
		return;

	if (shutdown)
		nvme_shutdown_ctrl(&dev->ctrl);
	else
		nvme_disable_ctrl(&dev->ctrl, lo_hi_readq(
						dev->bar + NVME_REG_CAP));

	spin_lock_irq(&nvmeq->q_lock);
	nvme_process_cq(nvmeq);
	spin_unlock_irq(&nvmeq->q_lock);
}

static int nvme_cmb_qdepth(struct nvme_dev *dev, int nr_io_queues,
				int entry_size)
{
	int q_depth = dev->q_depth;
	unsigned q_size_aligned = roundup(q_depth * entry_size,
					  dev->ctrl.page_size);

	if (q_size_aligned * nr_io_queues > dev->cmb_size) {
		u64 mem_per_q = div_u64(dev->cmb_size, nr_io_queues);
		mem_per_q = round_down(mem_per_q, dev->ctrl.page_size);
		q_depth = div_u64(mem_per_q, entry_size);

		/*
		 * Ensure the reduced q_depth is above some threshold where it
		 * would be better to map queues in system memory with the
		 * original depth
		 */
		if (q_depth < 64)
			return -ENOMEM;
	}

	return q_depth;
}

static int nvme_alloc_sq_cmds(struct nvme_dev *dev, struct nvme_queue *nvmeq,
				int qid, int depth)
{

	/* CMB SQEs will be mapped before creation */
	if (qid && dev->cmb && use_cmb_sqes && NVME_CMB_SQS(dev->cmbsz))
		return 0;

	nvmeq->sq_cmds = dma_alloc_coherent(dev->dev, SQ_SIZE(depth),
					    &nvmeq->sq_dma_addr, GFP_KERNEL);
	if (!nvmeq->sq_cmds)
		return -ENOMEM;

	return 0;
}

static struct nvme_queue *nvme_alloc_queue(struct nvme_dev *dev, int qid,
							int depth)
{
	struct nvme_queue *nvmeq = kzalloc(sizeof(*nvmeq), GFP_KERNEL);
	if (!nvmeq)
		return NULL;

	nvmeq->cqes = dma_zalloc_coherent(dev->dev, CQ_SIZE(depth),
					  &nvmeq->cq_dma_addr, GFP_KERNEL);
	if (!nvmeq->cqes)
		goto free_nvmeq;

	if (nvme_alloc_sq_cmds(dev, nvmeq, qid, depth))
		goto free_cqdma;

	nvmeq->q_dmadev = dev->dev;
	nvmeq->dev = dev;
	snprintf(nvmeq->irqname, sizeof(nvmeq->irqname), "nvme%dq%d",
			dev->ctrl.instance, qid);
	spin_lock_init(&nvmeq->q_lock);
	nvmeq->cq_head = 0;
	nvmeq->cq_phase = 1;
	nvmeq->q_db = &dev->dbs[qid * 2 * dev->db_stride];
	nvmeq->q_depth = depth;
	nvmeq->qid = qid;
	nvmeq->cq_vector = -1;
	dev->queues[qid] = nvmeq;
	dev->queue_count++;

	return nvmeq;

 free_cqdma:
	dma_free_coherent(dev->dev, CQ_SIZE(depth), (void *)nvmeq->cqes,
							nvmeq->cq_dma_addr);
 free_nvmeq:
	kfree(nvmeq);
	return NULL;
}

static int queue_request_irq(struct nvme_queue *nvmeq)
{
	if (use_threaded_interrupts)
		return request_threaded_irq(nvmeq_irq(nvmeq), nvme_irq_check,
				nvme_irq, IRQF_SHARED, nvmeq->irqname, nvmeq);
	else
		return request_irq(nvmeq_irq(nvmeq), nvme_irq, IRQF_SHARED,
				nvmeq->irqname, nvmeq);
}

static void nvme_init_queue(struct nvme_queue *nvmeq, u16 qid)
{
	struct nvme_dev *dev = nvmeq->dev;

	spin_lock_irq(&nvmeq->q_lock);
	nvmeq->sq_tail = 0;
	nvmeq->cq_head = 0;
	nvmeq->cq_phase = 1;
	nvmeq->q_db = &dev->dbs[qid * 2 * dev->db_stride];
	memset((void *)nvmeq->cqes, 0, CQ_SIZE(nvmeq->q_depth));
	dev->online_queues++;
	spin_unlock_irq(&nvmeq->q_lock);
}

static int nvme_create_queue(struct nvme_queue *nvmeq, int qid)
{
	struct nvme_dev *dev = nvmeq->dev;
	int result;

	if (qid && dev->cmb && use_cmb_sqes && NVME_CMB_SQS(dev->cmbsz)) {
		unsigned offset = (qid - 1) * roundup(SQ_SIZE(nvmeq->q_depth),
						      dev->ctrl.page_size);
		nvmeq->sq_dma_addr = dev->cmb_bus_addr + offset;
		nvmeq->sq_cmds_io = dev->cmb + offset;
	}

	nvmeq->cq_vector = qid - 1;
	result = adapter_alloc_cq(dev, qid, nvmeq);
	if (result < 0)
		goto release_vector;

	result = adapter_alloc_sq(dev, qid, nvmeq);
	if (result < 0)
		goto release_cq;

	nvme_init_queue(nvmeq, qid);
	result = queue_request_irq(nvmeq);
	if (result < 0)
		goto release_sq;

	return result;

 release_sq:
	dev->online_queues--;
	adapter_delete_sq(dev, qid);
 release_cq:
	adapter_delete_cq(dev, qid);
 release_vector:
	nvmeq->cq_vector = -1;
	return result;
}

static struct blk_mq_ops nvme_mq_admin_ops = {
	.queue_rq	= nvme_queue_rq,
	.complete	= nvme_complete_rq,
	.init_hctx	= nvme_admin_init_hctx,
	.exit_hctx      = nvme_admin_exit_hctx,
	.init_request	= nvme_admin_init_request,
	.timeout	= nvme_timeout,
};

static struct blk_mq_ops nvme_mq_ops = {
	.queue_rq	= nvme_queue_rq,
	.complete	= nvme_complete_rq,
	.init_hctx	= nvme_init_hctx,
	.init_request	= nvme_init_request,
	.map_queues	= nvme_pci_map_queues,
	.timeout	= nvme_timeout,
	.poll		= nvme_poll,
};

static void nvme_dev_remove_admin(struct nvme_dev *dev)
{
	if (dev->ctrl.admin_q && !blk_queue_dying(dev->ctrl.admin_q)) {
		/*
		 * If the controller was reset during removal, it's possible
		 * user requests may be waiting on a stopped queue. Start the
		 * queue to flush these to completion.
		 */
		blk_mq_start_stopped_hw_queues(dev->ctrl.admin_q, true);
		blk_cleanup_queue(dev->ctrl.admin_q);
		blk_mq_free_tag_set(&dev->admin_tagset);
	}
}

static int nvme_alloc_admin_tags(struct nvme_dev *dev)
{
	if (!dev->ctrl.admin_q) {
		dev->admin_tagset.ops = &nvme_mq_admin_ops;
		dev->admin_tagset.nr_hw_queues = 1;

		/*
		 * Subtract one to leave an empty queue entry for 'Full Queue'
		 * condition. See NVM-Express 1.2 specification, section 4.1.2.
		 */
		dev->admin_tagset.queue_depth = NVME_AQ_BLKMQ_DEPTH - 1;
		dev->admin_tagset.timeout = ADMIN_TIMEOUT;
		dev->admin_tagset.numa_node = dev_to_node(dev->dev);
		dev->admin_tagset.cmd_size = nvme_cmd_size(dev);
		dev->admin_tagset.driver_data = dev;

		if (blk_mq_alloc_tag_set(&dev->admin_tagset))
			return -ENOMEM;

		dev->ctrl.admin_q = blk_mq_init_queue(&dev->admin_tagset);
		if (IS_ERR(dev->ctrl.admin_q)) {
			blk_mq_free_tag_set(&dev->admin_tagset);
			return -ENOMEM;
		}
		if (!blk_get_queue(dev->ctrl.admin_q)) {
			nvme_dev_remove_admin(dev);
			dev->ctrl.admin_q = NULL;
			return -ENODEV;
		}
	} else
		blk_mq_start_stopped_hw_queues(dev->ctrl.admin_q, true);

	return 0;
}

static int nvme_configure_admin_queue(struct nvme_dev *dev)
{
	int result;
	u32 aqa;
	u64 cap = lo_hi_readq(dev->bar + NVME_REG_CAP);
	struct nvme_queue *nvmeq;

	dev->subsystem = readl(dev->bar + NVME_REG_VS) >= NVME_VS(1, 1, 0) ?
						NVME_CAP_NSSRC(cap) : 0;

	if (dev->subsystem &&
	    (readl(dev->bar + NVME_REG_CSTS) & NVME_CSTS_NSSRO))
		writel(NVME_CSTS_NSSRO, dev->bar + NVME_REG_CSTS);

	result = nvme_disable_ctrl(&dev->ctrl, cap);
	if (result < 0)
		return result;

	nvmeq = dev->queues[0];
	if (!nvmeq) {
		nvmeq = nvme_alloc_queue(dev, 0, NVME_AQ_DEPTH);
		if (!nvmeq)
			return -ENOMEM;
	}

	aqa = nvmeq->q_depth - 1;
	aqa |= aqa << 16;

	writel(aqa, dev->bar + NVME_REG_AQA);
	lo_hi_writeq(nvmeq->sq_dma_addr, dev->bar + NVME_REG_ASQ);
	lo_hi_writeq(nvmeq->cq_dma_addr, dev->bar + NVME_REG_ACQ);

	result = nvme_enable_ctrl(&dev->ctrl, cap);
	if (result)
		return result;

	nvmeq->cq_vector = 0;
	nvme_init_queue(nvmeq, 0);
	result = queue_request_irq(nvmeq);
	if (result) {
		nvmeq->cq_vector = -1;
		return result;
	}

	return result;
}

static bool nvme_should_reset(struct nvme_dev *dev, u32 csts)
{

	/* If true, indicates loss of adapter communication, possibly by a
	 * NVMe Subsystem reset.
	 */
	bool nssro = dev->subsystem && (csts & NVME_CSTS_NSSRO);

	/* If there is a reset ongoing, we shouldn't reset again. */
	if (dev->ctrl.state == NVME_CTRL_RESETTING)
		return false;

	/* We shouldn't reset unless the controller is on fatal error state
	 * _or_ if we lost the communication with it.
	 */
	if (!(csts & NVME_CSTS_CFS) && !nssro)
		return false;

	/* If PCI error recovery process is happening, we cannot reset or
	 * the recovery mechanism will surely fail.
	 */
	if (pci_channel_offline(to_pci_dev(dev->dev)))
		return false;

	return true;
}

static void nvme_watchdog_timer(unsigned long data)
{
	struct nvme_dev *dev = (struct nvme_dev *)data;
	u32 csts = readl(dev->bar + NVME_REG_CSTS);

	/* Skip controllers under certain specific conditions. */
	if (nvme_should_reset(dev, csts)) {
		if (!nvme_reset(dev))
			dev_warn(dev->dev,
				"Failed status: 0x%x, reset controller.\n",
				csts);
		return;
	}

	mod_timer(&dev->watchdog_timer, round_jiffies(jiffies + HZ));
}

static int nvme_create_io_queues(struct nvme_dev *dev)
{
	unsigned i, max;
	int ret = 0;

	for (i = dev->queue_count; i <= dev->max_qid; i++) {
		if (!nvme_alloc_queue(dev, i, dev->q_depth)) {
			ret = -ENOMEM;
			break;
		}
	}

	max = min(dev->max_qid, dev->queue_count - 1);
	for (i = dev->online_queues; i <= max; i++) {
		ret = nvme_create_queue(dev->queues[i], i);
		if (ret)
			break;
	}

	/*
	 * Ignore failing Create SQ/CQ commands, we can continue with less
	 * than the desired aount of queues, and even a controller without
	 * I/O queues an still be used to issue admin commands.  This might
	 * be useful to upgrade a buggy firmware for example.
	 */
	return ret >= 0 ? 0 : ret;
}

static ssize_t nvme_cmb_show(struct device *dev,
			     struct device_attribute *attr,
			     char *buf)
{
	struct nvme_dev *ndev = to_nvme_dev(dev_get_drvdata(dev));

	return snprintf(buf, PAGE_SIZE, "cmbloc : x%08x\ncmbsz  : x%08x\n",
		       ndev->cmbloc, ndev->cmbsz);
}
static DEVICE_ATTR(cmb, S_IRUGO, nvme_cmb_show, NULL);

static void __iomem *nvme_map_cmb(struct nvme_dev *dev)
{
	u64 szu, size, offset;
	resource_size_t bar_size;
	struct pci_dev *pdev = to_pci_dev(dev->dev);
	void __iomem *cmb;
	int bar;

	dev->cmbsz = readl(dev->bar + NVME_REG_CMBSZ);
	if (!(NVME_CMB_SZ(dev->cmbsz)))
		return NULL;
	dev->cmbloc = readl(dev->bar + NVME_REG_CMBLOC);

	if (!use_cmb_sqes)
		return NULL;

	szu = (u64)1 << (12 + 4 * NVME_CMB_SZU(dev->cmbsz));
	size = szu * NVME_CMB_SZ(dev->cmbsz);
	offset = szu * NVME_CMB_OFST(dev->cmbloc);
	bar = NVME_CMB_BIR(dev->cmbloc);
	bar_size = pci_resource_len(pdev, bar);

	if (offset > bar_size)
		return NULL;

	/*
	 * Controllers may support a CMB size larger than their BAR,
	 * for example, due to being behind a bridge. Reduce the CMB to
	 * the reported size of the BAR
	 */
	if (size > bar_size - offset)
		size = bar_size - offset;

	cmb = ioremap_wc(pci_resource_start(pdev, bar) + offset, size);
	if (!cmb)
		return NULL;

	dev->cmb_bus_addr = pci_bus_address(pdev, bar) + offset;
	dev->cmb_size = size;
	return cmb;
}

static inline void nvme_release_cmb(struct nvme_dev *dev)
{
	if (dev->cmb) {
		iounmap(dev->cmb);
		dev->cmb = NULL;
		sysfs_remove_file_from_group(&dev->ctrl.device->kobj,
					     &dev_attr_cmb.attr, NULL);
		dev->cmbsz = 0;
	}
}

static size_t db_bar_size(struct nvme_dev *dev, unsigned nr_io_queues)
{
	return 4096 + ((nr_io_queues + 1) * 8 * dev->db_stride);
}

static int nvme_setup_io_queues(struct nvme_dev *dev)
{
	struct nvme_queue *adminq = dev->queues[0];
	struct pci_dev *pdev = to_pci_dev(dev->dev);
	int result, nr_io_queues, size;

	nr_io_queues = num_online_cpus();
	result = nvme_set_queue_count(&dev->ctrl, &nr_io_queues);
	if (result < 0)
		return result;

	if (nr_io_queues == 0)
		return 0;

	if (dev->cmb && NVME_CMB_SQS(dev->cmbsz)) {
		result = nvme_cmb_qdepth(dev, nr_io_queues,
				sizeof(struct nvme_command));
		if (result > 0)
			dev->q_depth = result;
		else
			nvme_release_cmb(dev);
	}

	size = db_bar_size(dev, nr_io_queues);
	if (size > 8192) {
		iounmap(dev->bar);
		do {
			dev->bar = ioremap(pci_resource_start(pdev, 0), size);
			if (dev->bar)
				break;
			if (!--nr_io_queues)
				return -ENOMEM;
			size = db_bar_size(dev, nr_io_queues);
		} while (1);
		dev->dbs = dev->bar + 4096;
		adminq->q_db = dev->dbs;
	}

	/* Deregister the admin queue's interrupt */
	free_irq(pci_irq_vector(pdev, 0), adminq);

	/*
	 * If we enable msix early due to not intx, disable it again before
	 * setting up the full range we need.
	 */
	pci_free_irq_vectors(pdev);
	nr_io_queues = pci_alloc_irq_vectors(pdev, 1, nr_io_queues,
			PCI_IRQ_ALL_TYPES | PCI_IRQ_AFFINITY);
	if (nr_io_queues <= 0)
		return -EIO;
	dev->max_qid = nr_io_queues;

	/*
	 * Should investigate if there's a performance win from allocating
	 * more queues than interrupt vectors; it might allow the submission
	 * path to scale better, even if the receive path is limited by the
	 * number of interrupts.
	 */

	result = queue_request_irq(adminq);
	if (result) {
		adminq->cq_vector = -1;
		return result;
	}
	return nvme_create_io_queues(dev);
}

static void nvme_del_queue_end(struct request *req, int error)
{
	struct nvme_queue *nvmeq = req->end_io_data;

	blk_mq_free_request(req);
	complete(&nvmeq->dev->ioq_wait);
}

static void nvme_del_cq_end(struct request *req, int error)
{
	struct nvme_queue *nvmeq = req->end_io_data;

	if (!error) {
		unsigned long flags;

		/*
		 * We might be called with the AQ q_lock held
		 * and the I/O queue q_lock should always
		 * nest inside the AQ one.
		 */
		spin_lock_irqsave_nested(&nvmeq->q_lock, flags,
					SINGLE_DEPTH_NESTING);
		nvme_process_cq(nvmeq);
		spin_unlock_irqrestore(&nvmeq->q_lock, flags);
	}

	nvme_del_queue_end(req, error);
}

static int nvme_delete_queue(struct nvme_queue *nvmeq, u8 opcode)
{
	struct request_queue *q = nvmeq->dev->ctrl.admin_q;
	struct request *req;
	struct nvme_command cmd;

	memset(&cmd, 0, sizeof(cmd));
	cmd.delete_queue.opcode = opcode;
	cmd.delete_queue.qid = cpu_to_le16(nvmeq->qid);

	req = nvme_alloc_request(q, &cmd, BLK_MQ_REQ_NOWAIT, NVME_QID_ANY);
	if (IS_ERR(req))
		return PTR_ERR(req);

	req->timeout = ADMIN_TIMEOUT;
	req->end_io_data = nvmeq;

	blk_execute_rq_nowait(q, NULL, req, false,
			opcode == nvme_admin_delete_cq ?
				nvme_del_cq_end : nvme_del_queue_end);
	return 0;
}

static void nvme_disable_io_queues(struct nvme_dev *dev, int queues)
{
	int pass;
	unsigned long timeout;
	u8 opcode = nvme_admin_delete_sq;

	for (pass = 0; pass < 2; pass++) {
		int sent = 0, i = queues;

		reinit_completion(&dev->ioq_wait);
 retry:
		timeout = ADMIN_TIMEOUT;
		for (; i > 0; i--, sent++)
			if (nvme_delete_queue(dev->queues[i], opcode))
				break;

		while (sent--) {
			timeout = wait_for_completion_io_timeout(&dev->ioq_wait, timeout);
			if (timeout == 0)
				return;
			if (i)
				goto retry;
		}
		opcode = nvme_admin_delete_cq;
	}
}

/*
 * Return: error value if an error occurred setting up the queues or calling
 * Identify Device.  0 if these succeeded, even if adding some of the
 * namespaces failed.  At the moment, these failures are silent.  TBD which
 * failures should be reported.
 */
static int nvme_dev_add(struct nvme_dev *dev)
{
	if (!dev->ctrl.tagset) {
		dev->tagset.ops = &nvme_mq_ops;
		dev->tagset.nr_hw_queues = dev->online_queues - 1;
		dev->tagset.timeout = NVME_IO_TIMEOUT;
		dev->tagset.numa_node = dev_to_node(dev->dev);
		dev->tagset.queue_depth =
				min_t(int, dev->q_depth, BLK_MQ_MAX_DEPTH) - 1;
		dev->tagset.cmd_size = nvme_cmd_size(dev);
		dev->tagset.flags = BLK_MQ_F_SHOULD_MERGE;
		dev->tagset.driver_data = dev;

		if (blk_mq_alloc_tag_set(&dev->tagset))
			return 0;
		dev->ctrl.tagset = &dev->tagset;
	} else {
		blk_mq_update_nr_hw_queues(&dev->tagset, dev->online_queues - 1);

		/* Free previously allocated queues that are no longer usable */
		nvme_free_queues(dev, dev->online_queues);
	}

	return 0;
}

static int nvme_pci_enable(struct nvme_dev *dev)
{
	u64 cap;
	int result = -ENOMEM;
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	if (pci_enable_device_mem(pdev))
		return result;

	pci_set_master(pdev);

	if (dma_set_mask_and_coherent(dev->dev, DMA_BIT_MASK(64)) &&
	    dma_set_mask_and_coherent(dev->dev, DMA_BIT_MASK(32)))
		goto disable;

	if (readl(dev->bar + NVME_REG_CSTS) == -1) {
		result = -ENODEV;
		goto disable;
	}

	/*
	 * Some devices and/or platforms don't advertise or work with INTx
	 * interrupts. Pre-enable a single MSIX or MSI vec for setup. We'll
	 * adjust this later.
	 */
	result = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_ALL_TYPES);
	if (result < 0)
		return result;

	cap = lo_hi_readq(dev->bar + NVME_REG_CAP);

	dev->q_depth = min_t(int, NVME_CAP_MQES(cap) + 1, NVME_Q_DEPTH);
	dev->db_stride = 1 << NVME_CAP_STRIDE(cap);
	dev->dbs = dev->bar + 4096;

	/*
	 * Temporary fix for the Apple controller found in the MacBook8,1 and
	 * some MacBook7,1 to avoid controller resets and data loss.
	 */
	if (pdev->vendor == PCI_VENDOR_ID_APPLE && pdev->device == 0x2001) {
		dev->q_depth = 2;
		dev_warn(dev->dev, "detected Apple NVMe controller, set "
			"queue depth=%u to work around controller resets\n",
			dev->q_depth);
	}

	/*
	 * CMBs can currently only exist on >=1.2 PCIe devices. We only
	 * populate sysfs if a CMB is implemented. Since nvme_dev_attrs_group
	 * has no name we can pass NULL as final argument to
	 * sysfs_add_file_to_group.
	 */

	if (readl(dev->bar + NVME_REG_VS) >= NVME_VS(1, 2, 0)) {
		dev->cmb = nvme_map_cmb(dev);
		if (dev->cmb) {
			if (sysfs_add_file_to_group(&dev->ctrl.device->kobj,
						    &dev_attr_cmb.attr, NULL))
				dev_warn(dev->dev,
					 "failed to add sysfs attribute for CMB\n");
		}
	}

	pci_enable_pcie_error_reporting(pdev);
	pci_save_state(pdev);
	return 0;

 disable:
	pci_disable_device(pdev);
	return result;
}

static void nvme_dev_unmap(struct nvme_dev *dev)
{
	if (dev->bar)
		iounmap(dev->bar);
	pci_release_mem_regions(to_pci_dev(dev->dev));
}

static void nvme_pci_disable(struct nvme_dev *dev)
{
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	nvme_release_cmb(dev);
	pci_free_irq_vectors(pdev);

	if (pci_is_enabled(pdev)) {
		pci_disable_pcie_error_reporting(pdev);
		pci_disable_device(pdev);
	}
}

static void nvme_dev_disable(struct nvme_dev *dev, bool shutdown)
{
	int i, queues;
	bool dead = true;
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	del_timer_sync(&dev->watchdog_timer);

	mutex_lock(&dev->shutdown_lock);
	if (pci_is_enabled(pdev)) {
		u32 csts = readl(dev->bar + NVME_REG_CSTS);

		if (dev->ctrl.state == NVME_CTRL_LIVE)
			nvme_start_freeze(&dev->ctrl);
		dead = !!((csts & NVME_CSTS_CFS) || !(csts & NVME_CSTS_RDY) ||
			pdev->error_state  != pci_channel_io_normal);
	}

	/*
	 * Give the controller a chance to complete all entered requests if
	 * doing a safe shutdown.
	 */
	if (!dead && shutdown)
		nvme_wait_freeze_timeout(&dev->ctrl, NVME_IO_TIMEOUT);
	nvme_stop_queues(&dev->ctrl);

	queues = dev->online_queues - 1;
	for (i = dev->queue_count - 1; i > 0; i--)
		nvme_suspend_queue(dev->queues[i]);

	if (dead) {
		/* A device might become IO incapable very soon during
		 * probe, before the admin queue is configured. Thus,
		 * queue_count can be 0 here.
		 */
		if (dev->queue_count)
			nvme_suspend_queue(dev->queues[0]);
	} else {
		nvme_disable_io_queues(dev, queues);
		nvme_disable_admin_queue(dev, shutdown);
	}
	nvme_pci_disable(dev);

	blk_mq_tagset_busy_iter(&dev->tagset, nvme_cancel_request, &dev->ctrl);
	blk_mq_tagset_busy_iter(&dev->admin_tagset, nvme_cancel_request, &dev->ctrl);

	/*
	 * The driver will not be starting up queues again if shutting down so
	 * must flush all entered requests to their failed completion to avoid
	 * deadlocking blk-mq hot-cpu notifier.
	 */
	if (shutdown)
		nvme_start_queues(&dev->ctrl);
	mutex_unlock(&dev->shutdown_lock);
}

static int nvme_setup_prp_pools(struct nvme_dev *dev)
{
	dev->prp_page_pool = dma_pool_create("prp list page", dev->dev,
						PAGE_SIZE, PAGE_SIZE, 0);
	if (!dev->prp_page_pool)
		return -ENOMEM;

	/* Optimisation for I/Os between 4k and 128k */
	dev->prp_small_pool = dma_pool_create("prp list 256", dev->dev,
						256, 256, 0);
	if (!dev->prp_small_pool) {
		dma_pool_destroy(dev->prp_page_pool);
		return -ENOMEM;
	}
	return 0;
}

static void nvme_release_prp_pools(struct nvme_dev *dev)
{
	dma_pool_destroy(dev->prp_page_pool);
	dma_pool_destroy(dev->prp_small_pool);
}

static void nvme_pci_free_ctrl(struct nvme_ctrl *ctrl)
{
	struct nvme_dev *dev = to_nvme_dev(ctrl);

	put_device(dev->dev);
	if (dev->tagset.tags)
		blk_mq_free_tag_set(&dev->tagset);
	if (dev->ctrl.admin_q)
		blk_put_queue(dev->ctrl.admin_q);
	kfree(dev->queues);
	kfree(dev);
}

static void nvme_remove_dead_ctrl(struct nvme_dev *dev, int status)
{
	dev_warn(dev->ctrl.device, "Removing after probe failure status: %d\n", status);

	kref_get(&dev->ctrl.kref);
	nvme_dev_disable(dev, false);
	if (!schedule_work(&dev->remove_work))
		nvme_put_ctrl(&dev->ctrl);
}

static void nvme_reset_work(struct work_struct *work)
{
	struct nvme_dev *dev = container_of(work, struct nvme_dev, reset_work);
	int result = -ENODEV;

	if (WARN_ON(dev->ctrl.state != NVME_CTRL_RESETTING))
		goto out;

	/*
	 * If we're called to reset a live controller first shut it down before
	 * moving on.
	 */
	if (dev->ctrl.ctrl_config & NVME_CC_ENABLE)
		nvme_dev_disable(dev, false);

	result = nvme_pci_enable(dev);
	if (result)
		goto out;

	result = nvme_configure_admin_queue(dev);
	if (result)
		goto out;

	result = nvme_alloc_admin_tags(dev);
	if (result)
		goto out;

	result = nvme_init_identify(&dev->ctrl);
	if (result)
		goto out;

	result = nvme_setup_io_queues(dev);
	if (result)
		goto out;

	/*
	 * A controller that can not execute IO typically requires user
	 * intervention to correct. For such degraded controllers, the driver
	 * should not submit commands the user did not request, so skip
	 * registering for asynchronous event notification on this condition.
	 */
	if (dev->online_queues > 1)
		nvme_queue_async_events(&dev->ctrl);

	mod_timer(&dev->watchdog_timer, round_jiffies(jiffies + HZ));

	/*
	 * Keep the controller around but remove all namespaces if we don't have
	 * any working I/O queue.
	 */
	if (dev->online_queues < 2) {
		dev_warn(dev->ctrl.device, "IO queues not created\n");
		nvme_kill_queues(&dev->ctrl);
		nvme_remove_namespaces(&dev->ctrl);
	} else {
		nvme_start_queues(&dev->ctrl);
		nvme_wait_freeze(&dev->ctrl);
		nvme_dev_add(dev);
		nvme_unfreeze(&dev->ctrl);
	}

	if (!nvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_LIVE)) {
		dev_warn(dev->ctrl.device, "failed to mark controller live\n");
		goto out;
	}

	if (dev->online_queues > 1)
		nvme_queue_scan(&dev->ctrl);
	return;

 out:
	nvme_remove_dead_ctrl(dev, result);
}

static void nvme_remove_dead_ctrl_work(struct work_struct *work)
{
	struct nvme_dev *dev = container_of(work, struct nvme_dev, remove_work);
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	nvme_kill_queues(&dev->ctrl);
	if (pci_get_drvdata(pdev))
		device_release_driver(&pdev->dev);
	nvme_put_ctrl(&dev->ctrl);
}

static int nvme_reset(struct nvme_dev *dev)
{
	if (!dev->ctrl.admin_q || blk_queue_dying(dev->ctrl.admin_q))
		return -ENODEV;
	if (!nvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_RESETTING))
		return -EBUSY;
	if (!queue_work(nvme_workq, &dev->reset_work))
		return -EBUSY;
	return 0;
}

static int nvme_pci_reg_read32(struct nvme_ctrl *ctrl, u32 off, u32 *val)
{
	*val = readl(to_nvme_dev(ctrl)->bar + off);
	return 0;
}

static int nvme_pci_reg_write32(struct nvme_ctrl *ctrl, u32 off, u32 val)
{
	writel(val, to_nvme_dev(ctrl)->bar + off);
	return 0;
}

static int nvme_pci_reg_read64(struct nvme_ctrl *ctrl, u32 off, u64 *val)
{
	*val = readq(to_nvme_dev(ctrl)->bar + off);
	return 0;
}

static int nvme_pci_reset_ctrl(struct nvme_ctrl *ctrl)
{
	struct nvme_dev *dev = to_nvme_dev(ctrl);
	int ret = nvme_reset(dev);

	if (!ret)
		flush_work(&dev->reset_work);
	return ret;
}

static const struct nvme_ctrl_ops nvme_pci_ctrl_ops = {
	.name			= "pcie",
	.module			= THIS_MODULE,
	.reg_read32		= nvme_pci_reg_read32,
	.reg_write32		= nvme_pci_reg_write32,
	.reg_read64		= nvme_pci_reg_read64,
	.reset_ctrl		= nvme_pci_reset_ctrl,
	.free_ctrl		= nvme_pci_free_ctrl,
	.submit_async_event	= nvme_pci_submit_async_event,
};

static int nvme_dev_map(struct nvme_dev *dev)
{
	struct pci_dev *pdev = to_pci_dev(dev->dev);

	if (pci_request_mem_regions(pdev, "nvme"))
		return -ENODEV;

	dev->bar = ioremap(pci_resource_start(pdev, 0), 8192);
	if (!dev->bar)
		goto release;

       return 0;
  release:
       pci_release_mem_regions(pdev);
       return -ENODEV;
}

static int nvme_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	int node, result = -ENOMEM;
	struct nvme_dev *dev;

	node = dev_to_node(&pdev->dev);
	if (node == NUMA_NO_NODE)
		set_dev_node(&pdev->dev, first_memory_node);

	dev = kzalloc_node(sizeof(*dev), GFP_KERNEL, node);
	if (!dev)
		return -ENOMEM;
	dev->queues = kzalloc_node((num_possible_cpus() + 1) * sizeof(void *),
							GFP_KERNEL, node);
	if (!dev->queues)
		goto free;

	dev->dev = get_device(&pdev->dev);
	pci_set_drvdata(pdev, dev);

	result = nvme_dev_map(dev);
	if (result)
		goto free;

	INIT_WORK(&dev->reset_work, nvme_reset_work);
	INIT_WORK(&dev->remove_work, nvme_remove_dead_ctrl_work);
	setup_timer(&dev->watchdog_timer, nvme_watchdog_timer,
		(unsigned long)dev);
	mutex_init(&dev->shutdown_lock);
	init_completion(&dev->ioq_wait);

	result = nvme_setup_prp_pools(dev);
	if (result)
		goto put_pci;

	result = nvme_init_ctrl(&dev->ctrl, &pdev->dev, &nvme_pci_ctrl_ops,
			id->driver_data);
	if (result)
		goto release_pools;

	nvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_RESETTING);
	dev_info(dev->ctrl.device, "pci function %s\n", dev_name(&pdev->dev));

	queue_work(nvme_workq, &dev->reset_work);
	return 0;

 release_pools:
	nvme_release_prp_pools(dev);
 put_pci:
	put_device(dev->dev);
	nvme_dev_unmap(dev);
 free:
	kfree(dev->queues);
	kfree(dev);
	return result;
}

static void nvme_reset_notify(struct pci_dev *pdev, bool prepare)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);

	if (prepare)
		nvme_dev_disable(dev, false);
	else
		nvme_reset(dev);
}

static void nvme_shutdown(struct pci_dev *pdev)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);
	nvme_dev_disable(dev, true);
}

/*
 * The driver's remove may be called on a device in a partially initialized
 * state. This function must not have any dependencies on the device state in
 * order to proceed.
 */
static void nvme_remove(struct pci_dev *pdev)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);

	nvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_DELETING);

	cancel_work_sync(&dev->reset_work);
	pci_set_drvdata(pdev, NULL);

	if (!pci_device_is_present(pdev)) {
		nvme_change_ctrl_state(&dev->ctrl, NVME_CTRL_DEAD);
		nvme_dev_disable(dev, false);
	}

	flush_work(&dev->reset_work);
	nvme_uninit_ctrl(&dev->ctrl);
	nvme_dev_disable(dev, true);
	nvme_dev_remove_admin(dev);
	nvme_free_queues(dev, 0);
	nvme_release_prp_pools(dev);
	nvme_dev_unmap(dev);
	nvme_put_ctrl(&dev->ctrl);
}

static int nvme_pci_sriov_configure(struct pci_dev *pdev, int numvfs)
{
	int ret = 0;

	if (numvfs == 0) {
		if (pci_vfs_assigned(pdev)) {
			dev_warn(&pdev->dev,
				"Cannot disable SR-IOV VFs while assigned\n");
			return -EPERM;
		}
		pci_disable_sriov(pdev);
		return 0;
	}

	ret = pci_enable_sriov(pdev, numvfs);
	return ret ? ret : numvfs;
}

#ifdef CONFIG_PM_SLEEP
static int nvme_suspend(struct device *dev)
{
	struct pci_dev *pdev = to_pci_dev(dev);
	struct nvme_dev *ndev = pci_get_drvdata(pdev);

	nvme_dev_disable(ndev, true);
	return 0;
}

static int nvme_resume(struct device *dev)
{
	struct pci_dev *pdev = to_pci_dev(dev);
	struct nvme_dev *ndev = pci_get_drvdata(pdev);

	nvme_reset(ndev);
	return 0;
}
#endif

static SIMPLE_DEV_PM_OPS(nvme_dev_pm_ops, nvme_suspend, nvme_resume);

static pci_ers_result_t nvme_error_detected(struct pci_dev *pdev,
						pci_channel_state_t state)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);

	/*
	 * A frozen channel requires a reset. When detected, this method will
	 * shutdown the controller to quiesce. The controller will be restarted
	 * after the slot reset through driver's slot_reset callback.
	 */
	switch (state) {
	case pci_channel_io_normal:
		return PCI_ERS_RESULT_CAN_RECOVER;
	case pci_channel_io_frozen:
		dev_warn(dev->ctrl.device,
			"frozen state error detected, reset controller\n");
		nvme_dev_disable(dev, false);
		return PCI_ERS_RESULT_NEED_RESET;
	case pci_channel_io_perm_failure:
		dev_warn(dev->ctrl.device,
			"failure state error detected, request disconnect\n");
		return PCI_ERS_RESULT_DISCONNECT;
	}
	return PCI_ERS_RESULT_NEED_RESET;
}

static pci_ers_result_t nvme_slot_reset(struct pci_dev *pdev)
{
	struct nvme_dev *dev = pci_get_drvdata(pdev);

	dev_info(dev->ctrl.device, "restart after slot reset\n");
	pci_restore_state(pdev);
	nvme_reset(dev);
	return PCI_ERS_RESULT_RECOVERED;
}

static void nvme_error_resume(struct pci_dev *pdev)
{
	pci_cleanup_aer_uncorrect_error_status(pdev);
}

static const struct pci_error_handlers nvme_err_handler = {
	.error_detected	= nvme_error_detected,
	.slot_reset	= nvme_slot_reset,
	.resume		= nvme_error_resume,
	.reset_notify	= nvme_reset_notify,
};

/* Move to pci_ids.h later */
#define PCI_CLASS_STORAGE_EXPRESS	0x010802

static const struct pci_device_id nvme_id_table[] = {
	{ PCI_VDEVICE(INTEL, 0x0953),
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DISCARD_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0x0a53),
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DISCARD_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0x0a54),
		.driver_data = NVME_QUIRK_STRIPE_SIZE |
				NVME_QUIRK_DISCARD_ZEROES, },
	{ PCI_VDEVICE(INTEL, 0x5845),	/* Qemu emulated controller */
		.driver_data = NVME_QUIRK_IDENTIFY_CNS, },
	{ PCI_DEVICE(0x1c58, 0x0003),	/* HGST adapter */
		.driver_data = NVME_QUIRK_DELAY_BEFORE_CHK_RDY, },
	{ PCI_DEVICE(0x1c58, 0x0023),	/* WDC SN200 adapter */
		.driver_data = NVME_QUIRK_DELAY_BEFORE_CHK_RDY, },
	{ PCI_DEVICE(0x1c5f, 0x0540),	/* Memblaze Pblaze4 adapter */
		.driver_data = NVME_QUIRK_DELAY_BEFORE_CHK_RDY, },
	{ PCI_DEVICE_CLASS(PCI_CLASS_STORAGE_EXPRESS, 0xffffff) },
	{ PCI_DEVICE(PCI_VENDOR_ID_APPLE, 0x2001) },
	{ 0, }
};
MODULE_DEVICE_TABLE(pci, nvme_id_table);

static struct pci_driver nvme_driver = {
	.name		= "nvme",
	.id_table	= nvme_id_table,
	.probe		= nvme_probe,
	.remove		= nvme_remove,
	.shutdown	= nvme_shutdown,
	.driver		= {
		.pm	= &nvme_dev_pm_ops,
	},
	.sriov_configure = nvme_pci_sriov_configure,
	.err_handler	= &nvme_err_handler,
};

static int __init nvme_init(void)
{
	int result;

	nvme_workq = alloc_workqueue("nvme", WQ_UNBOUND | WQ_MEM_RECLAIM, 0);
	if (!nvme_workq)
		return -ENOMEM;

	result = pci_register_driver(&nvme_driver);
	if (result)
		destroy_workqueue(nvme_workq);
	return result;
}

static void __exit nvme_exit(void)
{
	pci_unregister_driver(&nvme_driver);
	destroy_workqueue(nvme_workq);
	_nvme_check_size();
}

/*key_value*/
#define MODE_KEY 0
#define MODE_P2P 1
typedef struct key_value_operations {
	long(*open) (const char __user*);
	long(*close) (const char __user*);
	long(*key_op) (int, unsigned long, char __user *, unsigned int);
	long(*key_p2p_op) (int, unsigned long, unsigned long, unsigned int);
	long(*key_batch_op) (unsigned long, int);
	long(*map_p2p_pages)(unsigned long, unsigned long, unsigned long, unsigned);
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
	/////////for batch_sync
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

#define BUF_NEXT(a,num) ((a+1)%num)


static void batch_completion(struct nvme_queue *nvmeq, void *ctx, struct nvme_completion *cqe) { //async
	struct batch_cmd_info *cmdinfo = ctx;
	unsigned long flags;
	//int r;
	cmdinfo->res->status = le16_to_cpup(&cqe->status) >> 1;
	blk_mq_free_hctx_request(nvmeq->hctx, cmdinfo->req);

	spin_lock_irqsave(&(cmdinfo->poll->lock),flags);
	cmdinfo->poll->node[cmdinfo->poll->r]=cmdinfo->num;
	cmdinfo->poll->r=BUF_NEXT(cmdinfo->poll->r,cmdinfo->poll->size);
	spin_unlock_irqrestore(&(cmdinfo->poll->lock),flags);
	wake_up_process(cmdinfo->task);

}
static void batch_completion_sync(struct nvme_queue *nvmeq, void *ctx, struct nvme_completion *cqe) { //sync
	struct batch_cmd_info *cmdinfo = ctx;
	unsigned long flags;
	cmdinfo->res->status = le16_to_cpup(&cqe->status) >> 1;
	blk_mq_free_hctx_request(nvmeq->hctx, cmdinfo->req);

	spin_lock_irqsave(&(cmdinfo->end->lock), flags);
//	cmdinfo->end->cnt++;
	if((++(cmdinfo->end->cnt))==cmdinfo->end->end){
		wake_up_process(cmdinfo->task);
	}
	spin_unlock_irqrestore(&(cmdinfo->end->lock),flags);
}

typedef void (*nvme_completion_fn)(struct nvme_queue *, void *,
						struct nvme_completion *);


struct nvme_cmd_info {
	nvme_completion_fn fn;
	void *ctx;
	int aborted;
	struct nvme_queue *nvmeq;
	struct nvme_iod iod[0];
};

static void nvme_set_info(struct nvme_cmd_info *cmd, void *ctx,
				nvme_completion_fn handler)
{
	cmd->fn = handler;
	cmd->ctx = ctx;
	cmd->aborted = 0;
	blk_mq_start_request(blk_mq_rq_from_pdu(cmd));
}

static int nvme_submit_cmd(struct nvme_queue *nvmeq, struct nvme_command *cmd)
{
	unsigned long flags;
	int ret = 0;
	spin_lock_irqsave(&nvmeq->q_lock, flags);
	__nvme_submit_cmd(nvmeq, cmd); //원래함수도항상리턴0이라걍ret = 0 을해줌
	spin_unlock_irqrestore(&nvmeq->q_lock, flags);
	return ret;
}


extern int Key_value_module;
extern KEY_OPERATIONS* Key_op;
int nvme_key_batch_command(struct block_device *bdev, unsigned long arg, int len,int mode) { //only p2p
	struct nvme_ns *ns = bdev->bd_disk->private_data;
//	struct nvme_dev *dev = ns->dev; no used
	Command_list **list = (Command_list**)arg;
	//struct nvme_queue *nvmeq;
	int i;
//,length; no used so jusuk
//	unsigned timeout; no used so jusuk
	
	if(mode==0){/////sync
		for(i=0;i<len;i++){
			struct request *req;
			struct nvme_cmd_info *cmd_rq;

			req = blk_mq_alloc_request(ns->queue, WRITE, 1); //__GFP_WAIT 빼면 어떻게 되나?  | __GFP_WAIT
			if (IS_ERR(req)) {
				printk("request allocation fail!!!!!!\n"); return -1;
			}
			req->timeout = list[i]->timeout;
			cmd_rq = blk_mq_rq_to_pdu(req);
			list[i]->cmdinfo.req = req;
			list[i]->cmdinfo.task=current;

			nvme_set_info(cmd_rq, &(list[i]->cmdinfo), batch_completion_sync);

			list[i]->c.common.command_id = req->tag;
			
			if(unlikely(i==len-1)){
				set_current_state(TASK_UNINTERRUPTIBLE);
				nvme_submit_cmd(cmd_rq->nvmeq,&(list[i]->c) );
				schedule();
			}
			else{
				nvme_submit_cmd(cmd_rq->nvmeq,&(list[i]->c) );
			}
		}
	}
	else{
		for (i = 0; i < len; i++) {
			struct request *req;
			struct nvme_cmd_info *cmd_rq;
			//printk("loop___KEY_BATCH | i : %d\n", i);
			req = blk_mq_alloc_request(ns->queue, WRITE, 1); //__GFP_WAIT 빼면 어떻게 되나?  | __GFP_WAIT
			if (IS_ERR(req)) {
				printk("request allocation fail!!!!!!\n"); return -1;
			}
			req->timeout = list[i]->timeout;
			cmd_rq = blk_mq_rq_to_pdu(req);
			list[i]->cmdinfo.req = req;
			nvme_set_info(cmd_rq, &(list[i]->cmdinfo), batch_completion);
			list[i]->c.common.command_id = req->tag;

			nvme_submit_cmd(cmd_rq->nvmeq,&(list[i]->c) );
		}
	}
	/*
	/////nvme_submit_cmd
	unsigned long flags;
	int ret;
	spin_lock_irqsave(&nvmeq->q_lock, flags);
	ret = __nvme_submit_cmd(nvmeq, cmd);
	spin_unlock_irqrestore(&nvmeq->q_lock, flags);
	return ret;
	*/
	return 0;

}
EXPORT_SYMBOL_GPL(nvme_key_batch_command); //key_value
int nvme_key_command(struct block_device *bdev, unsigned long arg,int mode){
	struct nvme_ns *ns=bdev->bd_disk->private_data;
	struct nvme_dev *dev=ns->dev;

	struct nvme_passthru_cmd *cmd=(struct nvme_passthru_cmd *)arg;
	struct nvme_command c;
	int status;
	unsigned length;
	struct nvme_iod *uninitialized_var(iod);
	unsigned timeout;
	struct request*tmpReq;

	memset(&c, 0, sizeof(c));
	c.common.opcode = cmd->opcode;
	c.common.flags = cmd->flags;
	c.common.nsid = cpu_to_le32(cmd->nsid);
	c.common.cdw2[0] = cpu_to_le32(cmd->cdw2);
	c.common.cdw2[1] = cpu_to_le32(cmd->cdw3);
	c.common.cdw10[0] = cpu_to_le32(cmd->cdw10);
	c.common.cdw10[1] = cpu_to_le32(cmd->cdw11);
	c.common.cdw10[2] = cpu_to_le32(cmd->cdw12);
	c.common.cdw10[3] = cpu_to_le32(cmd->cdw13);
	c.common.cdw10[4] = cpu_to_le32(cmd->cdw14);
	c.common.cdw10[5] = cpu_to_le32(cmd->cdw15);

	length = cmd->data_len;
	///////////////////////////// should change here code!
	if(mode==MODE_P2P){ 
//		printk("start P2P!\n");
	//	iod=nvme_map_p2p_pages(dev,cmd->addr,length);
		if(Key_op->map_p2p_pages((unsigned long)(&iod), (unsigned long)dev,cmd->addr,length)<0){
			printk("key_p2p_op : map_p2p_pages fail!\n"); return -1;
		}
		tmpReq = blk_mq_rq_from_pdu(iod);
		tmpReq = blk_mq_alloc_request(ns->queue, WRITE, 1);
//		length=nvme_setup_prps(dev,iod,length,GFP_KERNEL);
		length=nvme_setup_prps(dev,tmpReq,length);
		c.common.dptr.prp1=cpu_to_le64(sg_dma_address(iod->sg));
		c.common.dptr.prp2=cpu_to_le64(iod->first_dma);
//		printk("PRP1 : %lx, PRP2 : %lx\n",c.common.prp1,c.common.prp2);
	}
	else{
		if (cmd->data_len) {
			//			printk("start USER IO!\n");
			iod = nvme_map_user_pages(dev, cmd->opcode & 1, cmd->addr, length);
			if (IS_ERR(iod)){
				printk("nvme_key_command | iod alloc error!\n");
				return PTR_ERR(iod);
			}
			tmpReq = blk_mq_rq_from_pdu(iod);
			tmpReq = blk_mq_alloc_request(ns->queue, WRITE, 1);
	//		length=nvme_setup_prps(dev,iod,length,GFP_KERNEL);
			length=nvme_setup_prps(dev,tmpReq,length);
			c.common.dptr.prp1 = cpu_to_le64(sg_dma_address(iod->sg));
			c.common.dptr.prp2 = cpu_to_le64(iod->first_dma);
//		printk("PRP1 : %lx, PRP2 : %lx\n",c.common.prp1,c.common.prp2);
		}
	}

	timeout = cmd->timeout_ms ? msecs_to_jiffies(cmd->timeout_ms) :
								ADMIN_TIMEOUT;

	if (length != cmd->data_len){
		printk("nvme_key_command | length != cmd->data_len");
		status = -ENOMEM;
	}
	else if (ns) {
		// Same as nvme_submit_io_cmd()
		//struct request *req;
		
/*
		req = blk_mq_alloc_request(ns->queue, WRITE, 1);
		if (IS_ERR(req))
			status = PTR_ERR(req);
		else {
			status = nvme_submit_sync_cmd(req, &c, &cmd->result,
								timeout);
			blk_mq_free_request(req);
		}
*/
		status = nvme_submit_sync_cmd(ns->queue, &c, &cmd->result,
								timeout);
	} else{
		//curr queue x -> admin queue
		//status = __nvme_submit_admin_cmd(dev, &c, &cmd->result, timeout);
		status = nvme_submit_sync_cmd(dev->admin_q, &c, &cmd->result, timeout);
	}
	if(mode==MODE_P2P && cmd->data_len){
		tmpReq = blk_mq_rq_from_pdu(iod);
		tmpReq = blk_mq_alloc_request(ns->queue, WRITE, 1);
		nvme_free_iod(dev,tmpReq);
//		nvme_free_iod(dev, iod);
	}
	else if (cmd->data_len) {
		nvme_unmap_user_pages(dev, cmd->opcode & 1, iod);
		tmpReq = blk_mq_rq_from_pdu(iod);
		tmpReq = blk_mq_alloc_request(ns->queue, WRITE, 1);
		nvme_free_iod(dev,tmpReq);
//		nvme_free_iod(dev, iod);
	}

//	if ((status >= 0) && copy_to_user(&ucmd->result, &cmd.result,
//							sizeof(cmd.result)))
//		status = -EFAULT;

	return status;
}
EXPORT_SYMBOL_GPL(nvme_key_command); //key_value
unsigned int nvme_key_max_segments(struct block_device *bdev){
	unsigned int ret;
	struct nvme_ns *ns=bdev->bd_disk->private_data;
    ret=queue_max_hw_sectors(ns->queue);
    return ret;
}
EXPORT_SYMBOL_GPL(nvme_key_max_segments); //key_value

/////////////

/*key_value cp & paste*/
static inline void iod_init(struct nvme_iod *iod, unsigned nbytes,
				unsigned nseg, unsigned long private)
{
//	iod->private = private;
//	iod->offset = offsetof(struct nvme_iod, sg[nseg]);
	iod->npages = -1;
	iod->length = nbytes;
	iod->nents = 0;
	iod->first_dma = 0; ///////key_value
}

static struct nvme_iod *
__nvme_alloc_iod(unsigned nseg, unsigned bytes, struct nvme_dev *dev,
		 unsigned long priv, gfp_t gfp)
{
	struct nvme_iod *iod = kmalloc(sizeof(struct nvme_iod) +
				sizeof(__le64 *) * nvme_npages(bytes, dev) +
				sizeof(struct scatterlist) * nseg, gfp);

	if (iod)
		iod_init(iod, bytes, nseg, priv);

	return iod;
}


struct nvme_iod *nvme_map_user_pages(struct nvme_dev *dev, int write,
				unsigned long addr, unsigned length)
{
	int i, err, count, nents, offset;
	struct scatterlist *sg;
	struct page **pages;
	struct nvme_iod *iod;

	if (addr & 3)
		return ERR_PTR(-EINVAL);
	if (!length || length > INT_MAX - PAGE_SIZE)
		return ERR_PTR(-EINVAL);

	offset = offset_in_page(addr);
	count = DIV_ROUND_UP(offset + length, PAGE_SIZE);
	pages = kcalloc(count, sizeof(*pages), GFP_KERNEL);

//	printk("nvme_map_user_pages, offset :%d, count :%d\n",offset,count); //key_value

	if (!pages)
		return ERR_PTR(-ENOMEM);

	err = get_user_pages_fast(addr, count, 1, pages);
	if (err < count) {
		count = err;
		err = -EFAULT;
		goto put_pages;
	}

	err = -ENOMEM;
	iod = __nvme_alloc_iod(count, length, dev, 0, GFP_KERNEL);
	if (!iod)
		goto put_pages;

	sg = iod->sg;
	sg_init_table(sg, count);
	for (i = 0; i < count; i++) {
		sg_set_page(&sg[i], pages[i],
				min_t(unsigned, length, PAGE_SIZE - offset),
				offset);
		length -= (PAGE_SIZE - offset);
		offset = 0;
	}
	sg_mark_end(&sg[i - 1]);
	iod->nents = count;

	nents = dma_map_sg(&dev->pci_dev->dev, sg, count,
				write ? DMA_TO_DEVICE : DMA_FROM_DEVICE);
	if (!nents)
		goto free_iod;

	kfree(pages);
	return iod;

 free_iod:
	kfree(iod);
 put_pages:
	for (i = 0; i < count; i++)
		put_page(pages[i]);
	kfree(pages);
	return ERR_PTR(err);
}
EXPORT_SYMBOL_GPL(nvme_map_user_pages); //key_value

void nvme_unmap_user_pages(struct nvme_dev *dev, int write,
			struct nvme_iod *iod)
{
	int i;

	dma_unmap_sg(&dev->pci_dev->dev, iod->sg, iod->nents,
				write ? DMA_TO_DEVICE : DMA_FROM_DEVICE);

	for (i = 0; i < iod->nents; i++)
		put_page(sg_page(&iod->sg[i]));
}
EXPORT_SYMBOL_GPL(nvme_unmap_user_pages); //key_value
///

module_init(key_value_init);
module_exit(key_value_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Donggyu Park");
MODULE_DESCRIPTION("key_value_ module");
MODULE_VERSION("0.1");
