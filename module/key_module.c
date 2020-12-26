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

extern nvme_iod*nvme_map_user_pages(struct nvme_dev *dev, int write, unsigned long addr, unsigned length);
extern bool nvme_setup_prps(struct nvme_dev *dev, struct request *req, int total_len);
extern int nvme_key_batch_command(struct block_device *bdev, unsigned long arg, int len,int mode);
extern void nvme_unmap_user_pages(struct nvme_dev *dev, int write, struct nvme_iod *iod);
extern void nvme_free_iod(struct nvme_dev *dev, struct request *req);


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

module_init(key_value_init);
module_exit(key_value_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Donggyu Park");
MODULE_DESCRIPTION("key_value_ module");
MODULE_VERSION("0.1");
