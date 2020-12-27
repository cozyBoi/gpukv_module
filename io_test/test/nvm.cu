#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <getopt.h>
#include <stdbool.h>
#include <nvm_types.h>
#include <nvm_ctrl.h>
#include <nvm_dma.h>
#include <nvm_aq.h>
#include <nvm_admin.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/vfio.h>



#include <nvm_error.h>
#include <nvm_util.h>

struct bdf
{
	int     domain;
	int     bus;
	int     device;
	int     function;
	int     fd;
	int     vfio_group;
	int     vfio_cfd;
};

nvm_aq_ref reset_ctrl(const nvm_ctrl_t* ctrl, const nvm_dma_t* dma_window);
int identify_ctrl(nvm_aq_ref admin, void* ptr, uint64_t ioaddr);
int identify_ns(nvm_aq_ref admin, uint32_t nvm_namespace, void* ptr, uint64_t ioaddr);

static int pci_ioaddrs_iommu(int vfio_cfd, void* ptr, size_t page_size,
		size_t n_pages, uint64_t* ioaddrs) {
	for (size_t i_page = 0; i_page < n_pages; ++i_page)
	{
		struct vfio_iommu_type1_dma_map dma_map = { .argsz = sizeof(dma_map) };
		dma_map.vaddr = (uint64_t)ptr + (i_page * page_size);
		dma_map.size = page_size;
		dma_map.iova = ioaddrs[i_page];
		dma_map.flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE;

		int rc = ioctl(vfio_cfd, VFIO_IOMMU_MAP_DMA, &dma_map);
		if (rc != 0)
			return -errno;
	}
	return 0;
}

static int lookup_ioaddrs(void* ptr, size_t page_size, size_t n_pages, uint64_t* ioaddrs)
{
	uint64_t vaddr = (uint64_t) ptr;
	size_t offset = (vaddr / page_size) * sizeof(void*);

	FILE* fp = fopen("/proc/self/pagemap", "r");
	if (fp == NULL)
	{
		fprintf(stderr, "Failed to open page map: %s\n", strerror(errno));
		return errno;
	}

	if (fseek(fp, offset, SEEK_SET) != 0)
	{
		fclose(fp);
		fprintf(stderr, "Failed to seek: %s\n", strerror(errno));
		return errno;
	}

	if (fread(ioaddrs, sizeof(uint64_t), n_pages, fp) != n_pages)
	{
		fclose(fp);
		fprintf(stderr, "Failed to read: %s\n", strerror(errno));
		return errno;
	}

	fclose(fp);

	for (size_t i_page = 0; i_page < n_pages; ++i_page)
	{
		if (!(ioaddrs[i_page] & (1ULL << 63)))
		{
			fprintf(stderr, "Page not present in memory!\n");
			return EINVAL;
		}

		ioaddrs[i_page] = (ioaddrs[i_page] & ((1ULL << 54) - 1)) * page_size;
	}

	return 0;
}

static int identify(const nvm_ctrl_t* ctrl, uint32_t nvm_ns_id, int vfio_cfd)
{
	int status;
	void* memory;
	nvm_dma_t* window = NULL;
	nvm_aq_ref admin = NULL;
	uint64_t ioaddrs[3];

	long page_size = sysconf(_SC_PAGESIZE);
	if (page_size == -1)
	{
		fprintf(stderr, "Failed to look up page size: %s\n", strerror(errno));
		return 1;
	}

	status = posix_memalign(&memory, ctrl->page_size, 3 * page_size);
	if (status != 0)
	{
		fprintf(stderr, "Failed to allocate page-aligned memory: %s\n", strerror(status));
		return 1;
	}

	status = mlock(memory, 3 * page_size);
	if (status != 0)
	{
		free(memory);
		fprintf(stderr, "Failed to page-lock memory: %s\n", strerror(status));
		return 1;
	}

	status = lookup_ioaddrs(memory, page_size, 3, ioaddrs);
	if (status != 0)
	{
		munlock(memory, 3 * page_size);
		free(memory);
		goto out;
	}

	if (vfio_cfd >= 0) {
		status = pci_ioaddrs_iommu(vfio_cfd, memory, page_size, 3, ioaddrs);
		if (status != 0)
		{
			munlock(memory, 3 * page_size);
			free(memory);
			goto out;
		}
	}

	status = nvm_dma_map(&window, ctrl, memory, page_size, 3, ioaddrs);
	if (status != 0)
	{
		fprintf(stderr, "Failed to create DMA window: %s\n", strerror(status));
		status = 2;
		goto out;
	}

	admin = reset_ctrl(ctrl, window);
	if (admin == NULL)
	{
		goto out;
	}

	status = identify_ctrl(admin, ((unsigned char*) memory) + 2 * ctrl->page_size, ioaddrs[2]);
	if (status != 0)
	{
		goto out;
	}

	if (nvm_ns_id != 0)
	{
		status = identify_ns(admin, nvm_ns_id, ((unsigned char*) memory) + 2 * ctrl->page_size, ioaddrs[2]);
	}

out:
	nvm_aq_destroy(admin);
	nvm_dma_unmap(window);
	munlock(memory, 3 * page_size);
	free(memory);
	return status;
}


static int pci_enable_device(const struct bdf* dev)
{
	char path[64];
	sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/enable",
			dev->domain, dev->bus, dev->device, dev->function);
	printf("path: %s\n", path);

	FILE *fp = fopen(path, "w");
	if (fp == NULL)
	{
		fprintf(stderr, "Failed to open file descriptor: %s\n", strerror(errno));
		return errno;
	}

	fputc('1', fp);
	fclose(fp);
	return 0;
}


/*
 * Allow device to do DMA.
 */
static int pci_set_bus_master(const struct bdf* dev)
{
	char path[64];
	sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/config", 
			dev->domain, dev->bus, dev->device, dev->function);

	FILE* fp = fopen(path, "r+");
	if (fp == NULL)
	{
		fprintf(stderr, "Failed to open config space file: %s\n", strerror(errno));
		return errno;
	}

	uint16_t command;
	fseek(fp, 0x04, SEEK_SET);
	fread(&command, sizeof(command), 1, fp);

	if ((command & (1 << 0x02)) == 0) {
		command |= (1 << 0x02);

		fseek(fp, 0x04, SEEK_SET);
		fwrite(&command, sizeof(command), 1, fp);
	}

	fclose(fp);
	return 0;
}


/*
 * Open a file descriptor to device memory.
 */
static int pci_open_bar(struct bdf* dev, int bar)
{
	char path[64];
	sprintf(path, "/sys/bus/pci/devices/%04x:%02x:%02x.%x/resource%d", 
			dev->domain, dev->bus, dev->device, dev->function, bar);

	dev->fd = open(path, O_RDWR);
	if (dev->fd < 0)
	{
		fprintf(stderr, "Failed to open resource file: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

static int pci_open_vfio(struct bdf* dev, int bar, volatile void** ctrl_registers)
{
	char path[64];
	int group_fd, rc;
	struct vfio_group_status group_status;
	struct vfio_region_info region_info; 
	region_info.argsz = sizeof(region_info);
	region_info.index = VFIO_PCI_BAR0_REGION_INDEX;

	__u32 required_flags;

	if (bar != 0) {
		fprintf(stderr, "Only support mapping BAR 0\n");
		return -1;
	}

	/* Create a new container */
	dev->vfio_cfd = open("/dev/vfio/vfio", O_RDWR);
	if (dev->vfio_cfd < 0) {
		fprintf(stderr, "VFIO: Error opening /dev/vfio/vfio (%d)\n", errno);
		return -1;
	}

	if (ioctl(dev->vfio_cfd, VFIO_GET_API_VERSION) != VFIO_API_VERSION) {
		fprintf(stderr, "VFIO: Unknown API version\n");
		goto error_close_container_fd;
	}

	if (!ioctl(dev->vfio_cfd, VFIO_CHECK_EXTENSION, VFIO_TYPE1_IOMMU)) {
		fprintf(stderr, "VFIO: Doesn't support the VFIO_TYPE1_IOMMU driver (%d)\n", errno);
		goto error_close_container_fd;
	}

	/* Open the group */
	sprintf(path, "/dev/vfio/%d", dev->vfio_group);
	group_fd = open(path, O_RDWR);
	if (group_fd < 0) {
		fprintf(stderr, "VFIO: Error opening %s (%d)\n", path, errno);
		goto error_close_container_fd;
	}

	/* Test the group is viable and available */
	bzero(&group_status, sizeof(group_status));
	group_status.argsz = sizeof(group_status);
	rc = ioctl(group_fd, VFIO_GROUP_GET_STATUS, &group_status);
	if (rc != 0) {
		fprintf(stderr, "VFIO: Failed to get group status rc=%d, errno=%d\n", rc, errno);
		goto error_close_group_fd;
	}
	if (!(group_status.flags & VFIO_GROUP_FLAGS_VIABLE)) {
		fprintf(stderr, "VFIO: Group is not viable (ie, not all devices bound for vfio)\n");
		goto error_close_group_fd;
	}

	/* Add the group to the container */
	rc = ioctl(group_fd, VFIO_GROUP_SET_CONTAINER, &dev->vfio_cfd);
	if (rc != 0) {
		fprintf(stderr, "VFIO: Failed to set group to container (%d)\n", errno);
		goto error_close_group_fd;
	}

	/* Enable the IOMMU model we want */
	if (ioctl(dev->vfio_cfd, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU) != 0) {
		fprintf(stderr, "VFIO: Failed to set IOMMU model (%d)\n", errno);
		goto error_close_group_fd;
	}

	/* Get a file descriptor for the device */
	sprintf(path, "%04x:%02x:%02x.%x",
			dev->domain, dev->bus, dev->device, dev->function);
	dev->fd = ioctl(group_fd, VFIO_GROUP_GET_DEVICE_FD, path);
	if (dev->fd < 0) {
		fprintf(stderr, "VFIO: Error opening device %s (%d)\n", path, errno);
		goto error_close_group_fd;
	}

	/* Test and setup the device */
	rc = ioctl(dev->fd, VFIO_DEVICE_GET_REGION_INFO, &region_info);
	if (rc < 0) {
		fprintf(stderr, "VFIO: Error get device %s info (%d)\n", path, errno);
		goto error_close_dev_fd;
	}

	required_flags = VFIO_REGION_INFO_FLAG_READ |
		VFIO_REGION_INFO_FLAG_WRITE |
		VFIO_REGION_INFO_FLAG_MMAP;
	if ((region_info.flags & required_flags) != required_flags) {
		fprintf(stderr, "VFIO: Device %s does not have required flags\n", path);
		goto error_close_dev_fd;
	}

	if (region_info.size < NVM_CTRL_MEM_MINSIZE) {
		fprintf(stderr, "VFIO: Device %s region too small %llu < %u\n",
				path, region_info.size, NVM_CTRL_MEM_MINSIZE);
		goto error_close_dev_fd;
	}

	*ctrl_registers = mmap(NULL, NVM_CTRL_MEM_MINSIZE,
			PROT_READ | PROT_WRITE,
			MAP_SHARED, dev->fd, region_info.offset);
	if (*ctrl_registers == NULL || *ctrl_registers == MAP_FAILED)
	{
		fprintf(stderr, "Failed to memory map BAR reasource file: %s\n", strerror(errno));
		goto error_close_dev_fd;
	}

	/* Gratuitous device reset and go... */
	rc = ioctl(dev->fd, VFIO_DEVICE_RESET);
	if (rc < 0) {
		fprintf(stderr, "VFIO: Error reset device %s (%d)\n", path, errno);
		goto error_close_dev_fd;
	}

	close(group_fd);
	return 0;

error_close_dev_fd:
	close(dev->fd);
	dev->fd = -1;
error_close_group_fd:
	close(group_fd);
error_close_container_fd:
	close(dev->vfio_cfd);
	dev->vfio_cfd = -1;
	return -1;
}

static void print_ctrl_info(FILE* fp, const struct nvm_ctrl_info* info, uint16_t n_cqs, uint16_t n_sqs)
{
	unsigned char vendor[4];
	memcpy(vendor, &info->pci_vendor, sizeof(vendor));

	char serial[21];
	memset(serial, 0, 21);
	memcpy(serial, info->serial_no, 20);

	char model[41];
	memset(model, 0, 41);
	memcpy(model, info->model_no, 40);

	char revision[9];
	memset(revision, 0, 9);
	memcpy(revision, info->firmware, 8);

	fprintf(fp, "------------- Controller information -------------\n");
	fprintf(fp, "PCI Vendor ID           : %x %x\n", vendor[0], vendor[1]);
	fprintf(fp, "PCI Subsystem Vendor ID : %x %x\n", vendor[2], vendor[3]);
	fprintf(fp, "NVM Express version     : %u.%u.%u\n",
			info->nvme_version >> 16, (info->nvme_version >> 8) & 0xff, info->nvme_version & 0xff);
	fprintf(fp, "Controller page size    : %zu\n", info->page_size);
	fprintf(fp, "Max queue entries       : %u\n", info->max_entries);
	fprintf(fp, "Serial Number           : %s\n", serial);
	fprintf(fp, "Model Number            : %s\n", model);
	fprintf(fp, "Firmware revision       : %s\n", revision);
	fprintf(fp, "Max data transfer size  : %zu\n", info->max_data_size);
	fprintf(fp, "Max outstanding commands: %zu\n", info->max_out_cmds);
	fprintf(fp, "Max number of namespaces: %zu\n", info->max_n_ns);
	fprintf(fp, "Current number of CQs   : %u\n", n_cqs);
	fprintf(fp, "Current number of SQs   : %u\n", n_sqs);
	fprintf(fp, "--------------------------------------------------\n");
}


/*
 * Print namespace information.
 */
static void print_ns_info(FILE* fp, const struct nvm_ns_info* info)
{
	fprintf(fp, "------------- Namespace  information -------------\n");
	fprintf(fp, "Namespace identifier    : %x\n", info->ns_id);
	fprintf(fp, "Logical block size      : %zu bytes\n", info->lba_data_size);
	fprintf(fp, "Namespace size          : %zu blocks\n", info->size);
	fprintf(fp, "Namespace capacity      : %zu blocks\n", info->capacity);
	fprintf(fp, "--------------------------------------------------\n");
}



nvm_aq_ref reset_ctrl(const nvm_ctrl_t* ctrl, const nvm_dma_t* dma_window)
{
	int status;
	nvm_aq_ref admin;

	if (dma_window->n_ioaddrs < 2)
	{
		return NULL;
	}
	memset(dma_window->vaddr, 0, dma_window->page_size * 2);

	fprintf(stderr, "Resetting controller and setting up admin queues...\n");
	status = nvm_aq_create(&admin, ctrl, dma_window);
	if (status != 0)
	{
		fprintf(stderr, "Failed to reset controller: %s\n", strerror(status));
		return NULL;
	}

	return admin;
}



int identify_ctrl(nvm_aq_ref admin, void* ptr, uint64_t ioaddr)
{
	int status;
	uint16_t n_cqs = 0;
	uint16_t n_sqs = 0;
	struct nvm_ctrl_info info;

	status = nvm_admin_get_num_queues(admin, &n_cqs, &n_sqs);
	if (status != 0)
	{
		fprintf(stderr, "Failed to get number of queues\n");
		return status;
	}

	status = nvm_admin_ctrl_info(admin, &info, ptr, ioaddr);
	if (status != 0)
	{
		fprintf(stderr, "Failed to identify controller: %s\n", strerror(status));
		return status;
	}

	print_ctrl_info(stdout, &info, n_cqs, n_sqs);
	return 0;
}



int identify_ns(nvm_aq_ref admin, uint32_t nvm_namespace, void* ptr, uint64_t ioaddr)
{
	int status;
	struct nvm_ns_info info;

	status = nvm_admin_ns_info(admin, &info, nvm_namespace, ptr, ioaddr);
	if (status != 0)
	{
		fprintf(stderr, "Failed to identify namespace: %s\n", strerror(status));
		return status;
	}

	print_ns_info(stdout, &info);
	return 0;
}

int main(){
	struct bdf device={.domain=0, .bus=2, .device=0, .function=0};
	int status;
	void *memory;
	uint32_t nvm_ns_id;
	nvm_ctrl_t *ctrl;
	volatile void* ctrl_registers = NULL;

	device.vfio_group = -1;
	device.vfio_cfd = -1;

	status = pci_enable_device(&device);
	if (status != 0)
	{
		fprintf(stderr, "Failed to enable device %04x:%02x:%02x.%x\n",
				device.domain, device.bus, device.device, device.function);
		exit(1);
	}

	// Enable device DMA
	status = pci_set_bus_master(&device);
	if (status != 0)
	{
		fprintf(stderr, "Failed to access device config space %04x:%02x:%02x.%x\n",
				device.domain, device.bus, device.device, device.function);
		exit(2);
	}

	status = pci_open_bar(&device, 0);
	if (status < 0)
	{
		fprintf(stderr, "Failed to access device BAR memory\n");
		exit(3);
	}

	ctrl_registers = mmap(NULL, NVM_CTRL_MEM_MINSIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FILE, device.fd, 0);
	if (ctrl_registers == NULL || ctrl_registers == MAP_FAILED)
	{
		fprintf(stderr, "Failed to memory map BAR reasource file: %s\n", strerror(errno));
		close(device.fd);
		exit(3);
	}

	// Get controller reference
	status = nvm_raw_ctrl_init(&ctrl, ctrl_registers, NVM_CTRL_MEM_MINSIZE);
	if (status != 0)
	{
		munmap((void*) ctrl_registers, NVM_CTRL_MEM_MINSIZE);
		close(device.fd);
		if (device.vfio_cfd >= 0)
			close(device.vfio_cfd);
		fprintf(stderr, "Failed to get controller reference: %s\n", strerror(status));
		exit(4);
	}

	status = identify(ctrl, nvm_ns_id, device.vfio_cfd);

	nvm_ctrl_free(ctrl);
	munmap((void*) ctrl_registers, NVM_CTRL_MEM_MINSIZE);
	close(device.fd);
	if (device.vfio_cfd >= 0)
		close(device.vfio_cfd);

	fprintf(stderr, "Goodbye!\n");
	exit(status);
}


