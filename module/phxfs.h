#ifndef __PHOENIX_H__
#define __PHOENIX_H__

#include <linux/types.h>
#include <linux/blk-mq.h>
#include <linux/nvme.h>
#include <linux/memremap.h>
#include <linux/genalloc.h>
#include <linux/cdev.h>
#include <linux/pci.h>
#include <linux/printk.h>

#define MAX_DEV_NUM 16
#define MAX_GPU_DEVS 64

/* Forward declaration — full definition in phxfs-backend.h */
struct phxfs_page_table;

/* Phoenix logging macros */
extern int phxfs_debug;

#define phxfs_info(fmt, ...)					\
	do {							\
		if (phxfs_debug)				\
			printk(KERN_INFO fmt, ##__VA_ARGS__);	\
	} while (0)

#define phxfs_warn(fmt, ...)					\
	printk(KERN_WARNING fmt, ##__VA_ARGS__)

#define phxfs_err(fmt, ...)					\
	printk(KERN_ERR fmt, ##__VA_ARGS__)

#define PHXFS_REMAP_UNIT_SIZE  ((u64)16 * 1024 * 1024)  /* 16 MiB per remap unit */
#define PHXFS_RESERVED_SIZE    ((u64)128 * 1024 * 1024)  /* 128 MiB reserved at head/tail */

struct phxfs_bar_segment {
	u64 phys_start;    /* physical start address of this segment */
	u64 size;          /* segment size (multiple of PHXFS_REMAP_UNIT_SIZE) */
	void *va;          /* virtual address from devm_memremap_pages */
	struct pci_p2pdma_pagemap *p2p_pgmap;
};

struct pci_p2pdma_pagemap {
    struct dev_pagemap pgmap;
    struct pci_dev provider;
    u64 bus_offset;
};

struct phxfs_dev {
    struct pci_dev *dev; /*pci device */
    int domain;
    unsigned int bus;
    unsigned int devfn;
    u64 size; /* HBM pci bar 4 size */
    u64 paddr; /* HBM bus address space addr */
    struct resource pgmap_res;
    struct device device; /* char device. */
    struct cdev cdev;
    int idx;
    struct pci_p2pdma_pagemap *p2p_pgmap; /* legacy single-segment pgmap (kept for compat) */
    void __iomem *pci_mem_va; /* legacy single-segment VA (kept for compat) */
    bool remap;
    struct phxfs_bar_segment *segments; /* dynamically allocated segment array */
    int num_segments;    /* number of successfully mapped segments */
};

struct phxfs_ctrl {
    struct phxfs_dev phx_dev[MAX_DEV_NUM];
    int dev_num;
};

/* P2P mapping descriptor (vendor-agnostic) */
struct p2p_vmap;
typedef void (*release_fn)(struct p2p_vmap*);

struct gpu_region {
    struct phxfs_page_table *pt;
};

struct p2p_vmap {
    u64          gpuvaddr;
    u64          gpupaddr;
    u64          size;
    u64          cpuvaddr;
    release_fn   release;
    struct page **pages;
    unsigned long page_size;
    void        *data;           /* points to struct gpu_region */
    unsigned long n_addrs;
    uint64_t     addrs[1];
};

struct phxfs_dev_info_s {
    u64 dev_id;
} __attribute__((packed, aligned(8)));

struct phxfs_ioctl_map_s {
    struct phxfs_dev_info_s dev;
    u64 c_vaddr;
    u64 c_size;
    u64 n_vaddr;
    u64 n_size;
    u64 end_addr;
    u32 sbuf_block;
} __attribute__((packed, aligned(8)));
typedef struct phxfs_ioctl_map_s phxfs_ioctl_map_t;

struct phxfs_ioctl_io_s {
    u64 cpuvaddr; /* cpu vaddr */
    loff_t offset; /* file offset */
    u64 size; /* Read/Write length */
    u64 end_fence_value; /* End fence value for DMA completion */
    s64 ioctl_return;
    int fd; /* File descriptor */
} __attribute__((packed, aligned(8)));
typedef struct phxfs_ioctl_io_s phxfs_ioctl_io_t;

struct phxfs_ioctl_ret_s {
    s64 ret;
    u8 padding[40];
} __attribute__((packed, aligned(8)));
typedef struct phxfs_ioctl_ret_s phxfs_ioctl_ret_t;

union phxfs_ioctl_para_s {
    struct phxfs_ioctl_map_s map_param;
    struct phxfs_ioctl_io_s io_para;
    struct phxfs_ioctl_ret_s ret;
} __attribute__((packed, aligned(8)));
typedef union phxfs_ioctl_para_s phxfs_ioctl_para_t;


#define PHXFS_IOCTL 0x88 /* 0x4c */
#define PHXFS_IOCTL_MAP _IOW(PHXFS_IOCTL, 1, struct phxfs_ioctl_map_s)
#define PHXFS_IOCTL_UNMAP _IOW(PHXFS_IOCTL, 2, struct phxfs_ioctl_map_s)

void phxfs_map_dev_release(phxfs_ioctl_map_t *map_param, u64 devaddr, u64 dev_len, u64 cpuvaddr, u64 length);

#endif