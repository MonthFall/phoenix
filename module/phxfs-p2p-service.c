/*
 * Phoenix P2P Service Export
 *
 * When Phoenix has remapped a device's BAR via devm_memremap_pages
 * (MEMORY_DEVICE_PCI_P2PDMA), vendor-specific DMA map calls may fail
 * for that device. This service bypasses DMA mapping and returns the
 * physical addresses from the backend's get_pages directly as DMA bus
 * addresses (valid under IOMMU=pt).
 */

#include <linux/module.h>
#include <linux/slab.h>
#include <linux/compiler.h>

#include "phxfs-p2p-service.h"
#include "phxfs-backend.h"
#include "phxfs.h"        /* phxfs_err / phxfs_warn / phxfs_info */
#include "config-host.h"

struct phxfs_p2p_handle {
	struct phxfs_page_table       *pt;
	uint64_t                     *ioaddrs;    /* kmalloc'd */
	uint32_t                      n_addrs;
	uint64_t                      gpu_vaddr;
	int                           invalidated; /* set by free_callback */
};

/*
 * phxfs_p2p_free_cb - called by NVIDIA driver when it forcefully
 * reclaims the GPU pages (e.g. GPU context destroyed).
 *
 * We must NOT call put_pages here (the driver is already
 * reclaiming). We free the page_table struct via free_page_table and
 * mark the handle as invalidated so the caller's subsequent deregister
 * skips put_pages.
 *
 * We do NOT free ioaddrs or the handle itself -- the caller still
 * holds a borrowed pointer and will free via deregister.
 *
 * Module refcount: we call module_put here to balance the
 * try_module_get in phxfs_p2p_register. The caller's deregister
 * sees invalidated==1 and skips its own module_put.
 */
static void phxfs_p2p_free_cb(void *data)
{
	struct phxfs_p2p_handle *h = data;

	if (!h) {
		phxfs_warn("phxfs_p2p_free_cb: NULL data\n");
		return;
	}

	WRITE_ONCE(h->invalidated, 1);

	if (h->pt) {
		phxfs_p2p->free_page_table(h->pt);
		h->pt = NULL;
	}

	module_put(THIS_MODULE);

	phxfs_warn("phxfs_p2p_free_cb: GPU pages force-reclaimed for vaddr=0x%llx\n",
		   (unsigned long long)h->gpu_vaddr);
}

int phxfs_p2p_register(uint64_t gpu_vaddr,
                       uint64_t length,
                       struct phxfs_p2p_handle **out_handle,
                       const uint64_t **out_ioaddrs,
                       uint32_t *out_n_addrs)
{
	struct phxfs_p2p_handle *h;
	int err;

	if (!out_handle || !out_ioaddrs || !out_n_addrs)
		return -EINVAL;
	if (gpu_vaddr == 0 || length == 0)
		return -EINVAL;

	/* Prevent Phoenix from being unloaded while handles are active */
	if (!try_module_get(THIS_MODULE))
		return -ENODEV;

	h = kzalloc(sizeof(*h), GFP_KERNEL);
	if (!h) {
		module_put(THIS_MODULE);
		return -ENOMEM;
	}
	h->gpu_vaddr   = gpu_vaddr;
	h->invalidated = 0;

	/* Step 1: pin device pages */
	err = phxfs_p2p->get_pages(gpu_vaddr, length,
	                           &h->pt, phxfs_p2p_free_cb, h);
	if (err || !h->pt) {
		int ret = err ? err : -ENOMEM;
		phxfs_err("phxfs_p2p_register: get_pages failed, "
			  "vaddr=0x%llx len=%llu err=%d\n",
			  (unsigned long long)gpu_vaddr,
			  (unsigned long long)length, ret);
		kfree(h);
		module_put(THIS_MODULE);
		return ret;
	}

	/* Step 2: extract physical addresses as bus addresses.
	 * Under IOMMU=pt (deployment contract), phys == bus address.
	 * We deliberately skip vendor DMA map — it fails when Phoenix
	 * has remapped the BAR. */
	h->n_addrs = phxfs_p2p->get_n_pages(h->pt);
	h->ioaddrs = kmalloc_array(h->n_addrs, sizeof(uint64_t), GFP_KERNEL);
	if (!h->ioaddrs) {
		phxfs_p2p->put_pages(gpu_vaddr, h->pt);
		kfree(h);
		module_put(THIS_MODULE);
		return -ENOMEM;
	}
	err = phxfs_p2p->get_phys_addrs(h->pt, h->ioaddrs, h->n_addrs);
	if (err) {
		phxfs_err("phxfs_p2p_register: get_phys_addrs failed, err=%d\n", err);
		kfree(h->ioaddrs);
		phxfs_p2p->put_pages(gpu_vaddr, h->pt);
		kfree(h);
		module_put(THIS_MODULE);
		return err;
	}

	*out_handle  = h;
	*out_ioaddrs = h->ioaddrs;   /* borrowed */
	*out_n_addrs = h->n_addrs;

	phxfs_info("phxfs_p2p_register: success vaddr=0x%llx len=%llu n_addrs=%u\n",
		   (unsigned long long)gpu_vaddr,
		   (unsigned long long)length, h->n_addrs);
	return 0;
}
EXPORT_SYMBOL_GPL(phxfs_p2p_register);

void phxfs_p2p_deregister(struct phxfs_p2p_handle *handle)
{
	int need_module_put;
	uint64_t gpu_vaddr;
	int was_invalidated;

	if (!handle)
		return;

	/* Save fields needed for logging before freeing the handle */
	gpu_vaddr = handle->gpu_vaddr;

	/* If not invalidated by free_callback, do normal put_pages.
	 * If invalidated, the callback already freed the page_table
	 * and called module_put -- we only free heap here. */
	was_invalidated = READ_ONCE(handle->invalidated);
	need_module_put = !was_invalidated;

	if (need_module_put && handle->pt)
		phxfs_p2p->put_pages(gpu_vaddr, handle->pt);

	kfree(handle->ioaddrs);
	kfree(handle);

	if (need_module_put)
		module_put(THIS_MODULE);

	phxfs_info("phxfs_p2p_deregister: vaddr=0x%llx invalidated=%d\n",
		   (unsigned long long)gpu_vaddr, was_invalidated);
}
EXPORT_SYMBOL_GPL(phxfs_p2p_deregister);
