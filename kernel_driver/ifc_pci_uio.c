// SPDX-License-Identifier: GPL-2.0
/*-
 * Copyright(c) 2019-20 Intel Corporation. All rights reserved.
 */

#include <linux/device.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/uio_driver.h>
#include <linux/io.h>
#include <linux/irq.h>
#include <linux/msi.h>
#include <linux/version.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/eventfd.h>
#include <linux/rcupdate.h>
#include "./ifc_pci_uio.h"

#ifndef PCI_VENDOR_ID_REDHAT_QUMRANET
#define PCI_VENDOR_ID_REDHAT_QUMRANET 0x1af4
#endif

#define MSIX_CAPACITY 2048
#define MSIX_INTR_CTX_BAR 2
#define MSIX_INTR_CTX_ADDR 0x0000
#define MSIX_CH_NO_MASK 0xFFF00000
#define MSIX_IRQFD_MASK 0xFFFFF
#define MSIX_DISABLE_INTR 0xFFFFE
#define MSIX_IRQFD_BITS 20

#define DRV_VERSION "1.0.1.0"
#define DRV_SUMMARY "ifc_uio Intel(R) PCIe end point driver"
static const char ifc_uio_driver_version[] = DRV_VERSION;
static const char ifc_uio_driver_string[] = DRV_SUMMARY;
static const char ifc_uio_copyright[] =
	"Copyright (c) 2019-20, Intel Corporation.";

MODULE_AUTHOR("Intel Corporation, <linux.nics@intel.com>");
MODULE_DESCRIPTION(DRV_SUMMARY);
MODULE_LICENSE("GPL");
MODULE_VERSION(DRV_VERSION);

struct myrtle_dma_buf {
	void *virt;
	dma_addr_t dma_handle;
	size_t size;
	struct list_head list;
};

/**
 * A structure describing the private information for a uio device.
 */
struct ifc_uio_pci_dev {
	struct uio_info info;
	struct pci_dev *pdev;
	atomic_t refcnt;
	// a linked list of allocated dma buffers
	struct list_head buf_list;
};

static int ifc_uio_pci_open(struct uio_info *info, struct inode *inode)
{
	struct ifc_uio_pci_dev *udev = info->priv;

	if (udev == NULL)
		return -1;

	if (atomic_inc_return(&udev->refcnt) != 1) {
		atomic_dec(&udev->refcnt);
		return -EBUSY;
	}

	return 0;
}

static void chr_vma_close(struct vm_area_struct *vma)
{
	// dma buffers are freed when the file is closed
}

const struct vm_operations_struct myrtle_fpga_vm_ops = {
	.close = chr_vma_close,
};

/* A somewhat unorthodox way to allocate a DMA buffer where the user mmaps the uio file.
 * The iova is passed to user by writing a magic 64 bit followed by the iova to start of the mmaped region.
 * The user can mmap the file multiple times to get multiple different buffers.
 *
 * This way we can reuse the uio mmap functionality and not have to implement our own char device.
 */
static int ifc_uio_pci_mmap(struct uio_info *info, struct vm_area_struct *vma)
{
	struct ifc_uio_pci_dev *udev = info->priv;
	u64 *tmp;
	struct myrtle_dma_buf *buf;
	unsigned long len;
	int ret;
	pgoff_t pgoff;

	if (vma->vm_start > vma->vm_end) {
		pr_err("Invalid mmap region\n");
		return -EINVAL;
	}
	len = vma->vm_end - vma->vm_start;
	pgoff = vma->vm_pgoff;

	if (pgoff != 0) {
		pr_err("Only support mmaping the first page\n");
		return -EINVAL;
	}

	buf = kmalloc(sizeof(*buf), GFP_KERNEL);
	if (!buf)
		return -ENOMEM;

	buf->virt = dma_alloc_coherent(&udev->pdev->dev, (size_t)len,
				       &buf->dma_handle, GFP_KERNEL);
	if (buf->virt == NULL) {
		return -ENOMEM;
	}
	buf->size = (size_t)len;
	list_add(&buf->list, &udev->buf_list);

	tmp = (u64 *)(buf->virt);
	// ascii encoded 'myrtldma' for the magic number
	tmp[0] = 0x6d7972746c646d61;
	tmp[1] = buf->dma_handle;

	vma->vm_ops = &myrtle_fpga_vm_ops;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 3, 0)
	vm_flags_set(vma, VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP |
				  VM_DONTCOPY | VM_NORESERVE);
#else
	vma->vm_flags |= VM_PFNMAP | VM_DONTCOPY | VM_DONTEXPAND;
#endif

	vma->vm_private_data = 0;

	ret = dma_mmap_coherent(&udev->pdev->dev, vma, buf->virt,
				buf->dma_handle, len);

	if (ret < 0) {
		pr_err("failed to remap kernel buffer to user-space: %d", ret);
		goto free_buf;
	}

	return 0;

free_buf:
	dma_free_coherent(&udev->pdev->dev, buf->size, buf->virt,
			  buf->dma_handle);
	list_del(&buf->list);
	kfree(buf);
	return ret;
}

/* When releasing, free all the dma buffers that were allocated */
static int ifc_uio_pci_release(struct uio_info *info, struct inode *inode)
{
	struct ifc_uio_pci_dev *udev = info->priv;
	struct myrtle_dma_buf *buf, *tmp;
	if (udev == NULL)
		return -1;

	list_for_each_entry_safe (buf, tmp, &udev->buf_list, list) {
		dma_free_coherent(&udev->pdev->dev, buf->size, buf->virt,
				  buf->dma_handle);
		list_del(&buf->list);
		kfree(buf);
	}

	atomic_dec_and_test(&udev->refcnt);
	return 0;
}

static int ifcuio_pci_sriov_configure(struct pci_dev *dev, int vfs)
{
#if IS_ENABLED(CONFIG_PCI_IOV)
	int rc = 0;

	if (dev == NULL)
		return -EFAULT;

	if (!pci_sriov_get_totalvfs(dev))
		return -EINVAL;

	if (!vfs)
		pci_disable_sriov(dev);
	else if (!pci_num_vf(dev))
		rc = pci_enable_sriov(dev, vfs);
	else /* do nothing if change vfs number */
		rc = -EINVAL;
	return rc;
#else
	(void)dev;
	(void)vfs;
	return 0;
#endif
}

/* sriov sysfs */
static ssize_t max_vfs_show(struct device *dev, struct device_attribute *attr,
			    char *buf)
{
	if (buf == NULL)
		return 0;

	return snprintf(buf, 10, "%u\n", dev_num_vf(dev));
}

static ssize_t max_vfs_store(struct device *dev, struct device_attribute *attr,
			     const char *buf, size_t count)
{
	struct pci_dev *pdev = to_pci_dev(dev);
	unsigned long max_vfs;
	int err;

	if (buf == NULL)
		return -EINVAL;

	if (!sscanf(buf, "%lu", &max_vfs))
		return -EINVAL;

	err = ifcuio_pci_sriov_configure(pdev, max_vfs);
	return err ? err : count;
}

static DEVICE_ATTR(max_vfs, S_IRUGO | S_IWUSR, max_vfs_show, max_vfs_store);
static struct attribute *dev_attrs[] = {
	&dev_attr_max_vfs.attr,
	NULL,
};

static const struct attribute_group dev_attr_grp = {
	.attrs = dev_attrs,
};

/**
 * ifc_io_error_detected - called when PCI error is detected
 * @dev: Pointer to PCI device
 * @state: The current pci connection state
 *
 * This function is called after a PCI bus error affecting
 * this device has been detected.
 */

static pci_ers_result_t ifc_io_error_detected(struct pci_dev *dev,
					      pci_channel_state_t state)
{
	pr_err("PCI error occured: channel state: %u\n", state);
	switch (state) {
	case pci_channel_io_normal:
		/* Non Correctable Non-Fatal errors */
		return PCI_ERS_RESULT_CAN_RECOVER;
	case pci_channel_io_perm_failure:
		return PCI_ERS_RESULT_DISCONNECT;
	case pci_channel_io_frozen:
		pci_disable_device(dev);
		return PCI_ERS_RESULT_NEED_RESET;
	default:
		pr_err("default error\n");
	}

	return PCI_ERS_RESULT_NEED_RESET;
}

/**
 * ifc_io_slot_reset - called after the pci bus has been reset.
 * @dev: Pointer to PCI device
 *
 * Restart the card from scratch, as if from a cold-boot. Implementation
 * may resemble the first half of the ifc_resume routine.
 */
static pci_ers_result_t ifc_io_slot_reset(struct pci_dev *dev)
{
	int err;

	err = pci_enable_device_mem(dev);
	if (err) {
		pr_err("Cannot re-enable PCI device after reset.\n");
		return PCI_ERS_RESULT_DISCONNECT;
	}

	pci_set_master(dev);

	pci_restore_state(dev);

	return PCI_ERS_RESULT_RECOVERED;
}

/**
 * ifc_io_resume - called when traffic can start flowing again.
 * @dev: Pointer to PCI device
 *
 * This callback is called when the error recovery drivers tells us that
 * its OK to resume normal operation. Implementation resembles the
 * second-half of the ifc_resume function.
 */
static void ifc_io_resume(struct pci_dev *dev)
{
	if (dev == NULL)
		return;

	pci_set_master(dev);
	pci_restore_state(dev);
}

/* Remap pci resources described by bar #pci_bar in uio resource n. */
static int ifcuio_pci_setup_iomem(struct pci_dev *dev, struct uio_info *info,
				  int n, int pci_bar, const char *name)
{
	unsigned long addr, len;
	void *internal_addr;

	if (info == NULL) {
		dev_err(&dev->dev, "%s info is NULL\n", name);
		return -EINVAL;
	}

	if (n >= ARRAY_SIZE(info->mem)) {
		dev_err(&dev->dev, "%s n=%d >= ARRAY_SIZE(info->mem)=%ld\n",
			name, n, ARRAY_SIZE(info->mem));
		return -EINVAL;
	}

	addr = pci_resource_start(dev, pci_bar);
	len = pci_resource_len(dev, pci_bar);
	if (addr == 0 || len == 0) {
		dev_err(&dev->dev,
			"pci_resource_start/len failed addr=%lx len=%lx\n",
			addr, len);
		return -1;
	}
	internal_addr = ioremap(addr, len);
	if (!internal_addr) {
		dev_err(&dev->dev, "ioremap failed of internal_addr failed\n");
		return -1;
	}
	info->mem[n].name = name;
	info->mem[n].addr = addr;
	info->mem[n].internal_addr = internal_addr;
	info->mem[n].size = len;
	info->mem[n].memtype = UIO_MEM_PHYS;
	return 0;
}

/* Get pci port io resources described by bar #pci_bar in uio resource n. */
static int ifcuio_pci_setup_ioport(struct pci_dev *dev, struct uio_info *info,
				   int n, int pci_bar, const char *name)
{
	unsigned long addr, len;

	if (info == NULL)
		return -EINVAL;

	if (n >= ARRAY_SIZE(info->port))
		return -EINVAL;

	addr = pci_resource_start(dev, pci_bar);
	len = pci_resource_len(dev, pci_bar);
	if (addr == 0 || len == 0)
		return -EINVAL;

	info->port[n].name = name;
	info->port[n].start = addr;
	info->port[n].size = len;
	info->port[n].porttype = UIO_PORT_X86;

	return 0;
}

/* Unmap previously ioremap'd resources */
static void ifcuio_pci_release_iomem(struct uio_info *info)
{
	int i;

	for (i = 0; i < MAX_UIO_MAPS; i++) {
		if (info->mem[i].internal_addr)
			iounmap(info->mem[i].internal_addr);
	}
	return;
}

static int ifcuio_setup_bars(struct pci_dev *dev, struct uio_info *info)
{
	int i, iom, iop, ret;
	unsigned long flags;
	static const char *bar_names[PCI_STD_RESOURCE_END + 1] = {
		"BAR0", "BAR1", "BAR2", "BAR3", "BAR4", "BAR5",
	};

	iom = 0;
	iop = 0;

	for (i = 0; i < ARRAY_SIZE(bar_names); i++) {
		if (i == 4) {
			continue;
		}
		if (pci_resource_len(dev, i) != 0 &&
		    pci_resource_start(dev, i) != 0) {
			flags = pci_resource_flags(dev, i);
			if (flags & IORESOURCE_MEM) {
				ret = ifcuio_pci_setup_iomem(dev, info, iom, i,
							     bar_names[i]);
				if (ret != 0) {
					dev_err(&dev->dev,
						"Failed to setup iomem for %s\n",
						bar_names[i]);
					return ret;
				}
				iom++;
			} else if (flags & IORESOURCE_IO) {
				ret = ifcuio_pci_setup_ioport(dev, info, iop, i,
							      bar_names[i]);
				if (ret != 0) {
					dev_err(&dev->dev,
						"Failed to setup ioport for %s\n",
						bar_names[i]);
					return ret;
				}
				iop++;
			}
		}
	}

	return (iom != 0 || iop != 0) ? ret : -ENOENT;
}

static int ifcuio_pci_probe(struct pci_dev *dev, const struct pci_device_id *id)
{
	struct ifc_uio_pci_dev *udev;
	int err;

#ifdef HAVE_PCI_IS_BRIDGE_API
	if (pci_is_bridge(dev)) {
		dev_warn(&dev->dev, "Ignoring PCI bridge device\n");
		return -ENODEV;
	}
#endif

	udev = kzalloc(sizeof(*udev), GFP_KERNEL);
	if (!udev)
		return -ENOMEM;

	/*
	 * enable device: ask low-level code to enable I/O and
	 * memory
	 */
	err = pci_enable_device(dev);
	if (err != 0) {
		dev_err(&dev->dev, "Cannot enable PCI device\n");
		goto fail_free;
	}

	/* enable bus mastering on the device */
	pci_set_master(dev);

	/* remap IO memory */
	err = ifcuio_setup_bars(dev, &udev->info);
	if (err != 0) {
		dev_err(&dev->dev, "Cannot setup bars\n");
		goto fail_release_iomem;
	}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 1, 0)
	err = dma_set_mask_and_coherent(&dev->dev, DMA_BIT_MASK(64));
	if (err != 0) {
		dev_err(&dev->dev, "Cannot set DMA mask\n");
		goto fail_release_iomem;
	}
#else
	/* set 64-bit DMA mask */
	err = pci_set_dma_mask(dev, DMA_BIT_MASK(64));
	if (err != 0) {
		dev_err(&dev->dev, "Cannot set DMA mask\n");
		goto fail_release_iomem;
	}

	err = pci_set_consistent_dma_mask(dev, DMA_BIT_MASK(64));
	if (err != 0) {
		dev_err(&dev->dev, "Cannot set consistent DMA mask\n");
		goto fail_release_iomem;
	}
#endif

	INIT_LIST_HEAD(&udev->buf_list);

	/* fill uio infos */
	udev->info.name = "ifc_uio";
	udev->info.version = "0.1";
	udev->info.priv = udev;
	udev->pdev = dev;
	udev->info.irq = -1;

	/* Setup the interrupt handler to disable the interrupts
	 * user space driver responsible to poll for interrupts
	 * and acknowledge
	 */
	udev->info.open = ifc_uio_pci_open;
	udev->info.mmap = ifc_uio_pci_mmap;
	udev->info.release = ifc_uio_pci_release;

	/* create /sysfs entry */
	if (pci_sriov_get_totalvfs(dev)) {
		err = sysfs_create_group(&dev->dev.kobj, &dev_attr_grp);
		if (err != 0) {
			dev_err(&dev->dev, "Cannot create sysfs group\n");
			goto fail_release_iomem;
		}
	}

	/* register uio driver */
	err = uio_register_device(&dev->dev, &udev->info);
	if (err != 0) {
		dev_err(&dev->dev, "Cannot create sysfs group\n");
		goto fail_remove_group;
	}

	pci_set_drvdata(dev, udev);

	return 0;

fail_remove_group:
	sysfs_remove_group(&dev->dev.kobj, &dev_attr_grp);
fail_release_iomem:
	ifcuio_pci_release_iomem(&udev->info);
	pci_disable_device(dev);
fail_free:
	kfree(udev);

	return err;
}

static void ifcuio_pci_remove(struct pci_dev *dev)
{
	struct ifc_uio_pci_dev *udev = pci_get_drvdata(dev);

	sysfs_remove_group(&dev->dev.kobj, &dev_attr_grp);
	uio_unregister_device(&udev->info);
	ifcuio_pci_release_iomem(&udev->info);
	if (!pci_num_vf(dev))
		pci_disable_device(dev);
	pci_set_drvdata(dev, NULL);
	kfree(udev);
}

const struct pci_device_id ifc_pci_tbl[] = { { PCI_DEVICE(0x1ed9, 0x766f) },
					     { PCI_DEVICE(0x12ba, 0x0069) },
					     {
						     0,
					     } };

static const struct pci_error_handlers ifc_err_handler = {
	.error_detected = ifc_io_error_detected,
	.slot_reset = ifc_io_slot_reset,
	.resume = ifc_io_resume
};

static struct pci_driver ifcuio_pci_driver = {
	.name = "ifc_uio",
	.id_table = ifc_pci_tbl,
	.probe = ifcuio_pci_probe,
	.remove = ifcuio_pci_remove,
	.sriov_configure = ifcuio_pci_sriov_configure,
	.err_handler = &ifc_err_handler
};

static int __init ifcuio_pci_init_module(void)
{
	pr_info("%s - version %s\n", ifc_uio_driver_string,
		ifc_uio_driver_version);
	pr_info("%s\n", ifc_uio_copyright);
	__INTEL__USE_DBG_CHK();

	return pci_register_driver(&ifcuio_pci_driver);
}

static void __exit ifcuio_pci_exit_module(void)
{
	pci_unregister_driver(&ifcuio_pci_driver);
}

module_init(ifcuio_pci_init_module);
module_exit(ifcuio_pci_exit_module);
