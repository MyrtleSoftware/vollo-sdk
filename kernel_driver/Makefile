# SPDX-License-Identifier: GPL-2.0
# Copyright (c) 2019, Intel Corporation.

__cflags += -O2
__cflags += -Wformat=2 -Wmissing-prototypes -Wstrict-prototypes
__cflags += -MMD -MP -fstack-protector-strong -DFORTIFY_SOURCE=2 -DGENHDR_STRUCT

# Flags Prevent compiler from optimizing out security checks
# -fno-strict-overflow - Dont assume strict overflow does not occure
# -fno-delete-null-pointer - Dont delete NULL pointer checks
# -fwrapv - Signed integers wrapping may occure
__cflags += -fno-strict-overflow -fno-delete-null-pointer-checks -fwrapv

ifneq ($(KERNELRELEASE),)
# kbuild part of makefile
#

obj-m += ifc_uio.o

ifc_uio-y += ifc_pci_uio.o

else

DRIVER := ifc_uio
BUILD_KERNEL := $(shell uname -r)

KDIR?=/lib/modules/$(shell uname -r)/build

kernelbuild = ${MAKE} CFLAGS_MODULE="${__cflags}" -C "${KDIR}" M=$(shell pwd) ${2} ${1}

all:
	$(call kernelbuild, modules)

clean:
	$(call kernelbuild,clean)

install: all
	$(call kernelbuild,modules_install)
endif
