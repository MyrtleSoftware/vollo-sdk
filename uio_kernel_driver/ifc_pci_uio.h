// SPDX-License-Identifier: GPL-2.0
/*-
 * Copyright(c) 2019-20 Intel Corporation. All rights reserved.
 */
#ifndef _IFC_PCI_UIO_H_
#define _IFC_PCI_UIO_H_

#ifdef __INTEL__DEBUG_CHK

char *intel_dbg_message = "WARNING: this is a DEBUG driver";
#define __INTEL__USE_DBG_CHK() asm("" ::"m"(intel_dbg_message))

#else /* __INTEL__DEBUG_CHK */

char *intel_release_message = "Intel PRODUCTION driver";
#define __INTEL__USE_DBG_CHK() asm("" ::"m"(intel_release_message))

#endif /* __INTEL_DEBUG_CHK */
#endif
