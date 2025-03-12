#ifndef __SRC_IO_MMLOADER_H__
#define __SRC_IO_MMLOADER_H__

#include <petscmat.h>
#include "mmio.h"

PetscErrorCode MatCreateFromMTX(Mat *A, const char *filein, PetscBool aijonly);

#endif // __SRC_IO_MMLOADER_H__