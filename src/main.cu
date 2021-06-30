/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Mauro Bisson <maurob@nvidia.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getopt.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "cudamacro.h" /* for time() */
#include "utils.h"

#define DIV_UP(a,b)     (((a)+((b)-1))/(b))

#define THREADS  128

#define BIT_X_SPIN (4)

#define CRIT_TEMP	(2.26918531421f)
#define	ALPHA_DEF	(0.1f)
#define MIN_TEMP	(0.05f*CRIT_TEMP)

#define MIN(a,b)	(((a)<(b))?(a):(b))
#define MAX(a,b)	(((a)>(b))?(a):(b))

// 2048+: 16, 16, 2, 1
//  1024: 16, 16, 1, 2
//   512:  8,  8, 1, 1
//   256:  4,  8, 1, 1
//   128:  2,  8, 1, 1

#define BLOCK_X (16)
#define BLOCK_Y (16)
#define BMULT_X (2)
#define BMULT_Y (1)

#define MAX_GPU	(256)

#define TOTAL_UPDATES_DEFAULT (10000)
#define SEED_DEFAULT  (463463564571ull)

#define MAX_CORR_LEN (128)

__device__ __forceinline__ unsigned int __mypopc(const unsigned int x) {
	return __popc(x);
}

__device__ __forceinline__ unsigned long long int __mypopc(const unsigned long long int x) {
	return __popcll(x);
}

enum {C_BLACK, C_WHITE};

__device__ __forceinline__ uint2 __mymake_int2(const unsigned int x,
		                               const unsigned int y) {
	return make_uint2(x, y);
}

__device__ __forceinline__ ulonglong2 __mymake_int2(const unsigned long long x,
		                                    const unsigned long long y) {
	return make_ulonglong2(x, y);
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X,
	 int LOOP_Y,
	 int BITXSP,
	 int COLOR,
	 typename INT_T,
	 typename INT2_T>
__global__  void latticeInit_k(const int devid,
			       const long long seed,
                               const int it,
                               const long long begY,
                               const long long dimX, // ld
                                     INT2_T *__restrict__ vDst) {

	const int __i = blockIdx.y * BDIM_Y * LOOP_Y + threadIdx.y;
	const int __j = blockIdx.x * BDIM_X * LOOP_X + threadIdx.x;

	const int SPIN_X_WORD = 8 * sizeof(INT_T) / BITXSP;

	const long long tid = ((devid * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * BDIM_X * BDIM_Y +
	                       threadIdx.y * BDIM_X + threadIdx.x;

	curandStatePhilox4_32_10_t st;
	curand_init(seed, tid, static_cast<long long>(2 * SPIN_X_WORD) * LOOP_X * LOOP_Y * (2 * it + COLOR), &st);

	INT2_T __tmp[LOOP_Y][LOOP_X];
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__tmp[i][j] = __mymake_int2(INT_T(0),INT_T(0));
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			#pragma unroll
			for(int k = 0; k < 8 * sizeof(INT_T); k += BITXSP) {
				if (curand_uniform(&st) < 0.5f) {
					__tmp[i][j].x |= INT_T(1) << k;
				}
				if (curand_uniform(&st) < 0.5f) {
					__tmp[i][j].y |= INT_T(1) << k;
				}
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			vDst[(begY + __i + i * BDIM_Y) * dimX + __j + j * BDIM_X] = __tmp[i][j];
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X,
	 int LOOP_Y,
	 int BITXSP,
	 typename INT_T,
	 typename INT2_T>
__global__  void hamiltInitB_k(const int devid,
			       const float tgtProb,
			       const long long seed,
                               const long long begY,
                               const long long dimX, // ld
                                     INT2_T *__restrict__ hamB) {

	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + threadIdx.y;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + threadIdx.x;

	const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
	                       threadIdx.y*BDIM_X + threadIdx.x;

	curandStatePhilox4_32_10_t st;
	curand_init(seed, tid, 0, &st);

	INT2_T __tmp[LOOP_Y][LOOP_X];
	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__tmp[i][j] = __mymake_int2(INT_T(0),INT_T(0));
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			#pragma unroll
			for(int k = 0; k < 8*sizeof(INT_T); k += BITXSP) {
				#pragma unroll
				for(int l = 0; l < BITXSP; l++) {
					if (curand_uniform(&st) < tgtProb) {
						__tmp[i][j].x |= INT_T(1) << (k+l);
					}
					if (curand_uniform(&st) < tgtProb) {
						__tmp[i][j].y |= INT_T(1) << (k+l);
					}
				}
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			hamB[(begY + __i+i*BDIM_Y)*dimX + __j+j*BDIM_X] = __tmp[i][j];
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X,
	 int LOOP_Y,
	 int BITXSP,
	 typename INT_T,
	 typename INT2_T>
__global__ void hamiltInitW_k(const int xsl,
			      const int ysl,
			      const long long begY,
		              const long long dimX,
		              const INT2_T *__restrict__ hamB,
		                    INT2_T *__restrict__ hamW) {

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + tidy;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + tidx;

	INT2_T __me[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__me[i][j] = hamB[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
		}
	}

	INT2_T __up[LOOP_Y][LOOP_X];
	INT2_T __ct[LOOP_Y][LOOP_X];
	INT2_T __dw[LOOP_Y][LOOP_X];
	INT2_T __sd[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__up[i][j].x = (__me[i][j].x & 0x8888888888888888ull) >> 1;
			__up[i][j].y = (__me[i][j].y & 0x8888888888888888ull) >> 1;

			__dw[i][j].x = (__me[i][j].x & 0x4444444444444444ull) << 1;
			__dw[i][j].y = (__me[i][j].y & 0x4444444444444444ull) << 1;
		}
	}

	const int readBack = !(__i%2); // this kernel reads only BLACK Js

	const int BITXWORD = 8*sizeof(INT_T);

	if (!readBack) {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {

				__ct[i][j].x = (__me[i][j].x & 0x2222222222222222ull) >> 1;
				__ct[i][j].y = (__me[i][j].y & 0x2222222222222222ull) >> 1;

				__ct[i][j].x |= (__me[i][j].x & 0x1111111111111111ull) << (BITXSP+1);
				__ct[i][j].y |= (__me[i][j].x & 0x1111111111111111ull) >> (BITXWORD-BITXSP - 1);
				__ct[i][j].y |= (__me[i][j].y & 0x1111111111111111ull) << (BITXSP+1);

				__sd[i][j].x = (__me[i][j].y & 0x1111111111111111ull) >> (BITXWORD-BITXSP - 1);
				__sd[i][j].y = 0;
			}
		}
	} else {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {

				__ct[i][j].x = (__me[i][j].x & 0x1111111111111111ull) << 1;
				__ct[i][j].y = (__me[i][j].y & 0x1111111111111111ull) << 1;

				__ct[i][j].y |= (__me[i][j].y & 0x2222222222222222ull) >> (BITXSP+1);
				__ct[i][j].x |= (__me[i][j].y & 0x2222222222222222ull) << (BITXWORD-BITXSP - 1);
				__ct[i][j].x |= (__me[i][j].x & 0x2222222222222222ull) >> (BITXSP+1);

				__sd[i][j].y = (__me[i][j].x & 0x2222222222222222ull) << (BITXWORD-BITXSP - 1);
				__sd[i][j].x = 0;
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {

		const int yoff = begY+__i + i*BDIM_Y;

		const int upOff = ( yoff   %ysl) == 0 ? yoff+ysl-1 : yoff-1;
		const int dwOff = ((yoff+1)%ysl) == 0 ? yoff-ysl+1 : yoff+1;

		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {

			const int xoff = __j + j*BDIM_X;

			atomicOr(&hamW[yoff*dimX + xoff].x, __ct[i][j].x);
			atomicOr(&hamW[yoff*dimX + xoff].y, __ct[i][j].y);

			atomicOr(&hamW[upOff*dimX + xoff].x, __up[i][j].x);
			atomicOr(&hamW[upOff*dimX + xoff].y, __up[i][j].y);

			atomicOr(&hamW[dwOff*dimX + xoff].x, __dw[i][j].x);
			atomicOr(&hamW[dwOff*dimX + xoff].y, __dw[i][j].y);

			const int sideOff = readBack ? (  (xoff   %xsl) == 0 ? xoff+xsl-1 : xoff-1 ):
						       ( ((xoff+1)%xsl) == 0 ? xoff-xsl+1 : xoff+1);

			atomicOr(&hamW[yoff*dimX + sideOff].x, __sd[i][j].x);
			atomicOr(&hamW[yoff*dimX + sideOff].y, __sd[i][j].y);
		}
	}
	return;
}


template<int BDIM_X,
	 int BDIM_Y,
	 int TILE_X,
	 int TILE_Y,
	 int FRAME_X,
	 int FRAME_Y,
	 typename INT2_T>
__device__ void loadTile(const int slX,
			 const int slY,
			 const long long begY,
			 const long long dimX,
			 const INT2_T *__restrict__ v,
			       INT2_T tile[][TILE_X+2*FRAME_X]) {

	const int blkx = blockIdx.x;
	const int blky = blockIdx.y;

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	const int startX =        blkx*TILE_X;
	const int startY = begY + blky*TILE_Y;

	#pragma unroll
	for(int j = 0; j < TILE_Y; j += BDIM_Y) {
		int yoff = startY + j+tidy;

		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = startX + i+tidx;
			tile[FRAME_Y + j+tidy][FRAME_X + i+tidx] = v[yoff*dimX + xoff];
		}
	}
	if (tidy == 0) {
		int yoff = (startY % slY) == 0 ? startY+slY-1 : startY-1;

		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = startX + i+tidx;
			tile[0][FRAME_X + i+tidx] = v[yoff*dimX + xoff];
		}

		yoff = ((startY+TILE_Y) % slY) == 0 ? startY+TILE_Y - slY : startY+TILE_Y;

		#pragma unroll
		for(int i = 0; i < TILE_X; i += BDIM_X) {
			const int xoff = startX + i+tidx;
			tile[FRAME_Y + TILE_Y][FRAME_X + i+tidx] = v[yoff*dimX + xoff];
		}

		// the other branch in slower so skip it if possible
		if (BDIM_X <= TILE_Y) {

			int xoff = (startX % slX) == 0 ? startX+slX-1 : startX-1;

			#pragma unroll
			for(int j = 0; j < TILE_Y; j += BDIM_X) {
				yoff = startY + j+tidx;
				tile[FRAME_Y + j+tidx][0] = v[yoff*dimX + xoff];
			}

			xoff = ((startX+TILE_X) % slX) == 0 ? startX+TILE_X - slX : startX+TILE_X;

			#pragma unroll
			for(int j = 0; j < TILE_Y; j += BDIM_X) {
				yoff = startY + j+tidx;
				tile[FRAME_Y + j+tidx][FRAME_X + TILE_X] = v[yoff*dimX + xoff];
			}
		} else {
			if (tidx < TILE_Y) {
				int xoff = (startX % slX) == 0 ? startX+slX-1 : startX-1;

				yoff = startY + tidx;
				tile[FRAME_Y + tidx][0] = v[yoff*dimX + xoff];;

				xoff = ((startX+TILE_X) % slX) == 0 ? startX+TILE_X - slX : startX+TILE_X;
				tile[FRAME_Y + tidx][FRAME_X + TILE_X] = v[yoff*dimX + xoff];
			}
		}
	}
	return;
}

template<int BDIM_X,
	 int BDIM_Y,
	 int LOOP_X,
	 int LOOP_Y,
	 int BITXSP,
	 int COLOR,
	 typename INT_T,
	 typename INT2_T>
__global__
void spinUpdateV_2D_k(const int devid,
		      const long long seed,
		      const int it,
		      const int slX, // sublattice size grid_width of one color (in words)
		      const int slY, // sublattice size grid_height of one color
		      const long long begY,
		      const long long dimX, // ld
		      const float vExp[][5],
		      const INT2_T *__restrict__ jDst,
		      const INT2_T *__restrict__ vSrc,
		            INT2_T *__restrict__ vDst) {

	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	__shared__ INT2_T shTile[BDIM_Y*LOOP_Y+2][BDIM_X*LOOP_X+2];

	loadTile<BDIM_X, BDIM_Y,
		 BDIM_X*LOOP_X,
		 BDIM_Y*LOOP_Y,
		 1, 1, INT2_T>(slX, slY, begY, dimX, vSrc, shTile);

	// __shExp[cur_s{0,1}][sum_s{0,1}] = __expf(-2*cur_s{-1,+1}*F{+1,-1}(sum_s{0,1})*INV_TEMP)
	__shared__ float __shExp[2][5];

	// for small lattices BDIM_X/grid_height may be smaller than 2/5
	#pragma unroll
	for(int i = 0; i < 2; i += BDIM_Y) {
		#pragma unroll
		for(int j = 0; j < 5; j += BDIM_X) {
			if (i+tidy < 2 && j+tidx < 5) {
				__shExp[i+tidy][j+tidx] = vExp[i+tidy][j+tidx];
			}
		}
	}
	__syncthreads();

	const int __i = blockIdx.y*BDIM_Y*LOOP_Y + tidy;
	const int __j = blockIdx.x*BDIM_X*LOOP_X + tidx;

	const long long tid = ((devid*gridDim.y + blockIdx.y)*gridDim.x + blockIdx.x)*BDIM_X*BDIM_Y +
	                       threadIdx.y*BDIM_X + threadIdx.x;

	INT2_T __me[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__me[i][j] = vDst[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
		}
	}

	INT2_T __up[LOOP_Y][LOOP_X];
	INT2_T __ct[LOOP_Y][LOOP_X];
	INT2_T __dw[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__up[i][j] = shTile[i*BDIM_Y +   tidy][j*BDIM_X + 1+tidx];
			__ct[i][j] = shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 1+tidx];
			__dw[i][j] = shTile[i*BDIM_Y + 2+tidy][j*BDIM_X + 1+tidx];
		}
	}

	// BDIM_Y is power of two so row parity won't change across loops
	const int readBack = (COLOR == C_BLACK) ? !(__i%2) : (__i%2);

	INT2_T __sd[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__sd[i][j] = (readBack) ? shTile[i*BDIM_Y + 1+tidy][j*BDIM_X +   tidx]:
						  shTile[i*BDIM_Y + 1+tidy][j*BDIM_X + 2+tidx];
		}
	}

	if (readBack) {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__sd[i][j].x = (__ct[i][j].x << BITXSP) | (__sd[i][j].y >> (8*sizeof(__sd[i][j].y)-BITXSP));
				__sd[i][j].y = (__ct[i][j].y << BITXSP) | (__ct[i][j].x >> (8*sizeof(__ct[i][j].x)-BITXSP));
			}
		}
	} else {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__sd[i][j].y = (__ct[i][j].y >> BITXSP) | (__sd[i][j].x << (8*sizeof(__sd[i][j].x)-BITXSP));
				__sd[i][j].x = (__ct[i][j].x >> BITXSP) | (__ct[i][j].y << (8*sizeof(__ct[i][j].y)-BITXSP));
			}
		}
	}

	if (jDst != NULL) {
		INT2_T __J[LOOP_Y][LOOP_X];

		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__J[i][j] = jDst[(begY+__i+i*BDIM_Y)*dimX + __j+j*BDIM_X];
			}
		}

		// apply them
		// the 4 bits of J codify: <upJ, downJ, leftJ, rightJ>
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {

				__up[i][j].x ^= (__J[i][j].x & 0x8888888888888888ull) >> 3;
				__up[i][j].y ^= (__J[i][j].y & 0x8888888888888888ull) >> 3;

				__dw[i][j].x ^= (__J[i][j].x & 0x4444444444444444ull) >> 2;
				__dw[i][j].y ^= (__J[i][j].y & 0x4444444444444444ull) >> 2;

				if (readBack) {
					// __sd[][] holds "left" spins
					// __ct[][] holds "right" spins
					__sd[i][j].x ^= (__J[i][j].x & 0x2222222222222222ull) >> 1;
					__sd[i][j].y ^= (__J[i][j].y & 0x2222222222222222ull) >> 1;

					__ct[i][j].x ^= (__J[i][j].x & 0x1111111111111111ull);
					__ct[i][j].y ^= (__J[i][j].y & 0x1111111111111111ull);
				} else {
					// __ct[][] holds "left" spins
					// __sd[][] holds "right" spins
					__ct[i][j].x ^= (__J[i][j].x & 0x2222222222222222ull) >> 1;
					__ct[i][j].y ^= (__J[i][j].y & 0x2222222222222222ull) >> 1;

					__sd[i][j].x ^= (__J[i][j].x & 0x1111111111111111ull);
					__sd[i][j].y ^= (__J[i][j].y & 0x1111111111111111ull);
				}
			}
		}
	}

	curandStatePhilox4_32_10_t st;
	curand_init(seed, tid, static_cast<long long>(2*SPIN_X_WORD)*LOOP_X*LOOP_Y*(2*it+COLOR), &st);

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__ct[i][j].x += __up[i][j].x;
			__dw[i][j].x += __sd[i][j].x;
			__ct[i][j].x += __dw[i][j].x;

			__ct[i][j].y += __up[i][j].y;
			__dw[i][j].y += __sd[i][j].y;
			__ct[i][j].y += __dw[i][j].y;
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			#pragma unroll
			for(int z = 0; z < 8*sizeof(INT_T); z += BITXSP) {

				const int2 __src = make_int2((__me[i][j].x >> z) & 0xF,
							     (__me[i][j].y >> z) & 0xF);

				const int2 __sum = make_int2((__ct[i][j].x >> z) & 0xF,
							     (__ct[i][j].y >> z) & 0xF);

				const INT_T ONE = static_cast<INT_T>(1);

				if (curand_uniform(&st) <= __shExp[__src.x][__sum.x]) {
					__me[i][j].x ^= ONE << z;
				}
				if (curand_uniform(&st) <= __shExp[__src.y][__sum.y]) {
					__me[i][j].y ^= ONE << z;
				}
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			vDst[(begY + __i+i*BDIM_Y)*dimX + __j+j*BDIM_X] = __me[i][j];
		}
	}
	return;
}

template<int BDIM_X,
	 int WSIZE,
	 typename T>
__device__ __forceinline__ T __block_sum(T v) {

	__shared__ T sh[BDIM_X/WSIZE];

	const int lid = threadIdx.x%WSIZE;
	const int wid = threadIdx.x/WSIZE;

	#pragma unroll
	for(int i = WSIZE/2; i; i >>= 1) {
		v += __shfl_down_sync(0xFFFFFFFF, v, i);
	}
	if (lid == 0) sh[wid] = v;

	__syncthreads();
	if (wid == 0) {
		v = (lid < (BDIM_X/WSIZE)) ? sh[lid] : 0;

		#pragma unroll
		for(int i = (BDIM_X/WSIZE)/2; i; i >>= 1) {
			v += __shfl_down_sync(0xFFFFFFFF, v, i);
		}
	}
	__syncthreads();
	return v;
}

// to be optimized
template<int BDIM_X,
	 int BITXSP,
         typename INT_T,
	 typename SUM_T>
__global__ void getMagn_k(const long long n,
			  const INT_T *__restrict__ v,
			        SUM_T *__restrict__ sum) {

	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	const long long nth = static_cast<long long>(blockDim.x)*gridDim.x;
	const long long tid = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

	SUM_T __cntP = 0;
	SUM_T __cntN = 0;

	for(long long i = 0; i < n; i += nth) {
		if (i+tid < n) {
			const int __c = __mypopc(v[i+tid]);
			__cntP += __c;
			__cntN += SPIN_X_WORD - __c;
		}
	}

	__cntP = __block_sum<BDIM_X, 32>(__cntP);
	__cntN = __block_sum<BDIM_X, 32>(__cntN);

	if (threadIdx.x == 0) {
		atomicAdd(sum+0, __cntP);
		atomicAdd(sum+1, __cntN);
	}
	return;
}

static void usage(const int SPIN_X_WORD, const char *pname) {

        const char *bname = rindex(pname, '/');
        if (!bname) {bname = pname;}
        else        {bname++;}

        fprintf(stdout,
                "Usage: %1$s [options]\n"
                "options:\n"
                "\t-x|--x <HORIZ_DIM>\n"
		"\t\tSpecifies the horizontal dimension of the entire  lattice  (black+white  spins),\n"
		"\t\tper GPU. This dimension must be a multiple of %2$d.\n"
                "\n"
                "\t-y|--y <VERT_DIM>\n"
		"\t\tSpecifies the vertical dimension of the entire lattice (black+white spins),  per\n"
		"\t\tGPU. This dimension must be a multiple of %3$d.\n"
                "\n"
                "\t-n|--n <NSTEPS>\n"
		"\t\tSpecifies the number of iteration to run.\n"
		"\t\tDefualt: %4$d\n"
                "\n"
                "\t-d|--devs <NUM_DEVICES>\n"
		"\t\tSpecifies the number of GPUs to use. Will use devices with ids [0, NUM_DEVS-1].\n"
		"\t\tDefualt: 1.\n"
                "\n"
                "\t-s|--seed <SEED>\n"
		"\t\tSpecifies the seed used to generate random numbers.\n"
		"\t\tDefault: %5$llu\n"
                "\n"
                "\t-a|--alpha <ALPHA>\n"
		"\t\tSpecifies the temperature in T_CRIT units.  If both this  option  and  '-t'  are\n"
		"\t\tspecified then the '-t' option is used.\n"
		"\t\tDefault: %6$f\n"
                "\n"
                "\t-t|--temp <TEMP>\n"
		"\t\tSpecifies the temperature in absolute units.  If both this option and  '-a'  are\n"
		"\t\tspecified then this option is used.\n"
		"\t\tDefault: %7$f\n"
                "\n"
                "\t-p|--print <STAT_FREQ>\n"
		"\t\tSpecifies the frequency, in no.  of  iteration,  with  which  the  magnetization\n"
		"\t\tstatistics is printed.  If this option is used together to the '-e' option, this\n"
		"\t\toption is ignored.\n"
		"\t\tDefault: only at the beginning and at end of the simulation\n"
                "\n"
                "\t-e|--exppr\n"
		"\t\tPrints the magnetization at time steps in the series 0 <= 2^(x/4) < NSTEPS.   If\n"
		"\t\tthis option is used  together  to  the  '-p'  option,  the  latter  is  ignored.\n"
		"\t\tDefault: disabled\n"
                "\n"
                "\t-c|--corr\n"
		"\t\tDumps to a  file  named  corr_{grid_width}x{grid_height}_T_{TEMP}  the  correlation  of each  point\n"
		"\t\twith the  %8$d points on the right and below.  The correlation is computed  every\n"
		"\t\ttime the magnetization is printed on screen (based on either the  '-p'  or  '-e'\n"
		"\t\toption) and it is written in the file one line per measure.\n"
		"\t\tDefault: disabled\n"
                "\n"
                "\t-m|--magn <TGT_MAGN>\n"
		"\t\tSpecifies the magnetization value at which the simulation is  interrupted.   The\n"
		"\t\tmagnetization of the system is checked against TGT_MAGN every STAT_FREQ, if  the\n"
		"\t\t'-p' option is specified, or according to the exponential  timestep  series,  if\n"
		"\t\tthe '-e' option is specified.  If neither '-p' not '-e' are specified then  this\n"
		"\t\toption is ignored.\n"
		"\t\tDefault: unset\n"
                "\n"
		"\t-J|--J <PROB>\n"
		"\t\tSpecifies the probability [0.0-1.0] that links  connecting  any  two  spins  are\n"
		"\t\tanti-ferromagnetic. \n"
		"\t\tDefault: 0.0\n"
                "\n"
		"\t   --xsl <HORIZ_SUB_DIM>\n"
		"\t\tSpecifies the horizontal dimension of each sub-lattice (black+white spins),  per\n"
		"\t\tGPU.  This dimension must be a divisor of the horizontal dimension of the entire\n"
		"\t\tlattice per  GPU  (specified  with  the  '-x'  option) and a multiple of %2$d.\n"
		"\t\tDefault: sub-lattices are disabled.\n"
                "\n"
		"\t   --ysl <VERT_SUB_DIM>\n"
		"\t\tSpecifies the vertical  dimension of each  sub-lattice (black+white spins),  per\n"
		"\t\tGPU.  This dimension must be a divisor of the vertical dimension of  the  entire\n"
		"\t\tlattice per  GPU  (specified  with  the  '-y'  option) and a multiple of %3$d.\n"
                "\n"
                "\t-o|--o\n"
		"\t\tEnables the file dump of  the lattice  every time  the magnetization is printed.\n"
		"\t\tDefault: off\n\n",
                bname,
		2*SPIN_X_WORD*2*BLOCK_X*BMULT_X,
		BLOCK_Y*BMULT_Y,
		TOTAL_UPDATES_DEFAULT,
		SEED_DEFAULT,
		ALPHA_DEF,
		ALPHA_DEF*CRIT_TEMP,
		MAX_CORR_LEN);
        exit(EXIT_SUCCESS);
}

static void countSpins(const int redBlocks,
								       const size_t total_length,
								       const size_t sublattice_length,
								       const unsigned long long *d_black_tiles,
								       const unsigned long long *d_white_tiles,
									     unsigned long long **sum_d,
									     unsigned long long *bsum,
									     unsigned long long *wsum)
{
	CHECK_CUDA(cudaMemset(sum_d[0], 0, 2*sizeof(**sum_d)));
	getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(total_length, d_black_tiles, sum_d[0]);
	CHECK_ERROR("getMagn_k");
	CHECK_CUDA(cudaDeviceSynchronize());

	bsum[0] = 0;
	wsum[0] = 0;

	unsigned long long sum_h[0][2];

	CHECK_CUDA(cudaMemcpy(sum_h[0], sum_d[0], 2*sizeof(**sum_h), cudaMemcpyDeviceToHost));
	bsum[0] += sum_h[0][0];
	wsum[0] += sum_h[0][1];

	return;
}


template<int BDIM_X,
	 int BITXSP,
	 int N_CORR,
   typename INT_T,
	 typename SUM_T>
__global__ void getCorr2D_k(const int corrLen,
												    const long long dimX,
												    const long long dimY,
												    const long long begY,
												    const INT_T *__restrict__ black,
												    const INT_T *__restrict__ white,
													  SUM_T *__restrict__ corr)
{
	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	const int tid = threadIdx.x;

	const long long startY = begY + blockIdx.x;

	const int SH_LEN = BDIM_X + DIV_UP(N_CORR/2, SPIN_X_WORD);

	__shared__ INT_T __shB[SH_LEN];
	__shared__ INT_T __shW[SH_LEN];
	__shared__ SUM_T __shC[N_CORR];

	#pragma unroll
	for(int j = 0; j < N_CORR; j += BDIM_X) {
		if (j+tid < N_CORR) {
			__shC[j+tid] = 0;
		}
	}

	const int chunkDimX = 2*BDIM_X*SPIN_X_WORD;

	for(long long l = 0; l < dimX; l += BDIM_X) {

		__syncthreads();
		#pragma unroll
		for(int j = 0; j < SH_LEN; j += BDIM_X) {
			if (j+tid < SH_LEN) {
				const int off = (l+j+tid < dimX) ? l+j+tid : l+j+tid - dimX;
				__shB[j+tid] = black[startY*dimX + off];
				__shW[j+tid] = white[startY*dimX + off];
			}
		}
		__syncthreads();

		for(int j = 1; j <= corrLen; j++) {

			SUM_T myCorr = 0;

			for(long long i = tid; i < chunkDimX; i += BDIM_X) {

				// horiz corr
				const long long myWrdX = (i/2) / SPIN_X_WORD;
				const long long myOffX = (i/2) % SPIN_X_WORD;

				INT_T __tmp = ((startY ^ i) & 1) ? __shW[myWrdX] : __shB[myWrdX];
				const int mySpin = (__tmp >> (myOffX*BITXSP)) & 0xF;

				const long long nextX = i+j;

				const long long nextWrdX = (nextX/2) / SPIN_X_WORD;
				const long long nextOffX = (nextX/2) % SPIN_X_WORD;

				__tmp = ((startY ^ nextX) & 1) ? __shW[nextWrdX] : __shB[nextWrdX];
				const int nextSpin = (__tmp >> (nextOffX*BITXSP)) & 0xF;

				myCorr += (mySpin == nextSpin) ? SUM_T(1) : SUM_T(-1);

				// vert corr
				const long long nextY = (startY+j >= dimY) ? startY+j-dimY : startY+j;

				__tmp = ((nextY ^ i) & 1) ? white[nextY*dimX + l+myWrdX]:
							    black[nextY*dimX + l+myWrdX];
				const int vertSpin = (__tmp >> (myOffX*BITXSP)) & 0xF;

				myCorr += (mySpin == vertSpin) ? SUM_T(1) : SUM_T(-1);
			}

			myCorr = __block_sum<BDIM_X, 32>(myCorr);
			if (!tid) {
				__shC[j-1] += myCorr;
			}
		}
	}
	__syncthreads();

	#pragma unroll
	for(int j = 0; j < N_CORR; j += BDIM_X) {
		if (j+tid < N_CORR) {
			atomicAdd(corr + j+tid, __shC[j+tid]);
		}
	}
	return;
}


template<int BDIM_X,
	 int BITXSP,
	 int N_CORR,
   typename INT_T,
	 typename SUM_T>
__global__ void getCorr2DRepl_k(const int corrLen,
																const long long dimX,
																const long long begY,
															  const long long slX, // sublattice size grid_width of one color (in words)
															  const long long slY, // sublattice size grid_height of one color
																const INT_T *__restrict__ black,
																const INT_T *__restrict__ white,
																      SUM_T *__restrict__ corr)
{
	const int tid = threadIdx.x;
	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSP;

	const long long startY = begY + blockIdx.x;
	const long long mySLY = startY / slY;

	const long long NSLX = 2ull*dimX*SPIN_X_WORD / slX;

	const int SH_LEN = BDIM_X + DIV_UP(N_CORR/2, SPIN_X_WORD);

	__shared__ INT_T __shB[SH_LEN];
	__shared__ INT_T __shW[SH_LEN];

	__shared__ SUM_T __shC[N_CORR];

	#pragma unroll
	for(int j = 0; j < N_CORR; j += BDIM_X) {
		if (j+tid < N_CORR) {
			__shC[j+tid] = 0;
		}
	}

	const int chunkDimX = MIN(2*BDIM_X*SPIN_X_WORD, slX);

	const int slXLD = (slX/2) / SPIN_X_WORD;

	for(long long sl = 0; sl < NSLX; sl++) {

		for(long long l = 0; l < slXLD; l += BDIM_X) {

			__syncthreads();
			#pragma unroll
			for(int j = 0; j < SH_LEN; j += BDIM_X) {
				if (j+tid < SH_LEN) {
					const int off = (l+j+tid) % slXLD;
					__shB[j+tid] = black[startY*dimX + sl*slXLD + off];
					__shW[j+tid] = white[startY*dimX + sl*slXLD + off];
				}
			}
			__syncthreads();

			for(int j = 1; j <= corrLen; j++) {

				SUM_T myCorr = 0;

				for(long long i = tid; i < chunkDimX; i += BDIM_X) {

					// horiz corr
					const long long myWrdX = (i/2) / SPIN_X_WORD;
					const long long myOffX = (i/2) % SPIN_X_WORD;

					INT_T __tmp = ((startY ^ i) & 1) ? __shW[myWrdX] : __shB[myWrdX];
					const int mySpin = (__tmp >> (myOffX*BITXSP)) & 0xF;

					const long long nextX = i+j;

					const long long nextWrdX = (nextX/2) / SPIN_X_WORD;
					const long long nextOffX = (nextX/2) % SPIN_X_WORD;

					__tmp = ((startY ^ nextX) & 1) ? __shW[nextWrdX] : __shB[nextWrdX];
					const int nextSpin = (__tmp >> (nextOffX*BITXSP)) & 0xF;

					myCorr += (mySpin == nextSpin) ? SUM_T(1) : SUM_T(-1);

					// vert corr
					const long long nextY = (startY+j >= (mySLY+1)*slY) ? startY+j-slY : startY+j;

					__tmp = ((nextY ^ i) & 1) ? white[nextY*dimX + sl*slXLD + l+myWrdX]:
								    black[nextY*dimX + sl*slXLD + l+myWrdX];
					const int vertSpin = (__tmp >> (myOffX*BITXSP)) & 0xF;

					myCorr += (mySpin == vertSpin) ? SUM_T(1) : SUM_T(-1);
				}

				myCorr = __block_sum<BDIM_X, 32>(myCorr);
				if (!tid) {
					__shC[j-1] += myCorr;
				}
			}
		}
	}
	__syncthreads();

	#pragma unroll
	for(int j = 0; j < N_CORR; j += BDIM_X) {
		if (j+tid < N_CORR) {
			atomicAdd(corr + j+tid, __shC[j+tid]);
		}
	}
	return;
}


int main(int argc, char **argv) {

	unsigned long long *d_spins=NULL;
	unsigned long long *d_black_tiles=NULL;
	unsigned long long *d_white_tiles=NULL;

	unsigned long long *hamB_d=NULL;
	unsigned long long *hamW_d=NULL;

	cudaEvent_t start, stop;
  float elapsed_time;

	const int SPIN_X_WORD = (8 * sizeof(*d_spins)) / BIT_X_SPIN;

	int grid_width = 2048;
	int grid_height = 2048;

	int total_updates = TOTAL_UPDATES_DEFAULT;

	unsigned long long seed = SEED_DEFAULT;

	float temp  = 0.666f;

	int XSL = 0;
	int YSL = 0;

	if (!grid_width || (grid_width % 2) || ((grid_width / 2) % (SPIN_X_WORD*2*BLOCK_X*BMULT_X))) {
		fprintf(stderr, "\nPlease specify an grid_width dim multiple of %d\n\n", 2 * SPIN_X_WORD * 2 * BLOCK_X * BMULT_X);
		usage(SPIN_X_WORD, argv[0]);
		exit(EXIT_FAILURE);
	}
	if (!grid_height || (grid_height % (BLOCK_Y * BMULT_Y))) {
		fprintf(stderr, "\nPlease specify a grid_height dim multiple of %d\n\n", BLOCK_Y * BMULT_Y);
		usage(SPIN_X_WORD, argv[0]);
		exit(EXIT_FAILURE);
	}

	XSL = grid_width;
	YSL = grid_height;

	cudaDeviceProp props;

	printf("\nUsing GPUs:\n");

	CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
	printf("\t%2d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
		0, props.name, props.multiProcessorCount,
		props.maxThreadsPerMultiProcessor,
		props.major, props.minor,
		props.ECCEnabled?"on":"off");

	printf("\n");

	size_t words_per_row = (grid_width / 2) / SPIN_X_WORD;
	// length of a single color section per GPU
	size_t sublattice_length = static_cast<size_t>(grid_height) * words_per_row;
	// total lattice length
	size_t total_length = 2ull * sublattice_length;

	dim3 grid(DIV_UP(words_per_row / 2, BLOCK_X * BMULT_X), DIV_UP(grid_height, BLOCK_Y * BMULT_Y));
	dim3 block(BLOCK_X, BLOCK_Y);

	printf("Run configuration:\n");
	printf("\tspin/word: %d\n", SPIN_X_WORD);
	printf("\tspins: %zu\n", total_length * SPIN_X_WORD);
	printf("\tseed: %llu\n", seed);
	printf("\titerations: %d\n", total_updates);
	printf("\tblock (x, y): %d, %d\n", block.x, block.y);
	printf("\ttile  (x, y): %d, %d\n", BLOCK_X * BMULT_X, BLOCK_Y * BMULT_Y);
	printf("\tgrid  (x, y): %d, %d\n", grid.x, grid.y);

	printf("\ttemp: %f (%f*T_crit)\n", temp, temp / CRIT_TEMP);

	printf("\n");

	printf("\tlattice size:      %8d x %8d\n", grid_height, grid_width);
	printf("\tlattice shape: 2 x %8d x %8zu (%12zu %s)\n", grid_height, words_per_row, total_length, sizeof(*d_spins) == 4 ? "uints" : "ulls");
	printf("\tmemory: %.2lf MB (%.2lf MB per GPU)\n", (total_length*sizeof(*d_spins))/(1024.0 * 1024.0), sublattice_length * 2 * sizeof(*d_spins) / (1024.0 * 1024.0));

	const int redBlocks = MIN(DIV_UP(total_length, THREADS),
				  (props.maxThreadsPerMultiProcessor/THREADS)*props.multiProcessorCount);

	unsigned long long cntPos;
	unsigned long long cntNeg;
	unsigned long long *sum_d[MAX_GPU];

	CHECK_CUDA(cudaMalloc(&d_spins, total_length*sizeof(*d_spins)));
	CHECK_CUDA(cudaMemset(d_spins, 0, total_length*sizeof(*d_spins)));

	CHECK_CUDA(cudaMalloc(&sum_d[0], 2*sizeof(**sum_d)));


	d_black_tiles = d_spins;
	d_white_tiles = d_spins + total_length/2;

	float *exp_d[MAX_GPU];
	float  exp_h[2][5];

	// precompute possible exponentials
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 5; j++) {
			if(temp > 0) {
				exp_h[i][j] = expf((i?-2.0f:2.0f)*static_cast<float>(j*2-4)*(1.0f/temp));
			} else {
				if(j == 2) {
					exp_h[i][j] = 0.5f;
				} else {
					exp_h[i][j] = (i?-2.0f:2.0f)*static_cast<float>(j*2-4);
				}
			}
		}
	}

	CHECK_CUDA(cudaSetDevice(0));
	CHECK_CUDA(cudaMalloc(exp_d, 2*5*sizeof(**exp_d)));
	CHECK_CUDA(cudaMemcpy(exp_d[0], exp_h, 2*5*sizeof(**exp_d), cudaMemcpyHostToDevice));


	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	CHECK_CUDA(cudaSetDevice(0));
	latticeInit_k<BLOCK_X, BLOCK_Y,
		      BMULT_X, BMULT_Y,
		      BIT_X_SPIN, C_BLACK,
		      unsigned long long><<<grid, block>>>(0,
							   seed,
							   0, 0, words_per_row/2,
							   reinterpret_cast<ulonglong2 *>(d_black_tiles));
	CHECK_ERROR("initLattice_k");

	latticeInit_k<BLOCK_X, BLOCK_Y,
		      BMULT_X, BMULT_Y,
		      BIT_X_SPIN, C_WHITE,
		      unsigned long long><<<grid, block>>>(0,
							   seed,
							   0, 0, words_per_row/2,
							   reinterpret_cast<ulonglong2 *>(d_white_tiles));
	CHECK_ERROR("initLattice_k");

	// computes sum over array
	countSpins(redBlocks, total_length, sublattice_length, d_black_tiles, d_white_tiles, sum_d, &cntPos, &cntNeg);
	printf("\nInitial magnetization: %9.6lf, up_s: %12llu, dw_s: %12llu\n",
	       abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (total_length*SPIN_X_WORD),
	       cntPos, cntNeg);


	CHECK_CUDA(cudaSetDevice(0));
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaEventRecord(start, 0));
  int iteration;
	// main update loop
	for(iteration = 0; iteration < total_updates; iteration++) {

		CHECK_CUDA(cudaSetDevice(0));
		spinUpdateV_2D_k<BLOCK_X, BLOCK_Y, BMULT_X, BMULT_Y, BIT_X_SPIN, C_BLACK, unsigned long long>
		<<<grid, block>>>
		(0, seed, iteration + 1, (XSL / 2) / SPIN_X_WORD / 2, YSL, 0, words_per_row / 2,
		 reinterpret_cast<float (*)[5]>(exp_d[0]),
		 reinterpret_cast<ulonglong2 *>(hamW_d),
		 reinterpret_cast<ulonglong2 *>(d_white_tiles),
		 reinterpret_cast<ulonglong2 *>(d_black_tiles));

		CHECK_CUDA(cudaSetDevice(0));
		spinUpdateV_2D_k<BLOCK_X, BLOCK_Y, BMULT_X, BMULT_Y, BIT_X_SPIN, C_WHITE, unsigned long long>
		<<<grid, block>>>
		(0, seed, iteration + 1, (XSL / 2) / SPIN_X_WORD / 2, YSL, 0, words_per_row / 2,
		 reinterpret_cast<float (*)[5]>(exp_d[0]),
		 reinterpret_cast<ulonglong2 *>(hamB_d),
		 reinterpret_cast<ulonglong2 *>(d_black_tiles),
		 reinterpret_cast<ulonglong2 *>(d_white_tiles));
	}
	CHECK_CUDA(cudaEventRecord(stop, 0));
	CHECK_CUDA(cudaEventSynchronize(stop));

	// compute total sum
	countSpins(redBlocks, total_length, sublattice_length, d_black_tiles, d_white_tiles, sum_d, &cntPos, &cntNeg);
	printf("Final   magnetization: %9.6lf, up_s: %12llu, dw_s: %12llu (iter: %8d)\n\n",
	       abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (total_length*SPIN_X_WORD),
	       cntPos, cntNeg, iteration);

	CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

	printf("Kernel execution time for %d update steps: %E ms, %.2lf flips/ns (BW: %.2lf GB/s)\n",
		iteration, elapsed_time, static_cast<double>(total_length*SPIN_X_WORD) * iteration / (elapsed_time * 1.0E+6),
		(2ull * iteration * (
			  sizeof(*d_spins)*((total_length / 2) + (total_length / 2) + (total_length / 2))  // src color read, dst color read, dst color write
			+ sizeof(*exp_d) * 5 * grid.x * grid.y ) * 1.0E-9) / (elapsed_time / 1.0E+3));

	CHECK_CUDA(cudaFree(d_spins));


	CHECK_CUDA(cudaFree(exp_d[0]));
	CHECK_CUDA(cudaFree(sum_d[0]));

  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaDeviceReset());

	return 0;
}
