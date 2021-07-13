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

#define BLOCK_DIMENSION_X_DEFINE (16)
#define BLOCK_DIMENSION_Y_DEFINE (16)
#define BMULT_X (2)
#define BMULT_Y (1)

#define TOTAL_UPDATES_DEFAULT (10000)
#define SEED_DEFAULT  (463463564571ull)


__device__ __forceinline__ unsigned int __mypopc(const unsigned int x) {return __popc(x);}

__device__ __forceinline__ unsigned long long int __mypopc(const unsigned long long int x) {return __popcll(x);}

enum {C_BLACK, C_WHITE};

// creates a vector with two components
__device__ __forceinline__ uint2 __mymake_int2(const unsigned int x, const unsigned int y) {return make_uint2(x, y);}

__device__ __forceinline__ ulonglong2 __mymake_int2(const unsigned long long x, const unsigned long long y) {return make_ulonglong2(x, y);}


template<int BLOCK_DIMENSION_X, int BLOCK_DIMENSION_Y, int LOOP_X, int LOOP_Y, int BITXSPIN, int COLOR,
         typename INT_T, typename INT2_T>
__global__  void initialise_traders(const long long seed, const long long number_of_columns, INT2_T *__restrict__ traders)
{
	const int row = blockIdx.y * BLOCK_DIMENSION_Y * LOOP_Y + threadIdx.y;
	const int col = blockIdx.x * BLOCK_DIMENSION_X * LOOP_X + threadIdx.x;

	const int SPIN_X_WORD = 8 * sizeof(INT_T) / BITXSPIN;

	const long long thread_id = ((gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * BLOCK_DIMENSION_X * BLOCK_DIMENSION_Y +
	                              threadIdx.y * BLOCK_DIMENSION_X + threadIdx.x;

	curandStatePhilox4_32_10_t rng;
	curand_init(seed, thread_id, static_cast<long long>(2 * SPIN_X_WORD) * LOOP_X * LOOP_Y * COLOR, &rng);

  // fill temporary 2d-array with 2d-vectors where both components are 0
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
			for(int bit_position = 0; bit_position < 8 * sizeof(INT_T); bit_position += BITXSPIN) {
				// These two if clauses are not identical since curand_uniform()
				// returns a different number on each invokation
				if (curand_uniform(&rng) < 0.5f) {
          /*
           * shift the spin with value 1 to its respective position and then
           * assign the matching bit the value 1 by using the bitwise
           * logical or operator |=
           * shift: 0000000000000000001 -> 0000000000010000000
           * logical bitwise or with tmp:
           * tmp[i][j] =                0000000000000001000
           * INT_T(1) << bit_position = 0000000000010000000
           * =>  tmp[i][j] =            0000000000010001000
           */
					__tmp[i][j].x |= INT_T(1) << bit_position;
				}
				if (curand_uniform(&rng) < 0.5f) {
					__tmp[i][j].y |= INT_T(1) << bit_position;
				}
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
      // copy temporary elements to their respective positions in the destination array
			traders[(row + i * BLOCK_DIMENSION_Y) * number_of_columns + col + j * BLOCK_DIMENSION_X] = __tmp[i][j];
		}
	}
	return;
}


template<int BLOCK_DIMENSION_X, int BLOCK_DIMENSION_Y, int TILE_SIZE_X, int TILE_SIZE_Y, typename INT2_T>
__device__ void load_tiles(const int grid_width, const int grid_height, const long long number_of_columns,
                           const INT2_T *__restrict__ traders, INT2_T tile[][TILE_SIZE_X + 2])
    /*
    Each block works on one tile with shape (TILE_SIZE_X, TILE_SIZE_Y).
    */
{
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	const int tile_start_x = blockIdx.x * TILE_SIZE_X;
	const int tile_start_y = blockIdx.y * TILE_SIZE_Y;

	#pragma unroll
	for(int tile_row = 0; tile_row < TILE_SIZE_Y; tile_row += BLOCK_DIMENSION_Y) {
		int row_in_source = tile_start_y + tile_row + tidy;

		#pragma unroll
		for(int tile_col = 0; tile_col < TILE_SIZE_X; tile_col += BLOCK_DIMENSION_X) {
			const int column_in_source = tile_start_x + tile_col + tidx;
			tile[1 + tile_row + tidy][1 + tile_col + tidx] = traders[row_in_source * number_of_columns + column_in_source];
		}
	}
	if (tidy == 0) {
		int row_in_source = (tile_start_y % grid_height) == 0 ? tile_start_y + grid_height - 1 : tile_start_y - 1;

		#pragma unroll
		for(int tile_col = 0; tile_col < TILE_SIZE_X; tile_col += BLOCK_DIMENSION_X) {
			const int column_in_source = tile_start_x + tile_col + tidx;
			tile[0][1 + tile_col + tidx] = traders[row_in_source * number_of_columns + column_in_source];
		}

		row_in_source = ((tile_start_y + TILE_SIZE_Y) % grid_height) == 0 ? tile_start_y + TILE_SIZE_Y - grid_height : tile_start_y + TILE_SIZE_Y;

		#pragma unroll
		for(int i = 0; i < TILE_SIZE_X; i += BLOCK_DIMENSION_X) {
			const int column_in_source = tile_start_x + i + tidx;
			tile[1 + TILE_SIZE_Y][1 + i + tidx] = traders[row_in_source * number_of_columns + column_in_source];
		}

		// the other branch in slower so skip it if possible
		if (BLOCK_DIMENSION_X <= TILE_SIZE_Y) {

			int column_in_source = (tile_start_x % grid_width) == 0 ? tile_start_x + grid_width - 1 : tile_start_x - 1;

			#pragma unroll
			for(int j = 0; j < TILE_SIZE_Y; j += BLOCK_DIMENSION_X) {
				row_in_source = tile_start_y + j + tidx;
				tile[1 + j + tidx][0] = traders[row_in_source * number_of_columns + column_in_source];
			}

			column_in_source = ((tile_start_x + TILE_SIZE_X) % grid_width) == 0 ? tile_start_x + TILE_SIZE_X - grid_width : tile_start_x + TILE_SIZE_X;

			#pragma unroll
			for(int j = 0; j < TILE_SIZE_Y; j += BLOCK_DIMENSION_X) {
				row_in_source = tile_start_y + j + tidx;
				tile[1 + j + tidx][1 + TILE_SIZE_X] = traders[row_in_source * number_of_columns + column_in_source];
			}
		} else {
			if (tidx < TILE_SIZE_Y) {
				int column_in_source = (tile_start_x % grid_width) == 0 ? tile_start_x + grid_width-1 : tile_start_x - 1;

				row_in_source = tile_start_y + tidx;
				tile[1 + tidx][0] = traders[row_in_source * number_of_columns + column_in_source];;

				column_in_source = ((tile_start_x + TILE_SIZE_X) % grid_width) == 0 ? tile_start_x + TILE_SIZE_X - grid_width : tile_start_x + TILE_SIZE_X;
				tile[1 + tidx][1 + TILE_SIZE_X] = traders[row_in_source * number_of_columns + column_in_source];
			}
		}
	}
	return;
}


template<int BLOCK_DIMENSION_X, int BLOCK_DIMENSION_Y, int LOOP_X, int LOOP_Y, int BITXSPIN, int COLOR, typename INT_T, typename INT2_T>
__global__ void update_strategies(const long long seed, const int number_of_previous_iterations,
		      const int grid_width, // lattice width of one color in words
		      const int grid_height, // lattice height (not in words)
		      const long long number_of_columns,
		      const float vExp[][5],
		      const INT2_T *__restrict__ traders,
		            INT2_T *__restrict__ checkerboard_agents)
{
	const int SPIN_X_WORD = 8 * sizeof(INT_T) / BITXSPIN;

	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	__shared__ INT2_T shared_tiles[BLOCK_DIMENSION_Y * LOOP_Y + 2][BLOCK_DIMENSION_X * LOOP_X + 2];

	load_tiles<BLOCK_DIMENSION_X, BLOCK_DIMENSION_Y, BLOCK_DIMENSION_X * LOOP_X, BLOCK_DIMENSION_Y * LOOP_Y, INT2_T>
  (grid_width, grid_height, number_of_columns, traders, shared_tiles);

	__shared__ float __shExp[2][5];

  // load precomputed exponentials into shared memory
  // in case a block consists of less than 2 * 5 threads iterate over the
  // precomputed array in each thread
	#pragma unroll
	for(int i = 0; i < 2; i += BLOCK_DIMENSION_Y) {
		#pragma unroll
		for(int j = 0; j < 5; j += BLOCK_DIMENSION_X) {
			if (i + tidy < 2 && j + tidx < 5) {
				__shExp[i + tidy][j + tidx] = vExp[i + tidy][j + tidx];
			}
		}
	}
	__syncthreads();

	const int row = blockIdx.y * BLOCK_DIMENSION_Y * LOOP_Y + tidy;
	const int col = blockIdx.x * BLOCK_DIMENSION_X * LOOP_X + tidx;

	const long long thread_id = (blockIdx.y * gridDim.x + blockIdx.x) * BLOCK_DIMENSION_X * BLOCK_DIMENSION_Y
                            +  threadIdx.y*BLOCK_DIMENSION_X + threadIdx.x;

	INT2_T __me[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__me[i][j] = checkerboard_agents[(row + i * BLOCK_DIMENSION_Y) * number_of_columns + col + j * BLOCK_DIMENSION_X];
		}
	}

	INT2_T __up[LOOP_Y][LOOP_X];
	INT2_T __ct[LOOP_Y][LOOP_X];
	INT2_T __dw[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__up[i][j] = shared_tiles[i * BLOCK_DIMENSION_Y +     tidy][j * BLOCK_DIMENSION_X + 1 + tidx];
			__ct[i][j] = shared_tiles[i * BLOCK_DIMENSION_Y + 1 + tidy][j * BLOCK_DIMENSION_X + 1 + tidx];
			__dw[i][j] = shared_tiles[i * BLOCK_DIMENSION_Y + 2 + tidy][j * BLOCK_DIMENSION_X + 1 + tidx];
		}
	}

	// BLOCK_DIMENSION_Y is power of two so row parity won't change across loops
	const int read_black = (COLOR == C_BLACK) ? !(row % 2) : (row%2);

	INT2_T __sd[LOOP_Y][LOOP_X];

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			__sd[i][j] = (read_black) ? shared_tiles[i*BLOCK_DIMENSION_Y + 1+tidy][j*BLOCK_DIMENSION_X +   tidx]:
						  shared_tiles[i*BLOCK_DIMENSION_Y + 1+tidy][j*BLOCK_DIMENSION_X + 2+tidx];
		}
	}

	if (read_black) {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__sd[i][j].x = (__ct[i][j].x << BITXSPIN) | (__sd[i][j].y >> (8*sizeof(__sd[i][j].y)-BITXSPIN));
				__sd[i][j].y = (__ct[i][j].y << BITXSPIN) | (__ct[i][j].x >> (8*sizeof(__ct[i][j].x)-BITXSPIN));
			}
		}
	} else {
		#pragma unroll
		for(int i = 0; i < LOOP_Y; i++) {
			#pragma unroll
			for(int j = 0; j < LOOP_X; j++) {
				__sd[i][j].y = (__ct[i][j].y >> BITXSPIN) | (__sd[i][j].x << (8*sizeof(__sd[i][j].x)-BITXSPIN));
				__sd[i][j].x = (__ct[i][j].x >> BITXSPIN) | (__ct[i][j].y << (8*sizeof(__ct[i][j].y)-BITXSPIN));
			}
		}
	}

	curandStatePhilox4_32_10_t rng;
	curand_init(seed, thread_id, static_cast<long long>(2*SPIN_X_WORD)*LOOP_X*LOOP_Y*(2*number_of_previous_iterations+COLOR), &rng);

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
			for(int z = 0; z < 8 * sizeof(INT_T); z += BITXSPIN) {

				const int2 __src = make_int2((__me[i][j].x >> z) & 0xF,
							     (__me[i][j].y >> z) & 0xF);

				const int2 __sum = make_int2((__ct[i][j].x >> z) & 0xF,
							     (__ct[i][j].y >> z) & 0xF);

				const INT_T ONE = static_cast<INT_T>(1);

				if (curand_uniform(&rng) <= __shExp[__src.x][__sum.x]) {
					__me[i][j].x ^= ONE << z;
				}
				if (curand_uniform(&rng) <= __shExp[__src.y][__sum.y]) {
					__me[i][j].y ^= ONE << z;
				}
			}
		}
	}

	#pragma unroll
	for(int i = 0; i < LOOP_Y; i++) {
		#pragma unroll
		for(int j = 0; j < LOOP_X; j++) {
			checkerboard_agents[(row+i*BLOCK_DIMENSION_Y)*number_of_columns + col+j*BLOCK_DIMENSION_X] = __me[i][j];
		}
	}
	return;
}


template<int BLOCK_DIMENSION_X, int WSIZE, typename T>
__device__ __forceinline__ T __block_sum(T traders)
{
	__shared__ T sh[BLOCK_DIMENSION_X / WSIZE];

	const int lid = threadIdx.x%WSIZE;
	const int wid = threadIdx.x/WSIZE;

	#pragma unroll
	for(int i = WSIZE/2; i; i >>= 1) {
		traders += __shfl_down_sync(0xFFFFFFFF, traders, i);
	}
	if (lid == 0) sh[wid] = traders;

	__syncthreads();
	if (wid == 0) {
		traders = (lid < (BLOCK_DIMENSION_X / WSIZE)) ? sh[lid] : 0;

		#pragma unroll
		for(int i = (BLOCK_DIMENSION_X/WSIZE)/2; i; i >>= 1) {
			traders += __shfl_down_sync(0xFFFFFFFF, traders, i);
		}
	}
	__syncthreads();
	return traders;
}

// to be optimized
template<int BLOCK_DIMENSION_X, int BITXSPIN, typename INT_T, typename SUM_T>
__global__ void getMagn_k(const long long n,
			                    const INT_T *__restrict__ traders,
			                    SUM_T *__restrict__ sum)
{

	const int SPIN_X_WORD = 8*sizeof(INT_T)/BITXSPIN;

	const long long nth = static_cast<long long>(blockDim.x)*gridDim.x;
	const long long thread_id = static_cast<long long>(blockDim.x)*blockIdx.x + threadIdx.x;

	SUM_T __cntP = 0;
	SUM_T __cntN = 0;

	for(long long i = 0; i < n; i += nth) {
		if (i+thread_id < n) {
			const int __c = __mypopc(traders[i+thread_id]);
			__cntP += __c;
			__cntN += SPIN_X_WORD - __c;
		}
	}
	__cntP = __block_sum<BLOCK_DIMENSION_X, 32>(__cntP);
	__cntN = __block_sum<BLOCK_DIMENSION_X, 32>(__cntN);

	if (threadIdx.x == 0) {
		atomicAdd(sum + 0, __cntP);
		atomicAdd(sum + 1, __cntN);
	}
	return;
}


static void countSpins(const int redBlocks,
								       const size_t total_length,
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


int main(int argc, char **argv) {

	unsigned long long *d_spins = NULL;
	unsigned long long *d_black_tiles = NULL;
	unsigned long long *d_white_tiles = NULL;

	cudaEvent_t start, stop;
  float elapsed_time;

	const int SPIN_X_WORD = (8 * sizeof(*d_spins)) / BIT_X_SPIN;

	int grid_width = 2048;
	int grid_height = 2048;

	int total_updates = TOTAL_UPDATES_DEFAULT;

	unsigned long long seed = SEED_DEFAULT;

	float temp  = 0.666f;

	int XSL = grid_width;
	int YSL = grid_height;

	if (!grid_width || (grid_width % 2) || ((grid_width / 2) % (SPIN_X_WORD * 2 * BLOCK_DIMENSION_X_DEFINE * BMULT_X))) {
		fprintf(stderr, "\nPlease specify an grid_width dim multiple of %d\n\n", 2 * SPIN_X_WORD * 2 * BLOCK_DIMENSION_X_DEFINE * BMULT_X);
		exit(EXIT_FAILURE);
	}
	if (!grid_height || (grid_height % (BLOCK_DIMENSION_Y_DEFINE * BMULT_Y))) {
		fprintf(stderr, "\nPlease specify a grid_height dim multiple of %d\n\n", BLOCK_DIMENSION_Y_DEFINE * BMULT_Y);
		exit(EXIT_FAILURE);
	}

	cudaDeviceProp props;

	printf("\nUsing GPU:\n");

	CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
	printf("\t%2d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
		0, props.name, props.multiProcessorCount,
		props.maxThreadsPerMultiProcessor,
		props.major, props.minor,
		props.ECCEnabled?"on":"off");

	printf("\n");

	size_t words_per_row = (grid_width / 2) / SPIN_X_WORD;
	// total lattice length
	size_t total_length = 2ull * static_cast<size_t>(grid_height) * words_per_row;

	dim3 grid(DIV_UP(words_per_row / 2, BLOCK_DIMENSION_X_DEFINE * BMULT_X), DIV_UP(grid_height, BLOCK_DIMENSION_Y_DEFINE * BMULT_Y));
	dim3 block(BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE);

	printf("Run configuration:\n");
	printf("\tspin/word: %d\n", SPIN_X_WORD);
	printf("\tspins: %zu\n", total_length * SPIN_X_WORD);
	printf("\tseed: %llu\n", seed);
	printf("\titerations: %d\n", total_updates);
	printf("\tblock (x, y): %d, %d\n", block.x, block.y);
	printf("\ttile  (x, y): %d, %d\n", BLOCK_DIMENSION_X_DEFINE * BMULT_X, BLOCK_DIMENSION_Y_DEFINE * BMULT_Y);
	printf("\tgrid  (x, y): %d, %d\n", grid.x, grid.y);

	printf("\ttemp: %f (%f*T_crit)\n", temp, temp / CRIT_TEMP);

	printf("\n");

	printf("\tlattice size:      %8d x %8d\n", grid_height, grid_width);
	printf("\tlattice shape: 2 x %8d x %8zu (%12zu %s)\n", grid_height, words_per_row, total_length, sizeof(*d_spins) == 4 ? "uints" : "ulls");
	printf("\tmemory: %.2lf MB \n", (total_length * sizeof(*d_spins)) / (1024.0 * 1024.0));

	const int redBlocks = MIN(DIV_UP(total_length, THREADS),
				  (props.maxThreadsPerMultiProcessor/THREADS)*props.multiProcessorCount);

	unsigned long long cntPos;
	unsigned long long cntNeg;
	unsigned long long *sum_d[0];

	CHECK_CUDA(cudaMalloc(&d_spins, total_length*sizeof(*d_spins)));
	CHECK_CUDA(cudaMemset(d_spins, 0, total_length*sizeof(*d_spins)));

	CHECK_CUDA(cudaMalloc(&sum_d[0], 2*sizeof(**sum_d)));


	d_black_tiles = d_spins;
	d_white_tiles = d_spins + total_length/2;

	float *exp_d[0];
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
	initialise_traders<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BMULT_X, BMULT_Y, BIT_X_SPIN, C_BLACK, unsigned long long>
	<<<grid, block>>>
	(seed, words_per_row / 2, reinterpret_cast<ulonglong2 *>(d_black_tiles));
	CHECK_ERROR("initialise_traders");

	initialise_traders<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BMULT_X, BMULT_Y, BIT_X_SPIN, C_WHITE, unsigned long long>
	<<<grid, block>>>
	(seed, words_per_row / 2, reinterpret_cast<ulonglong2 *>(d_white_tiles));
	CHECK_ERROR("initialise_traders");

	// computes sum over array
	countSpins(redBlocks, total_length, d_black_tiles, d_white_tiles, sum_d, &cntPos, &cntNeg);
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
		update_strategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BMULT_X, BMULT_Y, BIT_X_SPIN, C_BLACK, unsigned long long>
		<<<grid, block>>>
		(seed, iteration + 1, (XSL / 2) / SPIN_X_WORD / 2, YSL, words_per_row / 2,
		 reinterpret_cast<float (*)[5]>(exp_d[0]),
		 reinterpret_cast<ulonglong2 *>(d_white_tiles),
		 reinterpret_cast<ulonglong2 *>(d_black_tiles));

		CHECK_CUDA(cudaSetDevice(0));
		update_strategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BMULT_X, BMULT_Y, BIT_X_SPIN, C_WHITE, unsigned long long>
		<<<grid, block>>>
		(seed, iteration + 1, (XSL / 2) / SPIN_X_WORD / 2, YSL, words_per_row / 2,
		 reinterpret_cast<float (*)[5]>(exp_d[0]),
		 reinterpret_cast<ulonglong2 *>(d_black_tiles),
		 reinterpret_cast<ulonglong2 *>(d_white_tiles));
	}
	CHECK_CUDA(cudaEventRecord(stop, 0));
	CHECK_CUDA(cudaEventSynchronize(stop));

	// compute total sum
	countSpins(redBlocks, total_length, d_black_tiles, d_white_tiles, sum_d, &cntPos, &cntNeg);
	printf("Final   magnetization: %9.6lf, up_s: %12llu, dw_s: %12llu (iter: %8d)\n\n",
	       abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (total_length*SPIN_X_WORD),
	       cntPos, cntNeg, iteration);

	CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

	printf("Kernel execution time for %d update steps: %E ms, %.2lf flips/ns (BW: %.2lf GB/s)\n",
		iteration, elapsed_time, static_cast<double>(total_length * SPIN_X_WORD) * iteration / (elapsed_time * 1.0E+6),
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
