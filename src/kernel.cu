/*
 * Copyright (c) 2021, Keno Krieger, <kriegerk@uni-bremen.de>. All rights reserved.
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
#include "cudamacro.h"
#include "traders.cuh"


#define DIV_UP(a,b)  (((a) + ((b) - 1)) / (b))
#define MIN(a,b)	(((a) < (b)) ? (a) : (b))
#define MAX(a,b)	(((a) > (b)) ? (a) : (b))

#define THREADS 128
#define BIT_X_SPIN (4)

/*
 * 2048+: 16, 16, 2, 1
 *  1024: 16, 16, 1, 2
 *   512:  8,  8, 1, 1
 *   256:  4,  8, 1, 1
 *   128:  2,  8, 1, 1
*/
#define BLOCK_DIMENSION_X_DEFINE (16)
#define BLOCK_DIMENSION_Y_DEFINE (16)

#define TOTAL_UPDATES_DEFAULT (10000)
#define SEED_DEFAULT  (463463564571ull)


static void countSpins(const int redBlocks,
								       const size_t total_words,
								       const unsigned long long *d_black_tiles,
								       const unsigned long long *d_white_tiles,
									     unsigned long long **sum_d,
									     unsigned long long *bsum,
									     unsigned long long *wsum)
{
	CHECK_CUDA(cudaMemset(sum_d[0], 0, 2*sizeof(**sum_d)));
	getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(total_words, d_black_tiles, sum_d[0]);
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

	if (!grid_width || (grid_width % 2) || ((grid_width / 2) % (2 * SPIN_X_WORD * BLOCK_DIMENSION_X_DEFINE))) {
		fprintf(stderr, "\nPlease specify an grid_width dim multiple of %d\n\n", 2 * SPIN_X_WORD * 2 * BLOCK_DIMENSION_X_DEFINE);
		exit(EXIT_FAILURE);
	}
	if (!grid_height || (grid_height % (BLOCK_DIMENSION_Y_DEFINE))) {
		fprintf(stderr, "\nPlease specify a grid_height dim multiple of %d\n\n", BLOCK_DIMENSION_Y_DEFINE);
		exit(EXIT_FAILURE);
	}

	cudaDeviceProp props;

	printf("\nUsing GPU:\n");

	CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
	printf("\t%2d (%s, %d SMs, %d th/SM max, CC %d.%d, ECC %s)\n",
		0, props.name, props.multiProcessorCount,
		props.maxThreadsPerMultiProcessor,
		props.major, props.minor,
		props.ECCEnabled ? "on" : "off");

	printf("\n");

	size_t words_per_row = (grid_width / 2) / SPIN_X_WORD;
	// total lattice length
	size_t total_words = 2ull * static_cast<size_t>(grid_height) * words_per_row;

	// words_per_row / 2 because each entry in the array has two components
	dim3 blocks(DIV_UP(words_per_row / 2, BLOCK_DIMENSION_X_DEFINE), DIV_UP(grid_height, BLOCK_DIMENSION_Y_DEFINE));
	dim3 threads_per_block(BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE);

	printf("Run configuration:\n");
	printf("\tspin/word: %d\n", SPIN_X_WORD);
	printf("\tspins: %zu\n", total_words * SPIN_X_WORD);
	printf("\tseed: %llu\n", seed);
	printf("\titerations: %d\n", total_updates);
	printf("\tblock (x, y): %d, %d\n", threads_per_block.x, threads_per_block.y);
	printf("\ttile  (x, y): %d, %d\n", BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE);
	printf("\tgrid  (x, y): %d, %d\n", blocks.x, blocks.y);

	printf("\ttemp: %f \n", temp);

	printf("\n");

	printf("\tlattice size:      %8d x %8d\n", grid_height, grid_width);
	printf("\tlattice shape: 2 x %8d x %8zu (%12zu %s)\n", grid_height, words_per_row, total_words, sizeof(*d_spins) == 4 ? "uints" : "ulls");
	printf("\tmemory: %.2lf MB \n", (total_words * sizeof(*d_spins)) / (1024.0 * 1024.0));

	const int redBlocks = MIN(DIV_UP(total_words, THREADS), (props.maxThreadsPerMultiProcessor / THREADS) * props.multiProcessorCount);

	unsigned long long cntPos;
	unsigned long long cntNeg;
	unsigned long long *sum_d[0];

	CHECK_CUDA(cudaMalloc(&d_spins, total_words * sizeof(*d_spins)));
	CHECK_CUDA(cudaMemset(d_spins, 0, total_words * sizeof(*d_spins)));

	CHECK_CUDA(cudaMalloc(&sum_d[0], 2 * sizeof(**sum_d)));


	d_black_tiles = d_spins;
	d_white_tiles = d_spins + total_words / 2;

	float *exp_d[0];
	float  exp_h[2][5];

	// precompute possible exponentials
	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 5; j++) {
			if(temp > 0) {
				exp_h[i][j] = expf((i ? -2.0f : 2.0f) * static_cast<float>(j * 2 - 4) * (1.0f / temp));
			} else {
				if(j == 2) {
					exp_h[i][j] = 0.5f;
				} else {
					exp_h[i][j] = (i ? -2.0f : 2.0f) * static_cast<float>(j * 2 - 4);
				}
			}
		}
	}

	CHECK_CUDA(cudaSetDevice(0));
	CHECK_CUDA(cudaMalloc(exp_d, 2 * 5 * sizeof(**exp_d)));
	CHECK_CUDA(cudaMemcpy(exp_d[0], exp_h, 2 * 5 * sizeof(**exp_d), cudaMemcpyHostToDevice));

	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	CHECK_CUDA(cudaSetDevice(0));
	initialise_traders<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_BLACK, unsigned long long>
	<<<blocks, threads_per_block>>>
	(seed, words_per_row / 2, reinterpret_cast<ulonglong2 *>(d_black_tiles));
	CHECK_ERROR("initialise_traders");

	initialise_traders<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_WHITE, unsigned long long>
	<<<blocks, threads_per_block>>>
	(seed, words_per_row / 2, reinterpret_cast<ulonglong2 *>(d_white_tiles));
	CHECK_ERROR("initialise_traders");

	// compute sum over array
	countSpins(redBlocks, total_words, d_black_tiles, d_white_tiles, sum_d, &cntPos, &cntNeg);
	printf("\nInitial magnetization: %9.6lf, up_s: %12llu, dw_s: %12llu\n",
	       abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (total_words*SPIN_X_WORD),
	       cntPos, cntNeg);

	CHECK_CUDA(cudaSetDevice(0));
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaEventRecord(start, 0));
  int iteration;
	// main update loop
	for(iteration = 0; iteration < total_updates; iteration++) {

		CHECK_CUDA(cudaSetDevice(0));
		update_strategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_BLACK, unsigned long long>
		<<<blocks, threads_per_block>>>
		(seed, iteration + 1, (grid_width / 2) / SPIN_X_WORD / 2, grid_height, words_per_row / 2,
		 reinterpret_cast<float (*)[5]>(exp_d[0]),
		 reinterpret_cast<ulonglong2 *>(d_white_tiles),
		 reinterpret_cast<ulonglong2 *>(d_black_tiles));

		CHECK_CUDA(cudaSetDevice(0));
		update_strategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_WHITE, unsigned long long>
		<<<blocks, threads_per_block>>>
		(seed, iteration + 1, (grid_width / 2) / SPIN_X_WORD / 2, grid_height, words_per_row / 2,
		 reinterpret_cast<float (*)[5]>(exp_d[0]),
		 reinterpret_cast<ulonglong2 *>(d_black_tiles),
		 reinterpret_cast<ulonglong2 *>(d_white_tiles));
	}
	CHECK_CUDA(cudaEventRecord(stop, 0));
	CHECK_CUDA(cudaEventSynchronize(stop));

	// compute total sum
	countSpins(redBlocks, total_words, d_black_tiles, d_white_tiles, sum_d, &cntPos, &cntNeg);
	printf("Final   magnetization: %9.6lf, up_s: %12llu, dw_s: %12llu (iter: %8d)\n\n",
	       abs(static_cast<double>(cntPos)-static_cast<double>(cntNeg)) / (total_words*SPIN_X_WORD),
	       cntPos, cntNeg, iteration);

	CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

	printf("Kernel execution time for %d update steps: %E ms, %.2lf flips/ns (BW: %.2lf GB/s)\n",
		iteration, elapsed_time, static_cast<double>(total_words * SPIN_X_WORD) * iteration / (elapsed_time * 1.0E+6),
		(2ull * iteration * (
			  sizeof(*d_spins)*((total_words / 2) + (total_words / 2) + (total_words / 2))  // src color read, dst color read, dst color write
			+ sizeof(*exp_d) * 5 * blocks.x * blocks.y ) * 1.0E-9) / (elapsed_time / 1.0E+3));

	CHECK_CUDA(cudaFree(d_spins));
	CHECK_CUDA(cudaFree(exp_d[0]));
	CHECK_CUDA(cudaFree(sum_d[0]));

  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaDeviceReset());

	return 0;
}
