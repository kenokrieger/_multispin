#ifndef traders_cuh
#define traders_cuh

#include <curand_kernel.h>
#include "cudamacro.h"

#define BIT_X_SPIN (3)
#define THREADS 128

#define BLOCK_DIMENSION_X_DEFINE (4)
#define BLOCK_DIMENSION_Y_DEFINE (8)
#define BLOCK_DIMENSION_Z_DEFINE (8)

enum {C_BLACK, C_WHITE};

// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
__device__ __forceinline__ unsigned int __custom_popc(const unsigned int x) {return __popc(x);}
__device__ __forceinline__ unsigned long long int __custom_popc(const unsigned long long int x) {return __popcll(x);}

// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
__device__ __forceinline__ uint2 __custom_make_int2(const unsigned int x, const unsigned int y) {return make_uint2(x, y);}
__device__ __forceinline__ ulonglong2 __custom_make_int2(const unsigned long long x, const unsigned long long y) {return make_ulonglong2(x, y);}


template<int BITXSPIN, int COLOR, typename INT_T, typename INT2_T>
__global__  void initialise_traders(const unsigned long long seed,
																		const long long number_of_columns,
																		const long long lattice_size,
																	  INT2_T *__restrict__ traders,
																		float percentage = 0.5f)
{
	const int row = blockIdx.y * blockDim.y + threadIdx.y;
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int lid = blockIdx.z * blockDim.z + threadIdx.z;
  const long long index = lid * lattice_size + row * number_of_columns + col;
	const int SPIN_X_WORD = (8 * sizeof(INT_T)) / BITXSPIN;

	curandStatePhilox4_32_10_t rng;
	curand_init(seed, index, static_cast<long long>(2 * SPIN_X_WORD) * COLOR, &rng);

  traders[index] = __custom_make_int2(INT_T(0), INT_T(0));
	for(int spin_position = 0; spin_position < 8 * sizeof(INT_T); spin_position += BITXSPIN) {
		// The two if clauses are not identical since curand_uniform()
		// returns a different number on each invokation
		if (curand_uniform(&rng) < percentage) {
			traders[index].x |= INT_T(1) << spin_position;
		}
		if (curand_uniform(&rng) < percentage) {
			traders[index].y |= INT_T(1) << spin_position;
		}
	}
	return;
}


template<typename INT_T, typename INT2_T>
void initialise_arrays(dim3 blocks, dim3 threads_per_block,
								 			 const unsigned long long seed,
											 const unsigned long long number_of_columns,
											 const unsigned long long lattice_size,
								 		 	 INT2_T *__restrict__ d_black_tiles,
											 INT2_T *__restrict__ d_white_tiles,
								 		 	 float percentage = 0.5f)
{
		CHECK_CUDA(cudaSetDevice(0));
		initialise_traders<BIT_X_SPIN, C_BLACK, INT_T>
		<<<blocks, threads_per_block>>>
		(seed, number_of_columns, lattice_size, reinterpret_cast<ulonglong2 *>(d_black_tiles), percentage);
		CHECK_ERROR("initialise_traders");

		initialise_traders<BIT_X_SPIN, C_WHITE, INT_T>
		<<<blocks, threads_per_block>>>
		(seed, number_of_columns, lattice_size, reinterpret_cast<ulonglong2 *>(d_white_tiles), percentage);
		CHECK_ERROR("initialise_traders");
}


template<int TILE_SIZE_X, int TILE_SIZE_Y, typename INT2_T>
__device__ void load_tiles(const int grid_width, const int grid_height, const long long number_of_columns,
                           const INT2_T *__restrict__ traders, INT2_T tile[][TILE_SIZE_Y + 2][TILE_SIZE_X + 2])
{
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	const int tidz = threadIdx.z;
	const int lattice_offset = (blockIdx.z * blockDim.z + tidz) * number_of_columns * grid_height;

	const int tile_start_x = blockIdx.x * TILE_SIZE_X;
	const int tile_start_y = blockIdx.y * TILE_SIZE_Y;

	int row = tile_start_y + tidy;
	int col = tile_start_x + tidx;
	tile[tidz][1 + tidy][1 + tidx] = traders[lattice_offset + row * number_of_columns + col];

	if (tidy == 0) {
		row = (tile_start_y % grid_height) == 0 ? tile_start_y + grid_height - 1 : tile_start_y - 1;
		tile[tidz][0][1 + tidx] = traders[lattice_offset + row * number_of_columns + col];

		row = ((tile_start_y + TILE_SIZE_Y) % grid_height) == 0 ? tile_start_y + TILE_SIZE_Y - grid_height : tile_start_y + TILE_SIZE_Y;
		tile[tidz][1 + TILE_SIZE_Y][1 + tidx] = traders[lattice_offset + row * number_of_columns + col];

		row = tile_start_y + tidx;
		col = (tile_start_x % grid_width) == 0 ? tile_start_x + grid_width - 1 : tile_start_x - 1;
		tile[tidz][1 + tidx][0] = traders[lattice_offset + row * number_of_columns + col];

		row = tile_start_y + tidx;
		col = ((tile_start_x + TILE_SIZE_X) % grid_width) == 0 ? tile_start_x + TILE_SIZE_X - grid_width : tile_start_x + TILE_SIZE_X;
		tile[tidz][1 + tidx][1 + TILE_SIZE_X] = traders[lattice_offset + row * number_of_columns + col];
	}
	return;
}


__device__ void load_probabilities(const float *precomputed_probabilities, float shared_probabilities[][7])
{
  // load precomputed exponentials into shared memory.
  // in case a threads_per_block consists of less than 2 * 7 threads
  // multiple iterations in each thread are needed
  //in most cases loops will only trigger once
  #pragma unroll
  for(int i = 0; i < 2; i += blockDim.y) {
    #pragma unroll
    for(int j = 0; j < 7; j += blockDim.x) {
      if (i + threadIdx.y < 2 && j + threadIdx.x < 7) {
        shared_probabilities[i + threadIdx.y][j + threadIdx.x] = precomputed_probabilities[(i + threadIdx.y) * 7 + j + threadIdx.x];
      }
    }
  }
  return;
}


template<int BLOCK_DIMENSION_X, int BITXSPIN, int SPIN_X_WORD, typename INT2_T>
__device__ INT2_T compute_neighbor_sum(INT2_T front_neighbor, INT2_T back_neighbor,
	 																		 INT2_T shared_tiles[][BLOCK_DIMENSION_X + 2],
	 																		 const int tidx, const int tidy,
																			 const int shift_left)
{
	// three nearest neighbors
	INT2_T upper_neighbor  = shared_tiles[    tidy][1 + tidx];
	INT2_T center_neighbor = shared_tiles[1 + tidy][1 + tidx];
	INT2_T lower_neighbor  = shared_tiles[2 + tidy][1 + tidx];

	// remaining neighbor, either left or right
	INT2_T horizontal_neighbor = (shift_left) ? shared_tiles[1 + tidy][tidx] : shared_tiles[1 + tidy][2 + tidx];

	if (shift_left) {
		horizontal_neighbor.x = (center_neighbor.x << BITXSPIN) | (horizontal_neighbor.y >> (SPIN_X_WORD *  (BITXSPIN - 1)));
		horizontal_neighbor.y = (center_neighbor.y << BITXSPIN) | (center_neighbor.x >> (SPIN_X_WORD *  (BITXSPIN - 1)));
	} else {
		horizontal_neighbor.y = (center_neighbor.y >> BITXSPIN) | (horizontal_neighbor.x << (SPIN_X_WORD *  (BITXSPIN - 1)));
		horizontal_neighbor.x = (center_neighbor.x >> BITXSPIN) | (center_neighbor.y << (SPIN_X_WORD *  (BITXSPIN - 1)));
	}

	// this basically sums over all spins/word in parallel
	center_neighbor.x += upper_neighbor.x + lower_neighbor.x + horizontal_neighbor.x + front_neighbor.x + back_neighbor.x;
	center_neighbor.y += upper_neighbor.y + lower_neighbor.y + horizontal_neighbor.y + front_neighbor.y + back_neighbor.y;

	return center_neighbor;
}


template<int BITXSPIN, typename INT_T, typename INT2_T>
__device__ INT2_T flip_spins(curandStatePhilox4_32_10_t rng,
	 													 INT2_T target, INT2_T parallel_sum,
														 const float shared_probabilities[][7])
{
	const INT_T ONE = static_cast<INT_T>(1);
	for(int spin_position = 0; spin_position < 8 * sizeof(INT_T); spin_position += BITXSPIN) {

		const int2 spin = make_int2((target.x >> spin_position) & 0x7, (target.y >> spin_position) & 0x7);
		const int2 sum = make_int2((parallel_sum.x >> spin_position) & 0x7, (parallel_sum.y >> spin_position) & 0x7);

		if (curand_uniform(&rng) <= shared_probabilities[spin.x][sum.x]) {
			target.x |= (ONE << spin_position);
		} else {
			target.x &= ~(ONE << spin_position);
		}
		if (curand_uniform(&rng) <= shared_probabilities[spin.y][sum.y]) {
			target.y |= (ONE << spin_position);
		} else {
			target.y &= ~(ONE << spin_position);
		}
	}
	return target;
}


template<int BLOCK_DIMENSION_X, int BLOCK_DIMENSION_Y, int BLOCK_DIMENSION_Z, int BITXSPIN, int COLOR, typename INT_T, typename INT2_T>
__global__ void update_strategies(const unsigned long long seed, const int number_of_previous_iterations,
		       const int grid_width, // lattice width of one color in words
		       const int grid_height, // lattice height (not in words)
					 const int grid_depth,
		       const long long number_of_columns,
		       const float *precomputed_probabilities,
		       const INT2_T *__restrict__ checkerboard_agents,
		             INT2_T *__restrict__ traders)
{
	const int SPIN_X_WORD = (8 * sizeof(INT_T)) / BITXSPIN;
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	const int tidz = threadIdx.z;

	__shared__ INT2_T shared_tiles[BLOCK_DIMENSION_Z][BLOCK_DIMENSION_Y + 2][BLOCK_DIMENSION_X + 2];
	load_tiles<BLOCK_DIMENSION_X, BLOCK_DIMENSION_Y, INT2_T>
  (grid_width, grid_height, number_of_columns, checkerboard_agents, shared_tiles);

	__shared__ float shared_probabilities[2][7];
	load_probabilities(precomputed_probabilities, shared_probabilities);

	__syncthreads();


	const int row = blockIdx.y * BLOCK_DIMENSION_Y + tidy;
	const int col = blockIdx.x * BLOCK_DIMENSION_X + tidx;
	const int lid = blockIdx.z * BLOCK_DIMENSION_Z + tidz;
	const int first_tile_black = (lid % 2) ? (COLOR != C_BLACK) : (COLOR == C_BLACK);
	const int shift_left = (first_tile_black) ? !(row % 2) : (row % 2);
	const long long index = lid * number_of_columns * grid_height + row * number_of_columns + col;

	const long long back_index = ((lid - 1 < 0) ? grid_depth - 1 : lid - 1) * number_of_columns * grid_height + row * number_of_columns + col;
	const long long front_index = ((lid + 1 > grid_depth - 1) ? 0 : lid + 1) * number_of_columns * grid_height + row * number_of_columns + col;
	INT2_T front_neighbor = checkerboard_agents[front_index];
	INT2_T back_neighbor = checkerboard_agents[back_index];
	INT2_T parallel_sum = compute_neighbor_sum<BLOCK_DIMENSION_X, BITXSPIN, SPIN_X_WORD, INT2_T>(front_neighbor, back_neighbor, shared_tiles[tidz], tidx, tidy, shift_left);

	curandStatePhilox4_32_10_t rng;
	curand_init(seed, index, static_cast<long long>(2 * SPIN_X_WORD) * (2 * number_of_previous_iterations + COLOR), &rng);
	INT2_T target = traders[index];
	traders[index] = flip_spins<BITXSPIN, INT_T, INT2_T>(rng, target, parallel_sum, shared_probabilities);

	return;
}


void precompute_probabilities(float* probabilities, const float market_coupling,
	 														const float reduced_j)
{
		float h_probabilities[14];

		for (int spin = 0; spin < 2; spin++) {
			for (int idx = 0; idx < 7; idx++) {
				int neighbor_sum = 2 * idx - 6;
				float field = reduced_j * neighbor_sum + market_coupling * ((spin) ? 1 : -1);
				h_probabilities[spin * 7 + idx] = 1.0 / (1.0 + exp(field));
			}
		}
		CHECK_CUDA(cudaMemcpy(probabilities, h_probabilities, 2 * 7 * sizeof(*h_probabilities), cudaMemcpyHostToDevice));
		return;
}


// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
template<int BLOCK_DIMENSION_X, int WSIZE, typename T>
__device__ __forceinline__ T __block_sum(T traders)
{
	__shared__ T sh[BLOCK_DIMENSION_X / WSIZE];

	const int lid = threadIdx.x % WSIZE;
	const int wid = threadIdx.x / WSIZE;

	#pragma unroll
	for(int i = WSIZE/2; i; i >>= 1) {
		traders += __shfl_down_sync(0xFFFFFFFF, traders, i);
	}
	if (lid == 0) sh[wid] = traders;

	__syncthreads();
	if (wid == 0) {
		traders = (lid < (BLOCK_DIMENSION_X / WSIZE)) ? sh[lid] : 0;

		#pragma unroll
		for(int i = (BLOCK_DIMENSION_X / WSIZE) / 2; i; i >>= 1) {
			traders += __shfl_down_sync(0xFFFFFFFF, traders, i);
		}
	}
	__syncthreads();
	return traders;
}


// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
template<int BLOCK_DIMENSION_X, int BITXSPIN, typename INT_T, typename SUM_T>
__global__ void getMagn_k(const long long n,
			                    const INT_T *__restrict__ traders,
			                    SUM_T *__restrict__ sum)
{
	// to be optimized
	const int SPIN_X_WORD = (8 * sizeof(INT_T)) / BITXSPIN;

	const long long nth = static_cast<long long>(blockDim.x) * gridDim.x;
	const long long thread_id = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

	SUM_T __cntP = 0;
	SUM_T __cntN = 0;

	for(long long i = 0; i < n; i += nth) {
		if (i + thread_id < n) {
			const int __c = __custom_popc(traders[i + thread_id]);
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


// Copyright (c) 2019, NVIDIA CORPORATION. Mauro Bisson <maurob@nvidia.com>. All rights reserved.
static void countSpins(const int redBlocks,
								       const size_t total_words,
								       const unsigned long long *d_black_tiles,
									     unsigned long long *d_sum,
									     unsigned long long *bsum,
									     unsigned long long *wsum)
{
	CHECK_CUDA(cudaMemset(d_sum, 0, 2 * sizeof(*d_sum)));
	// Only the pointer to the black tiles is needed, since it provides access
	// to all spins (d_spins).
	// see definition in kernel.cu:
	// 		d_black_tiles = d_spins;
	// 		d_white_tiles = d_spins + total_words / 2;
	getMagn_k<THREADS, BIT_X_SPIN><<<redBlocks, THREADS>>>(total_words, d_black_tiles, d_sum);
	CHECK_ERROR("getMagn_k");
	CHECK_CUDA(cudaDeviceSynchronize());

	bsum[0] = 0;
	wsum[0] = 0;

	unsigned long long sum_h[2];

	CHECK_CUDA(cudaMemcpy(sum_h, d_sum, 2 * sizeof(*sum_h), cudaMemcpyDeviceToHost));
	bsum[0] += sum_h[0];
	wsum[0] += sum_h[1];

	return;
}


static void dumpLattice(const long long iteration,
			const int rows,
			const size_t columns,
		        const size_t total_number_of_words,
		        const unsigned long long *v_d) {

	char fname[256];

	unsigned long long *v_h = (unsigned long long *) malloc(total_number_of_words * sizeof(*v_h));
	CHECK_CUDA(cudaMemcpy(v_h, v_d, total_number_of_words * sizeof(*v_h), cudaMemcpyDeviceToHost));

	unsigned long long *black_h = v_h;
	unsigned long long *white_h = v_h + total_number_of_words / 2;

	snprintf(fname, sizeof(fname), "data/lattices/iteration_%lld.dat", iteration);
	FILE *fp = fopen(fname, "w");

	for(int i = 0; i < rows; i++) {
		for(int j = 0; j < columns; j++) {
			unsigned long long __b = black_h[i * columns + j];
			unsigned long long __w = white_h[i * columns + j];

			for(int k = 0; k < 8 * sizeof(*v_h); k += BIT_X_SPIN) {
				if (i & 1) {
					fprintf(fp, "%llX ",  (__w >> k) & 0xF);
					fprintf(fp, "%llX ",  (__b >> k) & 0xF);
				} else {
					fprintf(fp, "%llX ",  (__b >> k) & 0xF);
					fprintf(fp, "%llX ",  (__w >> k) & 0xF);
				}
			}
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	free(v_h);
	return;
}


template<int SPIN_X_WORD>
float update(int iteration,
					   dim3 blocks, dim3 threads_per_block, const int reduce_blocks,
						 unsigned long long *d_black_tiles,
	           unsigned long long *d_white_tiles,
						 unsigned long long *d_sum,
	           float *d_probabilities,
						 unsigned long long spins_up,
						 unsigned long long spins_down,
						 const unsigned long long seed,
	           const float reduced_alpha,
	           const float reduced_j,
	           const long long grid_height, const long long grid_width, const long long grid_depth,
					 	 const size_t words_per_row,
					 	 const size_t total_words)
{
		countSpins(reduce_blocks, total_words, d_black_tiles, d_sum, &spins_up, &spins_down);
		double magnetisation = static_cast<double>(spins_up) - static_cast<double>(spins_down);
		float reduced_magnetisation = magnetisation / static_cast<double>(grid_width * grid_height * grid_depth);
		float market_coupling = -reduced_alpha * abs(reduced_magnetisation);
		precompute_probabilities(d_probabilities, market_coupling, reduced_j);

		CHECK_CUDA(cudaSetDevice(0));
		update_strategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BLOCK_DIMENSION_Z_DEFINE, BIT_X_SPIN, C_BLACK, unsigned long long>
		<<<blocks, threads_per_block>>>
		(seed, iteration + 1, (grid_width / 2) / SPIN_X_WORD / 2, grid_height, grid_depth, words_per_row / 2,
		 d_probabilities,
		 reinterpret_cast<ulonglong2 *>(d_white_tiles),
		 reinterpret_cast<ulonglong2 *>(d_black_tiles));

		CHECK_CUDA(cudaSetDevice(0));
		update_strategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BLOCK_DIMENSION_Z_DEFINE, BIT_X_SPIN, C_WHITE, unsigned long long>
		<<<blocks, threads_per_block>>>
		(seed, iteration + 1, (grid_width / 2) / SPIN_X_WORD / 2, grid_height, grid_depth, words_per_row / 2,
		 d_probabilities,
		 reinterpret_cast<ulonglong2 *>(d_black_tiles),
		 reinterpret_cast<ulonglong2 *>(d_white_tiles));

    return reduced_magnetisation;
}


#endif
