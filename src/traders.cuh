#ifndef traders_cuh
#define traders_cuh

#include <curand_kernel.h>
#include "cudamacro.h"

#define BIT_X_SPIN (4)
#define THREADS 128

#define BLOCK_DIMENSION_X_DEFINE (16)
#define BLOCK_DIMENSION_Y_DEFINE (16)

enum {C_BLACK, C_WHITE};


__device__ __forceinline__ unsigned int __custom_popc(const unsigned int x) {return __popc(x);}
__device__ __forceinline__ unsigned long long int __custom_popc(const unsigned long long int x) {return __popcll(x);}

// creates a vector with two components
__device__ __forceinline__ uint2 __custom_make_int2(const unsigned int x, const unsigned int y) {return make_uint2(x, y);}
__device__ __forceinline__ ulonglong2 __custom_make_int2(const unsigned long long x, const unsigned long long y) {return make_ulonglong2(x, y);}


template<int BLOCK_DIMENSION_X, int BLOCK_DIMENSION_Y, int BITXSPIN, int COLOR, typename INT_T, typename INT2_T>
__global__  void initialise_traders(const unsigned long long seed, const long long number_of_columns, INT2_T *__restrict__ traders,
																		float percentage = 0.5f)
{
	const int row = blockIdx.y * BLOCK_DIMENSION_Y + threadIdx.y;
	const int col = blockIdx.x * BLOCK_DIMENSION_X + threadIdx.x;
  const int index = row * number_of_columns + col;
	const int SPIN_X_WORD = 8 * sizeof(INT_T) / BITXSPIN;

	const long long thread_id = ((gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * BLOCK_DIMENSION_X * BLOCK_DIMENSION_Y +
	                              threadIdx.y * BLOCK_DIMENSION_X + threadIdx.x;

	curandStatePhilox4_32_10_t rng;
	curand_init(seed, thread_id, static_cast<long long>(2 * SPIN_X_WORD) * COLOR, &rng);

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
								 const unsigned long long seed, const unsigned long long number_of_columns,
								 INT2_T *__restrict__ d_black_tiles, INT2_T *__restrict__ d_white_tiles,
								 float percentage = 0.5f)
{
		CHECK_CUDA(cudaSetDevice(0));
		initialise_traders<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_BLACK, INT_T>
		<<<blocks, threads_per_block>>>
		(seed, number_of_columns, reinterpret_cast<ulonglong2 *>(d_black_tiles), percentage);
		CHECK_ERROR("initialise_traders");

		initialise_traders<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_WHITE, INT_T>
		<<<blocks, threads_per_block>>>
		(seed, number_of_columns, reinterpret_cast<ulonglong2 *>(d_white_tiles), percentage);
		CHECK_ERROR("initialise_traders");
}


template<int TILE_SIZE_X, int TILE_SIZE_Y, typename INT2_T>
__device__ void load_tiles(const int grid_width, const int grid_height, const long long number_of_columns,
                           const INT2_T *__restrict__ traders, INT2_T tile[][TILE_SIZE_X + 2])
    /*
    Each threads_per_block works on one tile with shape (TILE_SIZE_Y + 2, TILE_SIZE_X + 2).
    */
{
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	const int tile_start_x = blockIdx.x * TILE_SIZE_X;
	const int tile_start_y = blockIdx.y * TILE_SIZE_Y;

	int row = tile_start_y + tidy;
	int col = tile_start_x + tidx;
	tile[1 + tidy][1 + tidx] = traders[row * number_of_columns + col];

	if (tidy == 0) {
		row = (tile_start_y % grid_height) == 0 ? tile_start_y + grid_height - 1 : tile_start_y - 1;
		tile[0][1 + tidx] = traders[row * number_of_columns + col];

		row = ((tile_start_y + TILE_SIZE_Y) % grid_height) == 0 ? tile_start_y + TILE_SIZE_Y - grid_height : tile_start_y + TILE_SIZE_Y;
		tile[1 + TILE_SIZE_Y][1 + tidx] = traders[row * number_of_columns + col];

		row = tile_start_y + tidx;
		col = (tile_start_x % grid_width) == 0 ? tile_start_x + grid_width - 1 : tile_start_x - 1;
		tile[1 + tidx][0] = traders[row * number_of_columns + col];

		row = tile_start_y + tidx;
		col = ((tile_start_x + TILE_SIZE_X) % grid_width) == 0 ? tile_start_x + TILE_SIZE_X - grid_width : tile_start_x + TILE_SIZE_X;
		tile[1 + tidx][1 + TILE_SIZE_X] = traders[row * number_of_columns + col];
	}
	return;
}


__device__ void load_probabilities(const float precomputed_probabilities[][7], float shared_probabilities[2][7],
                                   const int block_dimension_x, const int block_dimension_y,
                                   const int tidx, const int tidy)
{
  // load precomputed exponentials into shared memory.
  // in case a threads_per_block consists of less than 2 * 5 threads
  // multiple iterations in each thread are needed
  // otherwise loops will only trigger once
  #pragma unroll
  for(int i = 0; i < 2; i += block_dimension_x) {
    #pragma unroll
    for(int j = 0; j < 5; j += block_dimension_y) {
      if (i + tidy < 2 && j + tidx < 5) {
        shared_probabilities[i + tidy][j + tidx] = precomputed_probabilities[i + tidy][j + tidx];
      }
    }
  }
  return;
}


template<int BLOCK_DIMENSION_X, int BLOCK_DIMENSION_Y, int BITXSPIN, int COLOR, typename INT_T, typename INT2_T>
__global__ void update_strategies(const unsigned long long seed, const int number_of_previous_iterations,
		       const int grid_width, // lattice width of one color in words
		       const int grid_height, // lattice height (not in words)
		       const long long number_of_columns,
		       const float precomputed_probabilities[][7],
		       const INT2_T *__restrict__ checkerboard_agents,
		             INT2_T *__restrict__ traders)
{
	const int SPIN_X_WORD = 8 * sizeof(INT_T) / BITXSPIN;
	const INT_T ONE = static_cast<INT_T>(1);
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;

	__shared__ INT2_T shared_tiles[BLOCK_DIMENSION_Y + 2][BLOCK_DIMENSION_X + 2];
	load_tiles<BLOCK_DIMENSION_X, BLOCK_DIMENSION_Y, INT2_T>
  (grid_width, grid_height, number_of_columns, checkerboard_agents, shared_tiles);

	__shared__ float shared_probabilities[2][7];
  load_probabilities(precomputed_probabilities, shared_probabilities, BLOCK_DIMENSION_X, BLOCK_DIMENSION_Y, tidx, tidy);

	__syncthreads();


	const int row = blockIdx.y * BLOCK_DIMENSION_Y + tidy;
	const int col = blockIdx.x * BLOCK_DIMENSION_X + tidx;

	const long long thread_id = (blockIdx.y * gridDim.x + blockIdx.x) * BLOCK_DIMENSION_X * BLOCK_DIMENSION_Y
                            +  threadIdx.y * BLOCK_DIMENSION_X + threadIdx.x;

	INT2_T target = traders[row * number_of_columns + col];

	// three nearest neighbors
	INT2_T upper_neighbor = shared_tiles[    tidy][1 + tidx];
	INT2_T center_neighbor = shared_tiles[1 + tidy][1 + tidx];
	INT2_T lower_neighbor = shared_tiles[2 + tidy][1 + tidx];

	const int shift_left = (COLOR == C_BLACK) ? !(row % 2) : (row % 2);
	// remaining neighbor, either left or right
	INT2_T horizontal_neighbor = (shift_left) ? shared_tiles[1 + tidy][tidx] : shared_tiles[1 + tidy][2 + tidx];

	if (shift_left) {
  	horizontal_neighbor.x = (center_neighbor.x << BITXSPIN) | (horizontal_neighbor.y >> (8 * sizeof(horizontal_neighbor.y) - BITXSPIN));
  	horizontal_neighbor.y = (center_neighbor.y << BITXSPIN) | (center_neighbor.x >> (8 * sizeof(center_neighbor.x) - BITXSPIN));
	} else {
		horizontal_neighbor.y = (center_neighbor.y >> BITXSPIN) | (horizontal_neighbor.x << (8 * sizeof(horizontal_neighbor.x) - BITXSPIN));
		horizontal_neighbor.x = (center_neighbor.x >> BITXSPIN) | (center_neighbor.y << (8 * sizeof(center_neighbor.y) - BITXSPIN));
	}

	curandStatePhilox4_32_10_t rng;
	curand_init(seed, thread_id, static_cast<long long>(2 * SPIN_X_WORD) * (2 * number_of_previous_iterations + COLOR), &rng);

	// this basically sums over all spins/word in parallel
	center_neighbor.x += upper_neighbor.x + lower_neighbor.x + horizontal_neighbor.x;
	center_neighbor.y += upper_neighbor.y + lower_neighbor.y + horizontal_neighbor.y;

	for(int spin_position = 0; spin_position < 8 * sizeof(INT_T); spin_position += BITXSPIN) {

		const int2 source = make_int2((target.x >> spin_position) & 0xF, (target.y >> spin_position) & 0xF);
		const int2 sum = make_int2((center_neighbor.x >> spin_position) & 0xF, (center_neighbor.y >> spin_position) & 0xF);

		if (curand_uniform(&rng) <= shared_probabilities[source.x][sum.x]) {
			target.x |= ONE << spin_position;
		}
		if (curand_uniform(&rng) <= shared_probabilities[source.y][sum.y]) {
			target.y |= ONE << spin_position;
		}
	}

	traders[row * number_of_columns + col] = target;

	return;
}


void precompute_probabilities(float* probabilities, const float market_coupling, const float reduced_j) {
		float h_probabilities[2][7];

		for (int spin = 0; spin < 2; spin++) {
			for (int neighbor_sum = 0; neighbor_sum < 7; neighbor_sum++) {
				double field = reduced_j * neighbor_sum + market_coupling * ((spin) ? 1 : -1);
				h_probabilities[spin][neighbor_sum] = 1 / (1 + exp(field));
			}
		}
		CHECK_CUDA(cudaMemcpy(probabilities, h_probabilities, 2 * 7 * sizeof(**h_probabilities), cudaMemcpyHostToDevice));
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

	const long long nth = static_cast<long long>(blockDim.x) * gridDim.x;
	const long long thread_id = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

	SUM_T __cntP = 0;
	SUM_T __cntN = 0;

	for(long long i = 0; i < n; i += nth) {
		if (i+thread_id < n) {
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


template<int SPIN_X_WORD>
int update(int iteration,
				   dim3 blocks, dim3 threads_per_block, const int reduce_blocks,
					 unsigned long long *d_black_tiles,
           unsigned long long *d_white_tiles,
					 unsigned long long **sum_d,
           float* d_probabilities,
					 unsigned long long spins_up,
					 unsigned long long spins_down,
					 const unsigned long long seed,
           const float reduced_alpha,
           const float reduced_j,
           const long long grid_height, const long long grid_width, const long long grid_depth,
				 	 const size_t words_per_row,
				 	 const size_t total_words)
{
		countSpins(reduce_blocks, total_words, d_black_tiles, d_white_tiles, sum_d, &spins_up, &spins_down);
		int magnetisation = spins_up - spins_down;
		float reduced_magnetisation = abs(magnetisation / (grid_width * grid_height * grid_depth));
		float market_coupling = -reduced_alpha * reduced_magnetisation;
		precompute_probabilities(d_probabilities, market_coupling, reduced_j);

		CHECK_CUDA(cudaSetDevice(0));
		update_strategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_BLACK, unsigned long long>
		<<<blocks, threads_per_block>>>
		(seed, iteration + 1, (grid_width / 2) / SPIN_X_WORD / 2, grid_height, words_per_row / 2,
		 reinterpret_cast<float (*)[7]>(d_probabilities),
		 reinterpret_cast<ulonglong2 *>(d_white_tiles),
		 reinterpret_cast<ulonglong2 *>(d_black_tiles));

		CHECK_CUDA(cudaSetDevice(0));
		update_strategies<BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BIT_X_SPIN, C_WHITE, unsigned long long>
		<<<blocks, threads_per_block>>>
		(seed, iteration + 1, (grid_width / 2) / SPIN_X_WORD / 2, grid_height, words_per_row / 2,
		 reinterpret_cast<float (*)[7]>(d_probabilities),
		 reinterpret_cast<ulonglong2 *>(d_black_tiles),
		 reinterpret_cast<ulonglong2 *>(d_white_tiles));

     return magnetisation;
}


#endif
