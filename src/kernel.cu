#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>

#include "cudamacro.h"
#include "traders.cuh"

using namespace std;

#define DIV_UP(a,b)  (((a) + ((b) - 1)) / (b))
#define MIN(a,b)	(((a) < (b)) ? (a) : (b))
#define MAX(a,b)	(((a) > (b)) ? (a) : (b))

#define THREADS 128
#define BIT_X_SPIN (4)

#define BLOCK_DIMENSION_X_DEFINE (8)
#define BLOCK_DIMENSION_Y_DEFINE (8)
#define BLOCK_DIMENSION_Z_DEFINE (8)

map<string, string> read_config_file(string config_filename, string delimiter = "=")
{
    std::ifstream config_file;
    config_file.open(config_filename);
    map<string, string> config;

    if (!config_file.is_open()) {
        std::cout << "Could not open file '" << config_filename << "'" << std::endl;
        return config;
    } else {
        int row = 0;
        std::string line = "";
        std::string key = "";

        std::cout << "Launch configuration:" << std::endl;

        while (getline(config_file, line)) {
            if (line[0] == '#' || line == "") continue;
            int delimiter_position = line.find(delimiter);

            for (int idx = 0; idx < delimiter_position; idx++) {
                if (line[idx] != ' ') key += line[idx];
            }

            std::string value = line.substr(delimiter_position + 1, line.length() - 1);
            config[key] = value;
            std::cout << '\t' << key << ": ";
            std::cout << value << std::endl;
            row++;
            key = "";
        }
        config_file.close();
        return config;
    }
}


void validate_grid(const long long grid_width, const long long grid_height, const long long grid_depth,
                   const int spin_x_word)
{
	if (!grid_width || (grid_width % 2) || ((grid_width / 2) % (2 * spin_x_word * BLOCK_DIMENSION_X_DEFINE))) {
		fprintf(stderr, "\nPlease specify an grid_width multiple of %d\n\n", 2 * spin_x_word * 2 * BLOCK_DIMENSION_X_DEFINE);
		exit(EXIT_FAILURE);
	}
	if (!grid_height || (grid_height % (BLOCK_DIMENSION_Y_DEFINE))) {
		fprintf(stderr, "\nPlease specify a grid_height multiple of %d\n\n", BLOCK_DIMENSION_Y_DEFINE);
		exit(EXIT_FAILURE);
	}
  if (!grid_depth || (grid_depth % (BLOCK_DIMENSION_Z_DEFINE))) {
    fprintf(stderr, "\nPlease specify a grid_depth multiple of %d\n\n", BLOCK_DIMENSION_Z_DEFINE);
    exit(EXIT_FAILURE);
  }
}


cudaDeviceProp identify_gpu()
{
  cudaDeviceProp props;
  CHECK_CUDA(cudaGetDeviceProperties(&props, 0));
  printf("\nUsing GPU: %s, %d SMs, %d th/SM max, CC %d.%d, ECC %s\n",
    props.name, props.multiProcessorCount,
    props.maxThreadsPerMultiProcessor,
    props.major, props.minor,
    props.ECCEnabled ? "on" : "off");
  return props;
}


int main(int argc, char **argv) {

	unsigned long long *d_spins = NULL;
	unsigned long long *d_black_tiles = NULL;
	unsigned long long *d_white_tiles = NULL;

	const int SPIN_X_WORD = (8 * sizeof(*d_spins)) / BIT_X_SPIN;

	cudaEvent_t start, stop;
  float elapsed_time;

	string config_filename = (argc == 1) ? "multising.conf" : argv[1];
  map<string, string> config = read_config_file(config_filename);

  const long long grid_height = std::stoll(config["grid_height"]);
  const long long grid_width = std::stoll(config["grid_width"]);
  const long long grid_depth = std::stoll(config["grid_depth"]);
  const unsigned int total_updates = std::stoul(config["total_updates"]);
  const unsigned long long seed = std::stoull(config["seed"]);
  float alpha = std::stof(config["alpha"]);
  float j = std::stof(config["j"]);
  float beta = std::stof(config["beta"]);
  float percentage_up = std::stof(config["init_up"]);

  const float reduced_alpha = -2.0f * beta * alpha;
  const float reduced_j = -2.0f * beta * j;

	validate_grid(grid_width, grid_height, grid_depth, SPIN_X_WORD);

  cudaDeviceProp props = identify_gpu();

	const size_t words_per_row = (grid_width / 2) / SPIN_X_WORD;
	const size_t total_words = 2ull * static_cast<size_t>(grid_height) * words_per_row * static_cast<size_t>(grid_depth);
	// words_per_row / 2 because each entry in the array has two components
	dim3 blocks(DIV_UP(words_per_row / 2, BLOCK_DIMENSION_X_DEFINE), DIV_UP(grid_height, BLOCK_DIMENSION_Y_DEFINE), DIV_UP(grid_depth, BLOCK_DIMENSION_Z_DEFINE));
	dim3 threads_per_block(BLOCK_DIMENSION_X_DEFINE, BLOCK_DIMENSION_Y_DEFINE, BLOCK_DIMENSION_Z_DEFINE);
	const int reduce_blocks = MIN(DIV_UP(total_words, THREADS), (props.maxThreadsPerMultiProcessor / THREADS) * props.multiProcessorCount);

	unsigned long long spins_up;
	unsigned long long spins_down;
	unsigned long long *sum_d[0];

	CHECK_CUDA(cudaMalloc(&d_spins, total_words * sizeof(*d_spins)));
	CHECK_CUDA(cudaMemset(d_spins, 0, total_words * sizeof(*d_spins)));

	CHECK_CUDA(cudaMalloc(&sum_d[0], 2 * sizeof(**sum_d)));

	d_black_tiles = d_spins;
	d_white_tiles = d_spins + total_words / 2;

	float *d_probabilities;
	CHECK_CUDA(cudaMalloc(&d_probabilities, 2 * 7 * sizeof(*d_probabilities)));

	CHECK_CUDA(cudaEventCreate(&start));
	CHECK_CUDA(cudaEventCreate(&stop));

	initialise_arrays<unsigned long long>(blocks, threads_per_block, seed, words_per_row / 2, words_per_row / 2 * grid_height,
                                        d_black_tiles, d_white_tiles, percentage_up);

	CHECK_CUDA(cudaSetDevice(0));
	CHECK_CUDA(cudaDeviceSynchronize());

	CHECK_CUDA(cudaEventRecord(start, 0));
  int iteration;
  int global_market;
	std::ofstream file;
	file.open("data/magnetisation.dat");
	for(iteration = 0; iteration < total_updates; iteration++) {
		global_market = update<SPIN_X_WORD>(iteration, blocks, threads_per_block, reduce_blocks,
					 				      	d_black_tiles, d_white_tiles, sum_d, d_probabilities,
					 								spins_up, spins_down,
					 						  	seed, reduced_alpha, reduced_j,
	         								grid_height, grid_width, grid_depth,
					 						  	words_per_row, total_words);

	  if (iteration % 10 == 0)
		  file << global_market << ' ' << std::flush;

	}
	file.close();

	CHECK_CUDA(cudaEventRecord(stop, 0));
	CHECK_CUDA(cudaEventSynchronize(stop));

	CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));

	printf("Kernel execution time: %.2f s, %.2lf flips/ns \n",
		elapsed_time * 1.0E-3, static_cast<double>(total_words * SPIN_X_WORD) * iteration / (elapsed_time * 1.0E+6));
  printf("Final magnetisation: %d\n", global_market);

	CHECK_CUDA(cudaFree(d_spins));
	CHECK_CUDA(cudaFree(d_probabilities));
	CHECK_CUDA(cudaFree(sum_d[0]));

  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaDeviceReset());

	return 0;
}
