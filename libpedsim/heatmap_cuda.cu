// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality.
//
#include "heatmap_cuda.cuh"
// #include <stdio.h>

__constant__ int w[5][5]; // Weights for blur filter

// Sets up the heatmap
__host__ void hmcu_init(struct hmcu_s *hmcu, int agents_size) {
	// only blurred needs to (also) be on host
	struct pair_s *pairs_h, *pairs_d;
	cudaMallocHost(&pairs_h, sizeof(struct pair_s) * agents_size);
	cudaMalloc(&pairs_d, sizeof(struct pair_s) * agents_size);

	int *hm, *shm, *bhm;
	cudaMalloc(&hm, SIZE * SIZE * sizeof(int));
	cudaMemset(hm, 0, SIZE * SIZE * sizeof(int));
	cudaMalloc(&shm, SCALED_SIZE * SCALED_SIZE * sizeof(int));
	cudaMallocManaged(&bhm, SCALED_SIZE * SCALED_SIZE * sizeof(int));

	int **heatmap, **scaled_heatmap, **blurred_heatmap;
	cudaMalloc(&heatmap, SIZE * sizeof(int *));
	cudaMalloc(&scaled_heatmap, SCALED_SIZE * sizeof(int *));
	cudaMallocManaged(&blurred_heatmap, SCALED_SIZE * sizeof(int *));

	int **h_heatmap, **h_scaled_heatmap;
	h_heatmap = (int **)malloc(SIZE * sizeof(int *));
	h_scaled_heatmap = (int **)malloc(SCALED_SIZE * sizeof(int *));
	for (int i = 0; i < SIZE; i++) {
		h_heatmap[i] = hm + SIZE * i;
	}
	for (int i = 0; i < SCALED_SIZE; i++) {
		h_scaled_heatmap[i] = shm + SCALED_SIZE * i;
		blurred_heatmap[i] = bhm + SCALED_SIZE * i;
	}
	cudaMemcpy(heatmap, h_heatmap, SIZE * sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(scaled_heatmap, h_scaled_heatmap, SCALED_SIZE * sizeof(int *), cudaMemcpyHostToDevice);

	free(h_heatmap);
	free(h_scaled_heatmap);

	const int h_w[5][5] = {
		{1, 4, 7, 4, 1}, {4, 16, 26, 16, 4}, {7, 26, 41, 26, 7}, {4, 16, 26, 16, 4}, {1, 4, 7, 4, 1}};

	hmcu->heatmap = heatmap;
	hmcu->scaled_heatmap = scaled_heatmap;
	hmcu->blurred_heatmap = blurred_heatmap;
	hmcu->pairs_h = pairs_h;
	hmcu->pairs_d = pairs_d;
	hmcu->size = agents_size;
	cudaMemcpyToSymbol(w, h_w, sizeof(w));
}

__host__ void hmcu_dinit(struct hmcu_s *hmcu) {
	(void)hmcu;
	cudaDeviceReset();
}

__global__ void fade_heat(int **heatmap) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= SIZE || y >= SIZE) {
		return;
	}
	heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
}

__global__ void insert_heat(int **heatmap, struct pair_s *pairs, int agents_size) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= agents_size) {
		return;
	}
	const int x = pairs[idx].x;
	const int y = pairs[idx].y;
	if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) {
		return;
	}
	atomicAdd(&heatmap[y][x], 40);
}

__global__ void cap_scale_heat(int **heatmap, int **scaled_heatmap) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= SIZE || y >= SIZE) {
		return;
	}
	const int value = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
	heatmap[y][x] = value;
	// Scale the data for visual representation
	for (int cellY = 0; cellY < CELLSIZE; cellY++) {
		for (int cellX = 0; cellX < CELLSIZE; cellX++) {
			scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
		}
	}
}

#define HALO 2
#define SHM_X (THREADS_X + 2 * HALO)
#define SHM_Y (THREADS_Y + 2 * HALO)

// hack!
// spawn more threads to load global mem into shared easily
// excess workers do nothing
__global__ void blur_heat(int **scaled_heatmap, int **blurred_heatmap) {
	__shared__ int shm[SHM_Y][SHM_X];
	// DO NOT use blockDim.[xy] to avoid gaps in grid
	const int tx = threadIdx.x, x = threadIdx.x - HALO + blockIdx.x * THREADS_X;
	const int ty = threadIdx.y, y = threadIdx.y - HALO + blockIdx.y * THREADS_Y;
	// if (x == 0 && y == 0) {
	// 	printf("BLURRING HEATMAP!\n");
	// }
#define WEIGHTSUM 273
	// Apply gaussian blurfilter
	if (x >= 0 && x < SCALED_SIZE && y >= 0 && y < SCALED_SIZE) {
		shm[ty][tx] = scaled_heatmap[y][x];
	}
	__syncthreads();

	// first we need to check if we are an 'excess' worker
	if (tx >= HALO && tx < THREADS_X + HALO && ty >= HALO && ty < THREADS_Y + HALO) {
		if (x >= 2 && x < SCALED_SIZE - 2 && y >= 2 && y < SCALED_SIZE - 2) {
			int sum = 0;
			for (int k = -2; k < 3; k++) {
				for (int l = -2; l < 3; l++) {
					sum += w[2 + k][2 + l] * shm[ty + k][tx + l];
				}
			}
			int value = sum / WEIGHTSUM;
			blurred_heatmap[y][x] = 0x00FF0000 | value << 24;
		}
	}
}

__host__ void hmcu_update_heatmap(struct hmcu_s *hmcu) {
	static dim3 threads_per_block(THREADS_X, THREADS_Y, 1);
	static dim3 size_blocks(((SIZE + threads_per_block.x - 1) / threads_per_block.x),
							((SIZE + threads_per_block.y - 1) / threads_per_block.y),
							1);
	fade_heat<<<size_blocks, threads_per_block>>>(hmcu->heatmap);

	cudaMemcpy(hmcu->pairs_d, hmcu->pairs_h, hmcu->size * sizeof(struct pair_s), cudaMemcpyHostToDevice);
	static dim3 pairs_threads_per_block(PAIRS_THREADS, 1, 1);
	static dim3 pairs_blocks(
		((hmcu->size + pairs_threads_per_block.x - 1) / pairs_threads_per_block.x), 1, 1);
	insert_heat<<<pairs_blocks, pairs_threads_per_block>>>(hmcu->heatmap, hmcu->pairs_d, hmcu->size);

	cap_scale_heat<<<size_blocks, threads_per_block>>>(hmcu->heatmap, hmcu->scaled_heatmap);

	// hack!
	// spawn more threads to load global mem into shared easily
	// excess workers do nothing
	static dim3 halo_threads_per_block(SHM_X, SHM_Y, 1);
	static dim3 scaled_blocks(((SCALED_SIZE + threads_per_block.x - 1) / threads_per_block.x),
							  ((SCALED_SIZE + threads_per_block.y - 1) / threads_per_block.y),
							  1);
	blur_heat<<<scaled_blocks, halo_threads_per_block>>>(hmcu->scaled_heatmap, hmcu->blurred_heatmap);
}
