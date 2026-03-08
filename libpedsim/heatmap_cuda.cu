// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality.
//
#include "heatmap_cuda.cuh"

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

	int **h_heatmap, **h_scaled_heatmap, **h_blurred_heatmap;
	h_heatmap = (int **)malloc(SIZE * sizeof(int *));
	h_scaled_heatmap = (int **)malloc(SCALED_SIZE * sizeof(int *));
	h_blurred_heatmap = (int **)malloc(SCALED_SIZE * sizeof(int *));
	for (int i = 0; i < SIZE; i++) {
		h_heatmap[i] = hm + SIZE * i;
	}
	for (int i = 0; i < SCALED_SIZE; i++) {
		h_scaled_heatmap[i] = shm + SCALED_SIZE * i;
		h_blurred_heatmap[i] = bhm + SCALED_SIZE * i;
	}
	cudaMemcpy(&heatmap, h_heatmap, SIZE * sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&scaled_heatmap, h_scaled_heatmap, SCALED_SIZE * sizeof(int *), cudaMemcpyHostToDevice);
	cudaMemcpy(&blurred_heatmap, h_blurred_heatmap, SCALED_SIZE * sizeof(int *), cudaMemcpyHostToDevice);

	free(h_heatmap);
	free(h_scaled_heatmap);
	free(h_blurred_heatmap);

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
	cudaFree(hmcu->heatmap[0]);
	cudaFree(hmcu->scaled_heatmap[0]);
	cudaFree(hmcu->blurred_heatmap[0]);
	cudaFree(hmcu->heatmap);
	cudaFree(hmcu->scaled_heatmap);
	cudaFree(hmcu->blurred_heatmap);
	cudaFree(hmcu->pairs_d);
	cudaFreeHost(hmcu->pairs_h);
}

// Updates the heatmap according to the agent positions
__global__ void update_heatmap(struct hmcu_s *hmcu) {
	extern __shared__ int hm[];
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= SHARED_SIZE || y >= SHARED_SIZE) {
		return;
	}
	for (int y = threadIdx.y; y < SHARED_SIZE; y += blockDim.y) {
		for (int x = threadIdx.x; x < SHARED_SIZE; x += blockDim.x) {
			hm[y + SHARED_SIZE * x] = (int)round(
				hmcu->heatmap[y + blockIdx.y * blockDim.y][x + blockIdx.x * blockDim.x] * 0.80);
		}
	}
	__syncthreads();

	// Count how many agents want to go to each location
	for (int i = threadIdx.x; i < hmcu->size; i += blockDim.x) {
		const int x = hmcu->pairs_d[i].x;
		const int y = hmcu->pairs_d[i].y;

		if (x < blockIdx.x * blockDim.x || x >= SHARED_SIZE + blockIdx.x * blockDim.x || y < blockIdx.y ||
			y >= SHARED_SIZE + blockIdx.y * blockDim.y) {
			continue;
		}

		// intensify heat for better color results
		atomicAdd(&hm[y + SHARED_SIZE * x], 40);
	}

	for (int y = threadIdx.y; y < SHARED_SIZE; y += blockDim.y) {
		for (int x = threadIdx.x; x < SHARED_SIZE; x += blockDim.x) {
			int heat = hm[y + SHARED_SIZE * x] < 255 ? hm[y + SHARED_SIZE * x] : 255;
			hmcu->heatmap[y][x] = heat;
		}
	}
}

__global__ void scale(struct hmcu_s *hmcu) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= SHARED_SIZE || y >= SHARED_SIZE) {
		return;
	}
	// Scale the data for visual representation
	int value = hmcu->heatmap[y][x];
	for (int cellY = 0; cellY < CELLSIZE; cellY++) {
		for (int cellX = 0; cellX < CELLSIZE; cellX++) {
			hmcu->scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
		}
	}
}

__global__ void blur(struct hmcu_s *hmcu) {
	extern __shared__ int shm[];
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	for (int y = threadIdx.y; y < SCALED_SHARED_SIZE; y += blockDim.y) {
		for (int x = threadIdx.x; x < SCALED_SHARED_SIZE; x += blockDim.x) {
			shm[y + SHARED_SIZE * x] = hmcu->scaled_heatmap[y + blockIdx.y * blockDim.y]
														   [x + blockIdx.x * blockDim.x];
		}
	}
	__syncthreads();

#define WEIGHTSUM 273
	// Apply gaussian blurfilter
	if (x < 2 || x >= SCALED_SHARED_SIZE - 2 || y < 2 || y >= SCALED_SHARED_SIZE - 2) {
		return;
	}
	int sum = 0;
	for (int k = -2; k < 3; k++) {
		for (int l = -2; l < 3; l++) {
			sum += w[2 + k][2 + l] * shm[y + k + SHARED_SIZE * (x + l)];
		}
	}
	int value = sum / WEIGHTSUM;
	__syncthreads();
	hmcu->blurred_heatmap[y][x] = 0x00FF0000 | value << 24;
}

__host__ void
hmcu_update_heatmap(dim3 blocks, dim3 threads_per_block, size_t shared_bytes, struct hmcu_s *hmcu) {
	update_heatmap<<<blocks, threads_per_block, shared_bytes>>>(hmcu);
}

__host__ void hmcu_scale(dim3 blocks, dim3 threads_per_block, size_t shared_bytes, struct hmcu_s *hmcu) {
	scale<<<blocks, threads_per_block, shared_bytes>>>(hmcu);
}
__host__ void hmcu_blur(dim3 blocks, dim3 threads_per_block, size_t shared_bytes, struct hmcu_s *hmcu) {
	scale<<<blocks, threads_per_block, shared_bytes>>>(hmcu);
}
