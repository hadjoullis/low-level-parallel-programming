#ifndef _heatmap_cuda_h_
#define _heatmap_cuda_h_

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE (SIZE * CELLSIZE)

#define SHARED_SIZE (128)		 // 16 KB, each int is 4 bytes
#define SCALED_SHARED_SIZE (256) // 64 KB, each int is 4 bytes

#include "ped_move_parallel.h"

#include <cuda_runtime.h>
#include <stdlib.h>

struct hmcu_s {
	int **heatmap, **scaled_heatmap, **blurred_heatmap;
	struct pair_s *pairs_h, *pairs_d;
	int size;
};

__host__ void hmcu_init(struct hmcu_s *hmcu, int agents_size);
__host__ void hmcu_dinit(struct hmcu_s *hmcu);
__host__ void
hmcu_update_heatmap(dim3 blocks, dim3 threads_per_block, size_t shared_bytes, struct hmcu_s *hmcu);
__host__ void hmcu_scale(dim3 blocks, dim3 threads_per_block, size_t shared_bytes, struct hmcu_s *hmcu);
__host__ void hmcu_blur(dim3 blocks, dim3 threads_per_block, size_t shared_bytes, struct hmcu_s *hmcu);

#endif
