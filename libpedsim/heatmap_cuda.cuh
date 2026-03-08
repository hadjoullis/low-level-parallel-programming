#ifndef _heatmap_cuda_h_
#define _heatmap_cuda_h_

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE (SIZE * CELLSIZE)

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
__host__ void hmcu_update_heatmap(struct hmcu_s *hmcu);

#endif
