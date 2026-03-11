#ifndef _heatmap_cuda_h_
#define _heatmap_cuda_h_

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE (SIZE * CELLSIZE)

#define THREADS_X 16
#define THREADS_Y 16
#define PAIRS_THREADS 512

#include <cuda_runtime.h>
#include <stdlib.h>

struct pairs_s {
	int *x, *y;
};

struct hmcu_s {
	int **heatmap, **scaled_heatmap, **blurred_heatmap;
	struct pairs_s pairs_h, pairs_d;
	int size;
};

struct hmcu_time_s {
	cudaEvent_t sfade, sinsert, scap_scale, eblur;
	cudaEvent_t efade, einsert, ecap_scale, sblur;
	float fade, insert, cap_scale, blur;
};

__host__ void hmcu_init(struct hmcu_s *hmcu, int agents_size, struct hmcu_time_s *time);
__host__ void hmcu_dinit(struct hmcu_s *hmcu, struct hmcu_time_s *time);
__host__ void hmcu_update_heatmap(struct hmcu_s *hmcu, struct hmcu_time_s *time);
__host__ void
hmcu_update_heatmap_bn(struct hmcu_s *hmcu, int size, int *des_x, int *des_y, struct hmcu_time_s *time);

#endif
