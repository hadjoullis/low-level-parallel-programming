//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_simd_agents.h"

void simd_init(std::vector<Ped::Tagent *> agents, struct agents *agents_s) {
	const size_t agents_size = agents.size();
	const size_t int_bytes = sizeof(int) * agents_size;
	const size_t size_t_bytes = sizeof(size_t) * agents_size;
	const size_t ssize_t_bytes = sizeof(ssize_t) * agents_size;
	const size_t ptr_bytes = sizeof(double *) * agents_size;
	posix_memalign((void **)&agents_s->x, ALIGN, int_bytes);
	posix_memalign((void **)&agents_s->y, ALIGN, int_bytes);
	posix_memalign((void **)&agents_s->destination_idx, ALIGN, ssize_t_bytes);
	posix_memalign((void **)&agents_s->waypoints.x, ALIGN, ptr_bytes);
	posix_memalign((void **)&agents_s->waypoints.y, ALIGN, ptr_bytes);
	posix_memalign((void **)&agents_s->waypoints.r, ALIGN, ptr_bytes);
	for (size_t i = 0; i < agents_size; i++) {
		// allocate one extra element so when we index with offset '-1' we do
		// not go out of bounds ('simd_computeNextDesiredPosition')
		const size_t bytes = sizeof(double) * (agents[i]->getWaypointsSize() + 1);
		posix_memalign((void **)&agents_s->waypoints.x[i], ALIGN, bytes);
		posix_memalign((void **)&agents_s->waypoints.y[i], ALIGN, bytes);
		posix_memalign((void **)&agents_s->waypoints.r[i], ALIGN, bytes);

		agents_s->waypoints.x[i]++;
		agents_s->waypoints.y[i]++;
		agents_s->waypoints.r[i]++;
	}
	posix_memalign((void **)&agents_s->waypoints.sz, ALIGN, size_t_bytes);

	agents_s->size = agents_size;
	for (size_t i = 0; i < agents_size; i++) {
		agents_s->x[i] = agents[i]->getX();
		agents_s->y[i] = agents[i]->getY();
		agents_s->destination_idx[i] = -1;
		// we need to ensure that if we index with '-1' the condition 'len <
		// dst_r' evaluates to true so we satisfy both clauses of the second
		// if statement
		agents_s->waypoints.x[i][-1] = 0;
		agents_s->waypoints.y[i][-1] = 0;
		agents_s->waypoints.r[i][-1] = DBL_MAX;
		// agents_s->waypoints: already set
		agents_s->waypoints.sz[i] = agents[i]->getWaypointsSize();
		for (size_t j = 0; j < agents_s->waypoints.sz[i]; j++) {
			auto wp = agents[i]->getWaypoint(j);
			agents_s->waypoints.x[i][j] = wp->getx();
			agents_s->waypoints.y[i][j] = wp->gety();
			agents_s->waypoints.r[i][j] = wp->getr();
		}
	}
}

void simd_dinit(struct agents *agents_s) {
	free(agents_s->x);
	free(agents_s->y);
	free(agents_s->destination_idx);
	for (size_t i = 0; i < agents_s->size; i++) {
		agents_s->waypoints.x[i]--;
		agents_s->waypoints.y[i]--;
		agents_s->waypoints.r[i]--;
		free(agents_s->waypoints.x[i]);
		free(agents_s->waypoints.y[i]);
		free(agents_s->waypoints.r[i]);
	}
	free(agents_s->waypoints.x);
	free(agents_s->waypoints.y);
	free(agents_s->waypoints.r);
	free(agents_s->waypoints.sz);
}

static ssize_t single_get_nextDestination_idx(double **wps_x,
											  double **wps_y,
											  double **wps_r,
											  size_t *wps_sz,
											  const int agent_x,
											  const int agent_y,
											  const ssize_t dst_idx,
											  const size_t agent_idx) {
	ssize_t nextDestination_idx = -1;
	bool agentReachedDestination = false;

	if (dst_idx != -1) {
		// compute if agent reached its current destination
		const double diffX = wps_x[agent_idx][dst_idx] - agent_x;
		const double diffY = wps_y[agent_idx][dst_idx] - agent_y;
		const double len = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = len < wps_r[agent_idx][dst_idx];
	}

	if ((agentReachedDestination || dst_idx == -1) && wps_sz[agent_idx] != 0) {
		// Case 1: agent has reached destination (or has no current
		// destination); get next destination if available
		ssize_t wps_idx = dst_idx + 1;
		if (wps_idx == wps_sz[agent_idx]) {
			wps_idx = -1;
		}
		nextDestination_idx = wps_idx;
	} else {
		// Case 2: agent has not yet reached destination, continue to move
		// towards current destination
		nextDestination_idx = dst_idx;
	}
	return nextDestination_idx;
}

void single_computeNextDesiredPosition(const struct agents *agents, const size_t agent_idx) {
	const int agent_x = agents->x[agent_idx];
	const int agent_y = agents->y[agent_idx];
	ssize_t dst_idx = agents->destination_idx[agent_idx];

	ssize_t nextDestination_idx = single_get_nextDestination_idx(agents->waypoints.x,
																 agents->waypoints.y,
																 agents->waypoints.r,
																 agents->waypoints.sz,
																 agent_x,
																 agent_y,
																 dst_idx,
																 agent_idx);
	agents->destination_idx[agent_idx] = nextDestination_idx;
	dst_idx = nextDestination_idx;
	if (dst_idx == -1) {
		return; // no destination, no need to compute where to move to
	}

	const double diffX = agents->waypoints.x[agent_idx][dst_idx] - agent_x;
	const double diffY = agents->waypoints.y[agent_idx][dst_idx] - agent_y;
	const double len = sqrt(diffX * diffX + diffY * diffY);
	agents->x[agent_idx] = (int)round(agent_x + diffX / len);
	agents->y[agent_idx] = (int)round(agent_y + diffY / len);
}

static inline __m512d fetch_dsts(ssize_t *destination_idx, double **coord, const size_t agent_idx) {
	static double dsts[STEPS] __attribute__((aligned(ALIGN)));
	for (size_t i = 0; i < STEPS; i++) {
		const ssize_t dst_idx = destination_idx[agent_idx + i];
		dsts[i] = coord[agent_idx + i][dst_idx];
	}
	return _mm512_load_pd(&dsts);
}

static __m512i simd_get_nextDestination_idx(double **wps_x,
											double **wps_y,
											double **wps_r,
											size_t *wps_sz,
											const __m256i agent_x,
											const __m256i agent_y,
											const __m512d agent_xd,
											const __m512d agent_yd,
											ssize_t *destination_idx,
											const size_t agent_idx) {

	// ·since·this·is·simd,·we·have·to·uncoditionally·execute·both·branching
	// ·paths,·even·if·'dst_idx'·of·an·agent·is·'-1'.·Make·sure·this·is
	// ·taken·care·of·in·initial·allocations.
	const __m512d dst_x = fetch_dsts(destination_idx, wps_x, agent_idx);
	const __m512d diffX = _mm512_sub_pd(dst_x, agent_xd);

	const __m512d dst_y = fetch_dsts(destination_idx, wps_y, agent_idx);
	const __m512d diffY = _mm512_sub_pd(dst_y, agent_yd);

	const __m512d diffX_sqr = _mm512_mul_pd(diffX, diffX);
	const __m512d diffY_sqr = _mm512_mul_pd(diffY, diffY);

	const __m512d len = _mm512_sqrt_pd(_mm512_add_pd(diffX_sqr, diffY_sqr));
	const __m512d dst_r = fetch_dsts(destination_idx, wps_r, agent_idx);
	const __mmask8 agentReachedDestination = _mm512_cmplt_pd_mask(len, dst_r);

	const __m512i sz = _mm512_load_epi64(&wps_sz[agent_idx]);
	const __m512i zero = _mm512_setzero_si512();
	const __mmask8 non_empty_sz = _mm512_cmpneq_epi64_mask(sz, zero);

	const __m512i one = _mm512_set1_epi64(1);
	__m512i wps_idx = _mm512_load_epi64(&destination_idx[agent_idx]);
	wps_idx = _mm512_add_epi64(wps_idx, one);
	const __mmask8 overflow = _mm512_cmpeq_epi64_mask(wps_idx, sz);
	const __m512i minus_one = _mm512_set1_epi64(-1);

	// Case 1: agent has reached destination (or has no current destination);
	// get next destination if available
	const __m512i nextDestination_idx_if = _mm512_mask_blend_epi64(overflow, wps_idx, minus_one);
	// Case 2: agent has not yet reached destination, continue to move towards
	// current destination
	const __m512i nextDestination_idx_else = _mm512_load_epi64(&destination_idx[agent_idx]);

	const __m512i nextDestination_idx = _mm512_mask_blend_epi64(
		_kand_mask8(agentReachedDestination, non_empty_sz), nextDestination_idx_else, nextDestination_idx_if);
	return nextDestination_idx;
}

void simd_computeNextDesiredPosition(const struct agents *agents, const size_t agent_idx) {
	const __m256i agent_x = _mm256_load_epi32(&agents->x[agent_idx]);
	const __m256i agent_y = _mm256_load_epi32(&agents->y[agent_idx]);
	const __m512d agent_xd = _mm512_cvtepi32_pd(agent_x);
	const __m512d agent_yd = _mm512_cvtepi32_pd(agent_y);

	const __m512i nextDestination_idx = simd_get_nextDestination_idx(agents->waypoints.x,
																	 agents->waypoints.y,
																	 agents->waypoints.r,
																	 agents->waypoints.sz,
																	 agent_x,
																	 agent_y,
																	 agent_xd,
																	 agent_yd,
																	 agents->destination_idx,
																	 agent_idx);
	_mm512_store_epi64(&agents->destination_idx[agent_idx], nextDestination_idx);

	// no destination, no need to compute where to move to
	const __m512i dst_idx = nextDestination_idx;
	// copy from src when bit is NOT set
	const __m512i minus_one = _mm512_set1_epi64(-1);
	const __mmask8 has_dst = _mm512_cmpneq_epi64_mask(dst_idx, minus_one);

	const __m512d dst_x = fetch_dsts(agents->destination_idx, agents->waypoints.x, agent_idx);
	const __m512d diffX = _mm512_sub_pd(dst_x, agent_xd);
	const __m512d dst_y = fetch_dsts(agents->destination_idx, agents->waypoints.y, agent_idx);
	const __m512d diffY = _mm512_sub_pd(dst_y, agent_yd);
	const __m512d diffX_sqr = _mm512_mul_pd(diffX, diffX);
	const __m512d diffY_sqr = _mm512_mul_pd(diffY, diffY);
	const __m512d len = _mm512_sqrt_pd(_mm512_add_pd(diffX_sqr, diffY_sqr));

	const __m512d agent_x_new = _mm512_add_pd(agent_xd, _mm512_div_pd(diffX, len));
	const __m512d agent_y_new = _mm512_add_pd(agent_yd, _mm512_div_pd(diffY, len));

	const __m256i agent_x_blend = _mm512_mask_cvt_roundpd_epi32(agent_x, has_dst, agent_x_new, ROUND);
	const __m256i agent_y_blend = _mm512_mask_cvt_roundpd_epi32(agent_y, has_dst, agent_y_new, ROUND);

	_mm256_store_epi32(&agents->x[agent_idx], agent_x_blend);
	_mm256_store_epi32(&agents->y[agent_idx], agent_y_blend);
}
