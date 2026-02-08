//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_simd_agents.h"

static ssize_t single_get_nextDestination_idx(const struct agents *agents, const size_t agent_idx) {
	ssize_t nextDestination_idx = -1;
	bool agentReachedDestination = false;
	const int agent_x = agents->x[agent_idx];
	const int agent_y = agents->y[agent_idx];

	if (agents->destination_idx[agent_idx] != -1) {
		// compute if agent reached its current destination
		const ssize_t dst_idx = agents->destination_idx[agent_idx];
		const double diffX = agents->waypoints.x[agent_idx][dst_idx] - agent_x;
		const double diffY = agents->waypoints.y[agent_idx][dst_idx] - agent_y;
		const double len = sqrt(diffX * diffX + diffY * diffY);
		agentReachedDestination = len < agents->waypoints.r[agent_idx][dst_idx];
	}

	if ((agentReachedDestination || agents->destination_idx[agent_idx] == -1) &&
		agents->waypoints.sz[agent_idx] != 0) {
		// Case 1: agent has reached destination (or has no current
		// destination); get next destination if available
		ssize_t wps_idx = agents->destination_idx[agent_idx] + 1;
		if (wps_idx == agents->waypoints.sz[agent_idx]) {
			wps_idx = -1;
		}
		nextDestination_idx = wps_idx;
	} else {
		// Case 2: agent has not yet reached destination, continue to move
		// towards current destination
		nextDestination_idx = agents->destination_idx[agent_idx];
	}
	return nextDestination_idx;
}

void single_computeNextDesiredPosition(const struct agents *agents, const size_t agent_idx) {
	ssize_t nextDestination_idx = single_get_nextDestination_idx(agents, agent_idx);
	agents->destination_idx[agent_idx] = nextDestination_idx;
	if (agents->destination_idx[agent_idx] == -1) {
		return; // no destination, no need to compute where to move to
	}
	const int agent_x = agents->x[agent_idx];
	const int agent_y = agents->y[agent_idx];

	const ssize_t dst_idx = agents->destination_idx[agent_idx];
	const double diffX = agents->waypoints.x[agent_idx][dst_idx] - agent_x;
	const double diffY = agents->waypoints.y[agent_idx][dst_idx] - agent_y;
	const double len = sqrt(diffX * diffX + diffY * diffY);
	agents->desiredPositionX[agent_idx] = (int)round(agent_x + diffX / len);
	agents->desiredPositionY[agent_idx] = (int)round(agent_y + diffY / len);
}

static inline __m512d fetch_dsts(ssize_t *destination_idx, double **coord, const size_t agent_idx) {
	static double dsts[STEPS] __attribute__((aligned(ALIGN)));
	for (size_t i = 0; i < STEPS; i++) {
		const ssize_t dst_idx = destination_idx[agent_idx + i];
		dsts[i] = coord[agent_idx + i][dst_idx];
	}
	return _mm512_load_pd(&dsts);
}

static __m512i simd_get_nextDestination_idx(const struct agents *agents, const size_t agent_idx) {
	const __m256i agent_x = _mm256_load_epi32(&agents->x[agent_idx]);
	const __m256i agent_y = _mm256_load_epi32(&agents->y[agent_idx]);
	const __m512d agent_xd = _mm512_cvtepi32_pd(agent_x);
	const __m512d agent_yd = _mm512_cvtepi32_pd(agent_y);

	// ·since·this·is·simd,·we·have·to·uncoditionally·execute·both·branching
	// ·paths,·even·if·'dst_idx'·of·an·agent·is·'-1'.·Make·sure·this·is
	// ·taken·care·of·in·initial·allocations.
	const __m512d dst_x = fetch_dsts(agents->destination_idx, agents->waypoints.x, agent_idx);
	const __m512d diffX = _mm512_sub_pd(dst_x, agent_xd);

	const __m512d dst_y = fetch_dsts(agents->destination_idx, agents->waypoints.y, agent_idx);
	const __m512d diffY = _mm512_sub_pd(dst_y, agent_yd);

	const __m512d diffX_sqr = _mm512_mul_pd(diffX, diffX);
	const __m512d diffY_sqr = _mm512_mul_pd(diffY, diffY);

	const __m512d len = _mm512_sqrt_pd(_mm512_add_pd(diffX_sqr, diffY_sqr));
	const __m512d dst_r = fetch_dsts(agents->destination_idx, agents->waypoints.r, agent_idx);
	const __mmask8 agentReachedDestination = _mm512_cmplt_pd_mask(len, dst_r);

	const __m512i wps_sz = _mm512_load_epi64(&agents->waypoints.sz[agent_idx]);
	const __m512i zero = _mm512_setzero_si512();
	const __mmask8 non_empty_sz = _mm512_cmpneq_epi64_mask(wps_sz, zero);

	const __m512i one = _mm512_set1_epi64(1);
	__m512i wps_idx = _mm512_load_epi64(&agents->destination_idx[agent_idx]);
	wps_idx = _mm512_add_epi64(wps_idx, one);
	const __mmask8 overflow = _mm512_cmpeq_epi64_mask(wps_idx, wps_sz);
	const __m512i minus_one = _mm512_set1_epi64(-1);

	// Case 1: agent has reached destination (or has no current destination);
	// get next destination if available
	const __m512i nextDestination_idx_if = _mm512_mask_blend_epi64(overflow, wps_idx, minus_one);
	// Case 2: agent has not yet reached destination, continue to move towards
	// current destination
	const __m512i nextDestination_idx_else = _mm512_load_epi64(&agents->destination_idx[agent_idx]);

	const __m512i nextDestination_idx = _mm512_mask_blend_epi64(
		_kand_mask8(agentReachedDestination, non_empty_sz),
		nextDestination_idx_else,
		nextDestination_idx_if);
	return nextDestination_idx;
}

void simd_computeNextDesiredPosition(const struct agents *agents, const size_t agent_idx) {
	const __m512i nextDestination_idx = simd_get_nextDestination_idx(agents, agent_idx);
	_mm512_store_epi64(&agents->destination_idx[agent_idx], nextDestination_idx);

	const __m256i agent_x = _mm256_load_epi32(&agents->x[agent_idx]);
	const __m256i agent_y = _mm256_load_epi32(&agents->y[agent_idx]);
	const __m512d agent_xd = _mm512_cvtepi32_pd(agent_x);
	const __m512d agent_yd = _mm512_cvtepi32_pd(agent_y);

	// no destination, no need to compute where to move to
	const __m512i dst_idx = _mm512_load_epi64(&agents->destination_idx[agent_idx]);
	// copy from src when bit is NOT set
	const __m512i minus_one = _mm512_set1_epi64(-1);
	const __mmask8 has_dst = _mm512_cmpneq_epi64_mask(dst_idx, minus_one);

	const __m256i desiredPositionX = _mm256_load_epi32(&agents->desiredPositionX[agent_idx]);
	const __m256i desiredPositionY = _mm256_load_epi32(&agents->desiredPositionY[agent_idx]);

	const __m512d dst_x = fetch_dsts(agents->destination_idx, agents->waypoints.x, agent_idx);
	const __m512d diffX = _mm512_sub_pd(dst_x, agent_xd);
	const __m512d dst_y = fetch_dsts(agents->destination_idx, agents->waypoints.y, agent_idx);
	const __m512d diffY = _mm512_sub_pd(dst_y, agent_yd);
	const __m512d diffX_sqr = _mm512_mul_pd(diffX, diffX);
	const __m512d diffY_sqr = _mm512_mul_pd(diffY, diffY);
	const __m512d len = _mm512_sqrt_pd(_mm512_add_pd(diffX_sqr, diffY_sqr));

	const __m512d desiredPositionX_new = _mm512_add_pd(agent_xd, _mm512_div_pd(diffX, len));
	const __m512d desiredPositionY_new = _mm512_add_pd(agent_yd, _mm512_div_pd(diffY, len));

	const __m256i desiredPositionX_blend = _mm512_mask_cvt_roundpd_epi32(
		desiredPositionX, has_dst, desiredPositionX_new, ROUND);
	const __m256i desiredPositionY_blend = _mm512_mask_cvt_roundpd_epi32(
		desiredPositionY, has_dst, desiredPositionY_new, ROUND);

	_mm256_store_epi32(&agents->desiredPositionX[agent_idx], desiredPositionX_blend);
	_mm256_store_epi32(&agents->desiredPositionY[agent_idx], desiredPositionY_blend);
}
