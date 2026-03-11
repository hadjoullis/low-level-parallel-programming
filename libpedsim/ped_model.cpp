//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <stack>
#include <thread>

#ifndef NOCUDA
#include "cuda_testkernel.h"
#endif

void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario,
					   std::vector<Twaypoint *> destinationsInScenario,
					   IMPLEMENTATION implementation,
					   bool timing_mode) {
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
#else
	std::cout << "Not compiled for CUDA" << std::endl;
#endif

	// Set
	agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint *>(destinationsInScenario.begin(),
												 destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;
	this->timing_mode = timing_mode;

	switch (implementation) {
	case Ped::VECTOR:
		printf("Setting up data structures for SIMD...\n");
		agents_s = {0};
		simd_init(agents, &agents_s);
		printf("Data structures set up for SIMD complete.\n");
		break;
	case Ped::SEQ_MV_HM:
		printf("Setting up data structures for SEQ_MV_HM...\n");
		setupHeatmapSeq();
		printf("Data structures set up for SEQ_MV_HM complete.\n");
		break;
	case Ped::OMP_MV:
		printf("Setting up data structures for OMP_MV...\n");
		mv_parallel_regions_init(regions, agents);
		printf("Data structures set up for OMP_MV complete.\n");
		break;
	case Ped::OMP_MV_HM_SEQ:
		printf("Setting up data structures for OMP_MV_HM_SEQ...\n");
		total_hm_time = 0;
		ticks_cnt = 0;
		setupHeatmapSeq();
		mv_parallel_regions_init(regions, agents);
		printf("Data structures set up for OMP_MV_HM_SEQ complete.\n");
		break;
#ifndef NOCUDA
	case Ped::OMP_MV_HM:
		printf("Setting up data structures for OMP_MV_HM...\n");
		total_hm_time = 0;
		ticks_cnt = 0;
		hmcu_init(&hmcu, agents.size(), &hmcu_time);
		blurred_heatmap = hmcu.blurred_heatmap;
		mv_parallel_regions_init(regions, agents);
		printf("Data structures set up for OMP_MV_HM complete.\n");
		break;
	case Ped::OMP_MV_HM_BN:
		printf("Setting up data structures for OMP_MV_HM_BN...\n");
		cudaStreamCreate(&other_stream);
		total_hm_time = 0;
		ticks_cnt = 0;
		dscu_init(agents, &dscu_agents_h, &dscu_agents_d);
		hmcu_init(&hmcu, agents.size(), &hmcu_time);
		blurred_heatmap = hmcu.blurred_heatmap;
		mv_parallel_struct_regions_init(regions_bn, dscu_agents_h.size, dscu_agents_h.x);
		printf("Data structures set up for OMP_MV_HM_BN complete.\n");
		break;
	case Ped::CUDA:
		printf("Setting up data structures for cuda...\n");
		agents_s = {0};
		agents_d = {0};
		cuda_init(agents, &agents_s, &agents_d);
		printf("Data structures set up for cuda complete.\n");
		break;
#endif
	default:
		printf("No extra setup needed for given implementation\n");
	}
}

void Ped::Model::pthread_tick(const int k, int id) {
	auto &agents = this->agents;
	const int n = agents.size();

	const int chunk_sz = n / k;
	const int start = id * chunk_sz;
	const int end = ((id != k - 1) ? ((id + 1) * chunk_sz) : n);

	for (int i = start; i < end; i++) {
		auto *agent = agents[i];
		agent->computeNextDesiredPosition();
		const int x = agent->getDesiredX();
		const int y = agent->getDesiredY();
		agent->setX(x);
		agent->setY(y);
	}
}

void Ped::Model::tick() {
	// EDIT HERE FOR ASSIGNMENT 1
	switch (this->implementation) {
	case Ped::SEQ: {
		for (auto *const agent : this->agents) {
			agent->computeNextDesiredPosition();
			const int x = agent->getDesiredX();
			const int y = agent->getDesiredY();
			agent->setX(x);
			agent->setY(y);
		}
		break;
	}
	case Ped::SEQ_MV: {
		for (auto *const agent : this->agents) {
			agent->computeNextDesiredPosition();
			move(agent);
		}
		break;
	}
	case Ped::SEQ_MV_HM: {
		for (auto *const agent : this->agents) {
			agent->computeNextDesiredPosition();
			move(agent);
		}
		updateHeatmapSeq();
		break;
	}
	case Ped::OMP: {
		auto &agents = this->agents;
		const int n = agents.size();

#pragma omp parallel for default(none) shared(n, agents)
		for (int i = 0; i < n; i++) {
			auto *agent = agents[i];
			agent->computeNextDesiredPosition();
			const int x = agent->getDesiredX();
			const int y = agent->getDesiredY();
			agent->setX(x);
			agent->setY(y);
		}
		break;
	}
	case Ped::OMP_MV: {
		auto &agents = this->agents;
		const int n = agents.size();
		int CUR_NUM_REGIONS;

#pragma omp parallel default(none) shared(n, agents, CUR_NUM_REGIONS, regions)
		{
#pragma omp single nowait
			{
				CUR_NUM_REGIONS = mv_parallel_setup_regions(regions, agents);
			}
#pragma omp for
			for (int i = 0; i < n; i++) {
				auto *agent = agents[i];
				agent->computeNextDesiredPosition();
			} // implicit barrier
#pragma omp for
			for (int i = 0; i < CUR_NUM_REGIONS; i++) {
				mv_parallel_get_agents_in_region(agents, &regions[i]);
			} // implicit barrier
#pragma omp for
			for (int i = 0; i < CUR_NUM_REGIONS; i++) {
				const int size = regions[i].region_agents.size();
				for (int agent_idx = 0; agent_idx < size; agent_idx++) {
					move_parallel(&regions[i], agent_idx);
				}
				regions[i].region_agents.clear();
				regions[i].taken_positions.clear();
			}
		}
		break;
	}
	case Ped::OMP_MV_HM_SEQ: {
		auto &agents = this->agents;
		const int n = agents.size();
		int CUR_NUM_REGIONS;

#pragma omp parallel default(none) shared(n, agents, CUR_NUM_REGIONS, regions, total_hm_time, ticks_cnt)
		{
#pragma omp single nowait
			{
				CUR_NUM_REGIONS = mv_parallel_setup_regions(regions, agents);
			}
#pragma omp for
			for (int i = 0; i < n; i++) {
				auto *agent = agents[i];
				agent->computeNextDesiredPosition();
			} // implicit barrier
#pragma omp single nowait
			{
				const double start = omp_get_wtime();
				updateHeatmapSeq();
				const double end = omp_get_wtime();
				total_hm_time += end - start; // seconds
				ticks_cnt++;
			}
#pragma omp for
			for (int i = 0; i < CUR_NUM_REGIONS; i++) {
				mv_parallel_get_agents_in_region(agents, &regions[i]);
			} // implicit barrier
#pragma omp for
			for (int i = 0; i < CUR_NUM_REGIONS; i++) {
				const int size = regions[i].region_agents.size();
				for (int agent_idx = 0; agent_idx < size; agent_idx++) {
					move_parallel(&regions[i], agent_idx);
				}
				regions[i].region_agents.clear();
				regions[i].taken_positions.clear();
			}
		}
		break;
	}
	case Ped::PTHREAD: {
		static bool once = true;
		static int PTHREAD_NUM_THREADS = 8;
		if (once) {
			char *retval = getenv("PTHREAD_NUM_THREADS");
			if (retval) {
				PTHREAD_NUM_THREADS = atoi(retval);
			}
			once = false;
		}
		static std::vector<std::thread> tid(PTHREAD_NUM_THREADS);
		for (int i = 0; i < PTHREAD_NUM_THREADS; i++) {
			tid[i] = std::thread(&Ped::Model::pthread_tick, this, PTHREAD_NUM_THREADS, i);
		}
		for (auto &t : tid) {
			t.join();
		}
		break;
	}
	case Ped::VECTOR: {
		size_t i;
		for (i = 0; i + STEPS <= agents_s.size; i += STEPS) {
			simd_computeNextDesiredPosition(&agents_s, i);
		}
		for (; i < agents_s.size; i++) {
			single_computeNextDesiredPosition(&agents_s, i);
		}
		break;
	}
#ifndef NOCUDA
	case Ped::OMP_MV_HM: {
		// printf("START\n");
		auto &agents = this->agents;
		const int n = agents.size();
		int CUR_NUM_REGIONS;
		double start;

#pragma omp parallel default(none) shared(n, agents, CUR_NUM_REGIONS, regions, start)
		{
#pragma omp single nowait
			{
				CUR_NUM_REGIONS = mv_parallel_setup_regions(regions, agents);
			}
#pragma omp for
			for (int i = 0; i < n; i++) {
				auto *agent = agents[i];
				agent->computeNextDesiredPosition();
				hmcu.pairs_h.x[i] = agent->getDesiredX();
				hmcu.pairs_h.y[i] = agent->getDesiredY();
			} // implicit barrier
#pragma omp single nowait
			{
				start = omp_get_wtime();
				hmcu_update_heatmap(&hmcu, &hmcu_time);
			}
#pragma omp for
			for (int i = 0; i < CUR_NUM_REGIONS; i++) {
				mv_parallel_get_agents_in_region(agents, &regions[i]);
			} // implicit barrier
#pragma omp for
			for (int i = 0; i < CUR_NUM_REGIONS; i++) {
				// printf("MOVING AGENTS\n");
				const int size = regions[i].region_agents.size();
				for (int agent_idx = 0; agent_idx < size; agent_idx++) {
					move_parallel(&regions[i], agent_idx);
				}
				regions[i].region_agents.clear();
				regions[i].taken_positions.clear();
			}
		}
		// printf("MOVED ALL AGENTS!\n");
		cudaDeviceSynchronize();
		const double end = omp_get_wtime();
		total_hm_time += end - start; // seconds
		ticks_cnt++;

		cudaEventSynchronize(hmcu_time.eblur);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, hmcu_time.sfade, hmcu_time.efade);
		hmcu_time.fade += elapsedTime;
		cudaEventElapsedTime(&elapsedTime, hmcu_time.sinsert, hmcu_time.einsert);
		hmcu_time.insert += elapsedTime;
		cudaEventElapsedTime(&elapsedTime, hmcu_time.scap_scale, hmcu_time.ecap_scale);
		hmcu_time.cap_scale += elapsedTime;
		cudaEventElapsedTime(&elapsedTime, hmcu_time.sblur, hmcu_time.eblur);
		hmcu_time.blur += elapsedTime;
		// printf("END\n");
		break;
	}
	case Ped::OMP_MV_HM_BN: {
		// printf("START\n");
		int CUR_NUM_REGIONS;
		double start;

#pragma omp parallel default(none)                                                                           \
	shared(other_stream, dscu_agents_h, dscu_agents_d, CUR_NUM_REGIONS, regions_bn, start)
		{
#pragma omp single nowait
			{
				cudaMemcpy(dscu_agents_d.x,
						   dscu_agents_h.x,
						   dscu_agents_h.size * sizeof(int),
						   cudaMemcpyHostToDevice);
				cudaMemcpy(dscu_agents_d.y,
						   dscu_agents_h.y,
						   dscu_agents_h.size * sizeof(int),
						   cudaMemcpyHostToDevice);
				dscu_compute_next_desired_position(&dscu_agents_d);
				cudaMemcpyAsync(dscu_agents_h.des_x,
								dscu_agents_d.des_x,
								dscu_agents_h.size * sizeof(int),
								cudaMemcpyDeviceToHost,
								other_stream);
				cudaMemcpyAsync(dscu_agents_h.des_y,
								dscu_agents_d.des_y,
								dscu_agents_h.size * sizeof(int),
								cudaMemcpyDeviceToHost,
								other_stream);
				start = omp_get_wtime();
				hmcu_update_heatmap_bn(
					&hmcu, dscu_agents_h.size, dscu_agents_d.des_x, dscu_agents_d.des_y, &hmcu_time);
			}
#pragma omp single
			{
				CUR_NUM_REGIONS = mv_parallel_struct_setup_regions(regions_bn, dscu_agents_h.size);
			} // implicit barrier
#pragma omp for
			for (int i = 0; i < CUR_NUM_REGIONS; i++) {
				mv_parallel_struct_get_agents_in_region(&dscu_agents_h, &regions_bn[i]);
			} // implicit barrier
#pragma omp single
			{
				cudaStreamSynchronize(other_stream);
			} // implicit barrier
#pragma omp for
			for (int i = 0; i < CUR_NUM_REGIONS; i++) {
				// printf("MOVING AGENTS\n");
				const int size = regions_bn[i].region_agents.size();
				for (int agent_idx = 0; agent_idx < size; agent_idx++) {
					move_parallel_struct(&regions_bn[i], &dscu_agents_h, agent_idx);
				}
				regions_bn[i].region_agents.clear();
				regions_bn[i].taken_positions.clear();
			}
		}
		// printf("MOVED ALL AGENTS!\n");
		cudaDeviceSynchronize();
		const double end = omp_get_wtime();
		total_hm_time += end - start; // seconds
		ticks_cnt++;

		cudaEventSynchronize(hmcu_time.eblur);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, hmcu_time.sfade, hmcu_time.efade);
		hmcu_time.fade += elapsedTime;
		cudaEventElapsedTime(&elapsedTime, hmcu_time.sinsert, hmcu_time.einsert);
		hmcu_time.insert += elapsedTime;
		cudaEventElapsedTime(&elapsedTime, hmcu_time.scap_scale, hmcu_time.ecap_scale);
		hmcu_time.cap_scale += elapsedTime;
		cudaEventElapsedTime(&elapsedTime, hmcu_time.sblur, hmcu_time.eblur);
		hmcu_time.blur += elapsedTime;
		// printf("END\n");
		break;
	}
	case Ped::CUDA: {
		static dim3 threads_per_block(THREADS_PER_BLOCK, 1, 1);
		static dim3 blocks(((agents_s.size + threads_per_block.x - 1) / threads_per_block.x), 1, 1);
		static const size_t bytes = sizeof(int) * agents_s.size;

		kernel_launch(blocks, threads_per_block, &agents_d);

		if (timing_mode) {
			cudaDeviceSynchronize();
			break;
		}
		cudaMemcpy(agents_s.x, agents_d.x, bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(agents_s.y, agents_d.y, bytes, cudaMemcpyDeviceToHost);
		break;
	}
#endif
	default:
		fprintf(stderr, "ERROR: NOT IMPLEMENTED\n");
		exit(1);
	}
}

////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent) {
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int>> takenPositions;
	for (std::set<const Ped::Tagent *>::iterator neighborIt = neighbors.begin();
		 neighborIt != neighbors.end();
		 ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int>> prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0) {
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	} else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int>>::iterator it = prioritizedAlternatives.begin();
		 it != prioritizedAlternatives.end();
		 ++it) {
		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents
/// (search field is a square in the current implementation)
set<const Ped::Tagent *> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this
	// programmer is lazy.)
	return set<const Ped::Tagent *>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now.
}

Ped::Model::~Model() {
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent) { delete agent; });
	std::for_each(
		destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination) { delete destination; });

	switch (implementation) {
	case Ped::VECTOR:
		printf("Cleaning up data structures for SIMD...\n");
		simd_dinit(&agents_s);
		printf("Data structures for SIMD released.\n");
		break;
	case Ped::SEQ_MV_HM:
		printf("Cleaning up data structures for SEQ_MV_HM...\n");
		freeHeatmapSeq();
		printf("Data structures for SEQ_MV_HM released.\n");
		break;
	case Ped::OMP_MV:
		printf("Cleaning up data structures for OMP_MV...\n");
		mv_parallel_regions_dinit();
		printf("Data structures for OMP_MV released.\n");
		break;
	case Ped::OMP_MV_HM_SEQ:
		printf("HM_SEQ_TOTAL_TIME: %.6lf seconds\n", total_hm_time);
		printf("HM_SEQ_AVG_TIME: %.6lf seconds\n", total_hm_time / ticks_cnt);
		printf("Cleaning up data structures for OMP_MV_HM_SEQ...\n");
		freeHeatmapSeq();
		mv_parallel_regions_dinit();
		printf("Data structures for OMP_MV_HM_SEQ released.\n");
		break;
#ifndef NOCUDA
	case Ped::OMP_MV_HM:
		// from ms to seconds
		hmcu_time.fade /= 1000;
		hmcu_time.insert /= 1000;
		hmcu_time.cap_scale /= 1000;
		hmcu_time.blur /= 1000;
		printf("HM_CUDA_TOTAL_TIME: %.6lf seconds\n", total_hm_time);
		printf("HM_CUDA_AVG_TIME: %.6lf seconds\n", total_hm_time / ticks_cnt);
		printf("HM_CUDA_FADE_TOTAL_TIME: %.6lf seconds\n", hmcu_time.fade);
		printf("HM_CUDA_FADE_AVG_TIME: %.6lf seconds\n", hmcu_time.fade / ticks_cnt);
		printf("HM_CUDA_INSERT_TOTAL_TIME: %.6lf seconds\n", hmcu_time.insert);
		printf("HM_CUDA_INSERT_AVG_TIME: %.6lf seconds\n", hmcu_time.insert / ticks_cnt);
		printf("HM_CUDA_CAP_SCALE_TOTAL_TIME: %.6lf seconds\n", hmcu_time.cap_scale);
		printf("HM_CUDA_CAP_SCALE_AVG_TIME: %.6lf seconds\n", hmcu_time.cap_scale / ticks_cnt);
		printf("HM_CUDA_BLUR_TOTAL_TIME: %.6lf seconds\n", hmcu_time.blur);
		printf("HM_CUDA_BLUR_AVG_TIME: %.6lf seconds\n", hmcu_time.blur / ticks_cnt);
		printf("Cleaning up data structures for OMP_MV_HM...\n");
		hmcu_dinit(&hmcu, &hmcu_time);
		mv_parallel_regions_dinit();
		printf("Data structures for OMP_MV_HM released.\n");
		break;
	case Ped::OMP_MV_HM_BN:
		hmcu_time.fade /= 1000;
		hmcu_time.insert /= 1000;
		hmcu_time.cap_scale /= 1000;
		hmcu_time.blur /= 1000;
		printf("HM_BN_CUDA_TOTAL_TIME: %.6lf seconds\n", total_hm_time);
		printf("HM_BN_CUDA_AVG_TIME: %.6lf seconds\n", total_hm_time / ticks_cnt);
		printf("HM_BN_CUDA_FADE_TOTAL_TIME: %.6lf seconds\n", hmcu_time.fade);
		printf("HM_BN_CUDA_FADE_AVG_TIME: %.6lf seconds\n", hmcu_time.fade / ticks_cnt);
		printf("HM_BN_CUDA_INSERT_TOTAL_TIME: %.6lf seconds\n", hmcu_time.insert);
		printf("HM_BN_CUDA_INSERT_AVG_TIME: %.6lf seconds\n", hmcu_time.insert / ticks_cnt);
		printf("HM_BN_CUDA_CAP_SCALE_TOTAL_TIME: %.6lf seconds\n", hmcu_time.cap_scale);
		printf("HM_BN_CUDA_CAP_SCALE_AVG_TIME: %.6lf seconds\n", hmcu_time.cap_scale / ticks_cnt);
		printf("HM_BN_CUDA_BLUR_TOTAL_TIME: %.6lf seconds\n", hmcu_time.blur);
		printf("HM_BN_CUDA_BLUR_AVG_TIME: %.6lf seconds\n", hmcu_time.blur / ticks_cnt);
		printf("Setting up data structures for OMP_MV_HM_BN...\n");
		cudaStreamDestroy(other_stream);
		// hmcu_dinit(&hmcu, &hmcu_time);
		// dscu_dinit(&dscu_agents_h);
		// mv_parallel_struct_regions_dinit();
		printf("Data structures set up for OMP_MV_HM_BN complete.\n");
		break;
	case Ped::CUDA:
		printf("Cleaning up data structures for cuda...\n");
		cuda_dinit(&agents_s);
		printf("Data structures for cuda released.\n");
		break;
#endif
	default:
		printf("No extra cleanup needed for given implementation.\n");
	}
}
