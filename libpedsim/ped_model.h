//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <atomic>
#include <map>
#include <set>
#include <vector>

#include <omp.h>

#include "ped_agent.h"
#include "ped_simd_agents.h"

#ifndef NOCUDA
#include "ped_cuda_agent.cuh"
#endif

namespace Ped {
class Tagent;

// The implementation modes for Assignment 1 + 2:
// chooses which implementation to use for tick()
enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ, SEQ_MV, OMP_MV };

class Model {
  public:
	// Sets everything up
	void setup(std::vector<Tagent *> agentsInScenario,
			   std::vector<Twaypoint *> destinationsInScenario,
			   IMPLEMENTATION implementation,
			   bool timing_mode);

	// Coordinates a time step in the scenario: move all agents by one step (if applicable).
	void tick();

	// Returns the agents of this scenario
	const std::vector<Tagent *> &getAgents() const { return agents; };

	const struct agents *get_agents_s() const { return &agents_s; };
	enum IMPLEMENTATION get_implementation() const { return this->implementation; }

	// Adds an agent to the tree structure
	void placeAgent(const Ped::Tagent *a);

	// Cleans up the tree and restructures it. Worth calling every now and then.
	void cleanup();
	~Model();

	// Returns the heatmap visualizing the density of agents
	int const *const *getHeatmap() const { return blurred_heatmap; };
	int getHeatmapSize() const;

  private:
	// Denotes which implementation (sequential, parallel implementations..)
	// should be used for calculating the desired positions of
	// agents (Assignment 1)
	IMPLEMENTATION implementation;

	bool timing_mode;

	// The agents in this scenario
	std::vector<Tagent *> agents;
	struct agents agents_s;
	struct agents agents_d;
#define GRID_WIDTH 160
#define GRID_HEIGHT 120
#define MAX_NUM_REGIONS 16
#define NUM_ALTERNATIVES 3
	size_t CUR_NUM_REGIONS;
	struct pair_s {
		int x, y;
	};
	struct region_s {
		int x_start, x_end;
		std::vector<std::atomic<bool>> lborder, rborder;
		std::vector<Tagent *> region_agents;
		std::vector<struct pair_s> taken_positions;
	};
	std::vector<struct region_s> regions;
	size_t *agents_buckets;

	// The waypoints in this scenario
	std::vector<Twaypoint *> destinations;

	void pthread_tick(const int k, int id);

	// Moves an agent towards its next position
	void move(Ped::Tagent *agent);

	void regions_init(void);
	void regions_dinit(void);
	void move_parallel(struct region_s *region, int agent_idx);
	void setup_regions(void);
	void print_total_agents(void);
	void get_agents_in_region(struct region_s *region);
	void leave_border(struct region_s *region, Ped::Tagent *agent, int x, int y);
	bool try_place_on_border(struct region_s *region, Ped::Tagent *agent, int x, int y);
	bool try_migrate(struct region_s *region, Ped::Tagent *agent, int x, int y);
	bool try_migrate_outside_grid(struct region_s *region, Ped::Tagent *agent, int x, int y);
	bool find_pair(std::vector<struct pair_s> taken_positions, struct pair_s pair);

	////////////
	/// Everything below here won't be relevant until Assignment 3
	///////////////////////////////////////////////

	// Returns the set of neighboring agents for the specified position
	set<const Ped::Tagent *> getNeighbors(int x, int y, int dist) const;

	////////////
	/// Everything below here won't be relevant until Assignment 4
	///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE *CELLSIZE

	// The heatmap representing the density of agents
	int **heatmap;

	// The scaled heatmap that fits to the view
	int **scaled_heatmap;

	// The final heatmap: blurred and scaled to fit the view
	int **blurred_heatmap;

	void setupHeatmapSeq();
	void updateHeatmapSeq();
};
} // namespace Ped
#endif
