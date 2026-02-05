//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include <omp.h>
#include <thread>

#ifndef NOCDUA
#include "cuda_testkernel.h"
#endif

#include <stdlib.h>

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
#ifndef NOCUDA
	// Convenience test: does CUDA work on this machine?
	cuda_test();
#else
    std::cout << "Not compiled for CUDA" << std::endl;
#endif

	// Set
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();

	if (implementation != Ped::VECTOR) {
		return;
	}
	const size_t agents_size = agentsInScenario.size();
	const size_t align = 16;
	posix_memalign((void **)&agents_s.x, align, sizeof(int) * agents_size);
	posix_memalign((void **)&agents_s.y, align, sizeof(int) * agents_size);
	posix_memalign((void **)&agents_s.desiredPositionX, align, sizeof(int) * agents_size);
	posix_memalign((void **)&agents_s.desiredPositionY, align, sizeof(int) * agents_size);
	posix_memalign((void **)&agents_s.destination_idx, align, sizeof(ssize_t) * agents_size);
	posix_memalign((void **)&agents_s.lastDestination_idx, align, sizeof(ssize_t) * agents_size);
	// maybe have a single waypoints array for all agents...
	posix_memalign((void **)&agents_s.waypoints.x, align, sizeof(double *) * agents_size);
	posix_memalign((void **)&agents_s.waypoints.y, align, sizeof(double *) * agents_size);
	posix_memalign((void **)&agents_s.waypoints.r, align, sizeof(double *) * agents_size);
	for (size_t i = 0; i < agents_size; i++) {
		const size_t bytes = sizeof(double) * agents[i]->getWaypointsSize();
		posix_memalign((void **)&agents_s.waypoints.x[i], align, bytes);
		posix_memalign((void **)&agents_s.waypoints.y[i], align, bytes);
		posix_memalign((void **)&agents_s.waypoints.r[i], align, bytes);
	}
	posix_memalign((void **)&agents_s.waypoints.sz, align, sizeof(size_t) * agents_size);

	agents_s.size = agents_size;
	for (size_t i = 0; i < agents_size; i++) {
		agents_s.x[i] = agents[i]->getX();
		agents_s.y[i] = agents[i]->getY();
		// agents_s.desiredPositionX[i]: not set
		// agents_s.desiredPositionY[i]: not set
		agents_s.destination_idx[i] = -1;
		agents_s.lastDestination_idx[i] = -1;
		// agents_s.waypoints: already set
		agents_s.waypoints.sz[i] = agents[i]->getWaypointsSize();
		for (size_t j = 0; j < agents_s.waypoints.sz[i]; j++) {
			auto wp = agents[i]->getWaypoint(j);
			agents_s.waypoints.x[i][j] = wp->getx();
			agents_s.waypoints.y[i][j] = wp->gety();
			agents_s.waypoints.r[i][j] = wp->getr();
		}
	}
}

void Ped::Model::pthread_tick(const int k, int id) {
	auto& agents = this->agents;
	const int n = agents.size();

	const int chunk_sz = n / k;
	const int start = id * chunk_sz;
	const int end = ((id != k - 1) ? ((id + 1) * chunk_sz) : n);

	for (int i = start; i < end; i++) {
		auto* agent = agents[i];
		agent->computeNextDesiredPosition();
		const int x = agent->getDesiredX();
		const int y = agent->getDesiredY();
		agent->setX(x);
		agent->setY(y);
	}
}

void Ped::Model::tick()
{
	// EDIT HERE FOR ASSIGNMENT 1
	switch (this->implementation) {
	case Ped::SEQ: {
		for (auto* const agent: this->agents) {
			agent->computeNextDesiredPosition();
			const int x = agent->getDesiredX();
			const int y = agent->getDesiredY();
			agent->setX(x);
			agent->setY(y);
		}
		break;
	}
	case Ped::OMP: {
		auto& agents = this->agents;
		const int n = agents.size();

		#pragma omp parallel for default(none) shared(n,agents)
		for (int i = 0; i < n; i++) {
			auto* agent = agents[i];
			agent->computeNextDesiredPosition();
			const int x = agent->getDesiredX();
			const int y = agent->getDesiredY();
			agent->setX(x);
			agent->setY(y);
		}
		break;
	}
	case Ped::PTHREAD: {
		static bool once = true;
		static int PTHREAD_NUM_THREADS = 8;
		if (once) {
			char *retval = getenv("PTHREAD_NUM_THREADS");
			if (retval) { PTHREAD_NUM_THREADS = atoi(retval); }
			once = false;
		}
		static std::vector<std::thread> tid(PTHREAD_NUM_THREADS);
		for (int i = 0; i < PTHREAD_NUM_THREADS; i++) {
			tid[i] = std::thread(&Ped::Model::pthread_tick, this, PTHREAD_NUM_THREADS, i);
		}
		for (auto& t: tid) { t.join(); }
		break;
	}
	case Ped::VECTOR: {
		for (size_t i = 0; i < agents_s.size; i++) {
			struct_agents_computeNextDesiredPosition(&agents_s, i);
			agents_s.x[i] = agents_s.desiredPositionX[i];
			agents_s.y[i] = agents_s.desiredPositionY[i];
		}
		break;
	}
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
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

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
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
	if (this->implementation != Ped::VECTOR) {
		return;
	}
	free(agents_s.x);
	free(agents_s.y);
	free(agents_s.desiredPositionX);
	free(agents_s.desiredPositionY);
	free(agents_s.destination_idx);
	free(agents_s.lastDestination_idx);
	for (size_t i = 0; i < agents_s.size; i++) {
		free(agents_s.waypoints.x[i]);
		free(agents_s.waypoints.y[i]);
		free(agents_s.waypoints.r[i]);
	}
	free(agents_s.waypoints.x);
	free(agents_s.waypoints.y);
	free(agents_s.waypoints.r);
	free(agents_s.waypoints.sz);
}
