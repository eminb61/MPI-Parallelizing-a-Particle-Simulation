#include "common.h"
#include <mpi.h>

#include <vector>
#include <bits/stdc++.h>
#include <iostream>
#include <assert.h>
#include <cmath>

using namespace std;

#define BINSIZE (cutoff * 2)

#define MIN(x,y) (((x)<(y))?(x):(y))

// Put any static global variables here that you will use throughout the simulation.
int dim, remains, quotients;
int subdomain_size, subdomain_start;
int num_bins, bin_dim;

// 8 directions plus itself
int dir[9][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 0}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
// bins
vector< vector<particle_t> > bins;
// <rank, vector<bins>>
map<int, vector<int>> send_bins, receive_bins;
// <rank, vector<particles>>
map<int, vector<particle_t>> send_parts, receive_parts;
// vector of particles
vector<particle_t> send_part_all, receive_part_all;
// num of particles
vector<int> send_part_num, receive_part_num;
// <rank, MPI_Request>
map<int, array<MPI_Request, 2>> send_req, receive_req;

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;

    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

void move(particle_t& p, double size) {

    // move the particle
    p.x += (p.vx += p.ax * dt) * dt;
    p.y += (p.vy += p.ay * dt) * dt;

    // reset acceleration
    p.ax = p.ay = 0;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
    return;
}

int get_rank(int bin_idx){
    // get the rank for bin index
    if (bin_idx < remains * (quotients + 1)) {
      return bin_idx / (quotients+1);
    }else{
      return (bin_idx - remains * (quotients + 1)) / quotients + remains ;
    }
}

vector<int> get_size(vector<int> &num_particles){
    // helper function for getting size for MPI 
    int sums = 0;
    int n = num_particles.size();
    vector<int> result(n);
    for(int i = 0; i < n; ++i){
        result[i] = sums;
        sums += num_particles[i];
    }
    return result;
}

void send_receive_bins(int rank, unordered_map<int, set<int>> &temp_send_bins, 
                            unordered_map<int, set<int>> &temp_receive_bins){
    // get the send and receive bins for each process
    for(int i = 0; i < num_bins; ++i){
        int bi = (subdomain_start + i) / dim;
        int bj = (subdomain_start + i) % dim;
        for(int d = 0; d < 9; ++d){
            if (dir[d][0] == 0 and dir[d][1] == 0) 
                continue;
            int neigh_bi = bi + dir[d][0];
            int neigh_bj = bj + dir[d][1];
            if(neigh_bi < 0 or neigh_bi >= dim or neigh_bj < 0 or neigh_bj >= dim)
                continue;
            int neigh_rank = get_rank(neigh_bi * dim + neigh_bj);
            if(neigh_rank == rank)
                continue;
            // update receive and send bins
            temp_send_bins[neigh_rank].insert(subdomain_start + i);
            temp_receive_bins[neigh_rank].insert(neigh_bi * dim + neigh_bj);
        }
    }
}

void req_init(unordered_map<int, set<int>> &temp_send_bins, unordered_map<int, set<int>> &temp_receive_bins){
    // Initialize the MPI requests
    for(auto &it: temp_send_bins){
        int cur_rank = it.first;
        send_req[cur_rank];
        receive_req[cur_rank];

        auto &bin = it.second;
        for(int bin_idx: bin){
            send_bins[cur_rank].push_back(bin_idx);
        }

        auto &receive_bin = temp_receive_bins[cur_rank];
        for(int bin_idx: receive_bin){
            receive_bins[cur_rank].push_back(bin_idx);
        }
    }
}

void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here

    dim = floor(size / BINSIZE) + 1;
    quotients = dim * dim / num_procs;
    remains = dim * dim % num_procs;

    if(rank < remains){
        subdomain_start = rank * quotients + rank;
        num_bins = quotients +1;
    }else{
        subdomain_start = rank * quotients + remains;
        num_bins = quotients;
    }

    // initialize subdomain size
    int left_bi = subdomain_start / dim;
    int left_bj = subdomain_start % dim;
    subdomain_size = (left_bi - 1) * dim + (left_bj - 1);
    if (subdomain_size < 0)
        subdomain_size = 0;
    
    // initialize bin size
    int right_bi = (subdomain_start + num_bins - 1) / dim;
    int right_bj = (subdomain_start + num_bins - 1) % dim;
    bin_dim = MIN((right_bi + 1) * dim + (right_bj + 1), dim * dim-1);
    bins.resize(bin_dim  - subdomain_size + 1);

    // add particles to bins
    for(int i = 0; i < num_parts; ++i){
        int bin_idx = floor(parts[i].x / BINSIZE) * dim + floor(parts[i].y / BINSIZE);
        if(bin_idx >= subdomain_start && bin_idx < subdomain_start + num_bins){
            bins[bin_idx - subdomain_size].push_back(parts[i]);
        }
    }

    // initialize send and receive particles
    send_part_num.resize(num_procs);
    receive_part_num.resize(num_procs);

    unordered_map<int, set<int>> temp_send_bins;
    unordered_map<int, set<int>> temp_receive_bins;
    send_receive_bins(rank, temp_send_bins, temp_receive_bins);
    req_init(temp_send_bins, temp_receive_bins);

    // Wait for send and receive
    MPI_Barrier(MPI_COMM_WORLD);
}

void send_particles(int target_rank, vector<particle_t> &particles, array<MPI_Request, 2> &reqs, int &send_num){
    // MPI send particles
    MPI_Isend(&send_num, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(particles.data(), send_num, PARTICLE, target_rank, 0, MPI_COMM_WORLD, &reqs[1]);
}

void receive_particles(int rank){
    // MPI receive particles
    int receive_num;
    vector<particle_t> &receive_ps = receive_parts[rank];
    receive_ps.clear();
    auto &reqs = receive_req[rank];
    MPI_Recv(&receive_num, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    receive_ps.resize(receive_num);
    MPI_Irecv(receive_ps.data(), receive_num, PARTICLE, rank, 0, MPI_COMM_WORLD, &reqs[1]);
}

void rebin_particle(vector<particle_t> particles) {
    // rebin particle
    for(particle_t &p: particles){
        int bin_idx = floor(p.x / BINSIZE) * dim + floor(p.y / BINSIZE);
        bins[bin_idx - subdomain_size].push_back(p);
    }
}

void MPI_ALL(int num_procs, map<int, vector<particle_t>> &moving_out){
    // Get send particle number
    int send_total = 0;
    for(int i = 0; i < num_procs; ++i)
        send_part_num[i] = 0;
    for(auto &it: moving_out){
        int target_rank = it.first;
        auto &part_send = it.second;
        send_part_num[target_rank] += part_send.size();
        send_total += part_send.size();
    }

    // send and receive particles from each processor
    MPI_Alltoall(&send_part_num[0], 1, MPI_INT, &receive_part_num[0], 1, MPI_INT, MPI_COMM_WORLD);

    // resize receive particles
    int receive_nums = 0;
    for (int nums: receive_part_num)
        receive_nums += nums;
    receive_part_all.resize(receive_nums);

    // get send particles
    send_part_all.clear();
    for(auto &it: moving_out){
        for(particle_t &t: it.second){
            send_part_all.push_back(t);
        }
    }

    // Send and receive the particles all to all
    MPI_Alltoallv(&send_part_all[0], &send_part_num[0], &get_size(send_part_num)[0], PARTICLE, 
                &receive_part_all[0], &receive_part_num[0], &get_size(receive_part_num)[0], PARTICLE, MPI_COMM_WORLD);
    
    // Rebin the received particles
    rebin_particle(receive_part_all);
}

void move_particle(map<int, vector<particle_t>> &moving_out, vector<particle_t> &not_leaving, int size) {
    // move the particles
    for(int i = 0; i < num_bins; ++i){
        auto &bin = bins[i + (subdomain_start - subdomain_size)];

        // remove particle
        for(int j = bin.size()-1; j >= 0; --j){
            particle_t &p = bin[j];
            move(p, size);

            int new_bin_idx = floor(p.x / BINSIZE) * dim + floor(p.y / BINSIZE);
            if(new_bin_idx == subdomain_start + i){
                continue;
            }
            if(new_bin_idx >= subdomain_start && new_bin_idx < subdomain_start + num_bins){
                not_leaving.push_back(p);
                bin.erase(bin.begin() + j);
            }else{
                int target_rank = get_rank(new_bin_idx);
                moving_out[target_rank].push_back(p);
                bin.erase(bin.begin() + j);
            }
        }
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {

    // Send particles
    for(auto &it: send_bins){
        int tar_rank = it.first;
        auto &bins_idx = it.second;

        auto &parts_to_send = send_parts[tar_rank];
        parts_to_send.clear(); 

        int len = 0; 
        for(int i = 0; i < bins_idx.size(); ++i){
            int bin_idx = bins_idx[i];

            auto &bin = bins[bin_idx - subdomain_size];
            len += bin.size();
            for(int j = 0; j < bin.size(); ++j)
                parts_to_send.push_back(bin[j]);
        }
        auto &reqs = send_req[tar_rank];
        send_particles(tar_rank, parts_to_send, reqs, len);
    }

    // Receive particles
    for(auto &it: receive_bins){
        int cur_rank = it.first;
        receive_particles(cur_rank);
    }

    // Wait for sending particles
    for(auto &it: send_req){
        auto &req = it.second;

        MPI_Wait(&req[0], MPI_STATUS_IGNORE);
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);
    }

    // Clear neighbor bins
    for(int i = subdomain_size; i < subdomain_start; ++i){
        bins[i - subdomain_size].clear();
    }
    for(int i = subdomain_start + num_bins; i < bin_dim + 1; ++i){
        bins[i - subdomain_size].clear();
    }

    for(auto &it: receive_req){
        int cur_rank = it.first;
        auto &req = it.second;
        MPI_Wait(&req[1], MPI_STATUS_IGNORE);

        // Rebin the receive particles
        auto &parts_to_receive = receive_parts[cur_rank];
        rebin_particle(parts_to_receive);
    }

    // Set acceleration to 0
    for(int i = 0; i < num_bins; ++i){
        auto &bin = bins[i + (subdomain_start - subdomain_size)];
        for(particle_t &p: bin){
            p.ax = p.ay = 0;
        }
    }

    // Compute forces
    for(int i = 0; i < num_bins; ++i){
        auto &bin = bins[i + (subdomain_start - subdomain_size)];
        int bi = (subdomain_start + i) / dim;
        int bj = (subdomain_start + i) % dim;

        for(int d = 0; d < 9; ++d){
            int neigh_bi = bi + dir[d][0];
            int neigh_bj = bj + dir[d][1];
            if(neigh_bi < 0 or neigh_bi >= dim or neigh_bj < 0 or neigh_bj >= dim)
                continue;
            int neigh_idx = neigh_bi * dim + neigh_bj;
            auto &neigh_bin = bins[neigh_idx - subdomain_size];

            for(particle_t &p1: bin){
                for(particle_t &p2: neigh_bin){
                    apply_force(p1, p2);
                }
            }
        }
    }

    // get particles moving out and not leaving
    map<int, vector<particle_t>> moving_out;
    vector<particle_t> not_leaving;
    // move the particles
    // (Could be faster)
    // move_particle(moving_out, not_leaving, size);
    for(int i = 0; i < num_bins; ++i){
        auto &bin = bins[i + (subdomain_start - subdomain_size)];

        // remove particle
        for(int j = bin.size()-1; j >= 0; --j){
            particle_t &p = bin[j];
            move(p, size);

            int new_bin_idx = floor(p.x / BINSIZE) * dim + floor(p.y / BINSIZE);
            if(new_bin_idx == subdomain_start + i){
                continue;
            }
            if(new_bin_idx >= subdomain_start && new_bin_idx < subdomain_start + num_bins){
                not_leaving.push_back(p);
                bin.erase(bin.begin() + j);
            }else{
                int target_rank = get_rank(new_bin_idx);
                moving_out[target_rank].push_back(p);
                bin.erase(bin.begin() + j);
            }
        }
    }
    // rebin particles that are not leaving
    rebin_particle(not_leaving);
    // send and receive particles for moving out
    MPI_ALL(num_procs, moving_out);
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.

    int num_parts_to_send = 0;
    send_part_all.clear();

    for(int i = 0; i < num_bins; ++i){
        int n = bins[i + (subdomain_start - subdomain_size)].size();
        num_parts_to_send += n;
        for(int j = 0; j < n; ++j){
            send_part_all.push_back(bins[i + (subdomain_start - subdomain_size)][j]);
        }
    }

    if(rank == 0){
        MPI_Gather(&num_parts_to_send, 1, MPI_INT, receive_part_num.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    }else{
        MPI_Gather(&num_parts_to_send, 1, MPI_INT, nullptr, 0, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if(rank == 0){
        int receive_nums = 0;
        for (int nums: receive_part_num)
            receive_nums += nums;
        receive_part_all.resize(receive_nums);

        MPI_Gatherv(send_part_all.data(), send_part_all.size(), PARTICLE, receive_part_all.data(),
                receive_part_num.data(), get_size(receive_part_num).data(), PARTICLE, 0, MPI_COMM_WORLD);

        // order the particles
        for(particle_t &p: receive_part_all){
            parts[p.id-1] = p;
        }
    }else{
        MPI_Gatherv(send_part_all.data(), send_part_all.size(), PARTICLE, nullptr, nullptr, nullptr, PARTICLE, 0, MPI_COMM_WORLD);
    }
}
