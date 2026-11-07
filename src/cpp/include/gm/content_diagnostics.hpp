#pragma once
#include "model.hpp"

namespace gm {
std::vector<unsigned> catalogue_feature_masks(const Catalogue&);
unsigned content_feature_mask(const Slot&,const std::vector<unsigned>&);
void allocate_particle_contents(Arrays&,std::size_t trials,int particles,int capacity,int categories,int condition);
void record_particle_contents(Arrays&,std::size_t trial,const std::vector<ParticleState>&,
    const std::vector<TrialTape>&,const std::vector<double>& predictive,
    const std::vector<double>& filtered,const std::vector<unsigned>& rule_masks);
}
