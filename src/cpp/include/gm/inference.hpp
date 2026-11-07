#pragma once
#include "types.hpp"
namespace gm {
// Stable catalogue identities; separate from the four semantic-kind columns.
inline constexpr std::array<const char*,6> catalogue_state_names{
    "catalogue_prior", "catalogue_filtered_prior", "catalogue_posterior",
    "catalogue_predictive_execution_probability", "catalogue_execution_probability",
    "catalogue_active_probability"};
RunResult particle_filter(const Subject&, const Parameters&, const Catalogue&, int particle_count,
                          std::uint64_t seed, bool keep_states=false, double ess_fraction=.5,
                          bool export_particles=false);
RunResult simulate(const Subject&, const Parameters&, const Catalogue&, std::uint64_t seed,
                   bool keep_states=false);
// Pure geometric entry point for frozen reference tests.
std::vector<double> category_distances(const Catalogue&, int hypothesis,
                                     const std::array<double,4>& stimulus);
}
