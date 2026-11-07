#pragma once
#include "inference.hpp"
#include <functional>
#include <limits>
#include <string_view>

namespace gm {
inline constexpr std::array<const char*,10> parameter_names{
    "capacity", "similarity_width", "baseline_probability", "error_gain", "failure_decay",
    "global_search", "retention", "execution_inertia", "beta_init", "learning_rate"};
inline constexpr const char* optimizer_rng_identity = "1031-unified-v2:gm-mt19937_64-role-seeds-v1:optimizer-v1";
#ifdef _WIN32
inline constexpr int maximum_jobs=12;
#else
inline constexpr int maximum_jobs=std::numeric_limits<int>::max()/32;
#endif
using Evaluator = std::function<RunResult(const Subject&,const Parameters&,const Catalogue&,
                                        int,std::uint64_t,bool)>;
Json validate_space(const Json&);
Json optimizer_settings(const Json& config = Json::object());
double mean_nll(const std::vector<std::vector<double>>&,const std::vector<std::uint8_t>&);
std::vector<std::uint64_t> stage_seeds(const std::string& fingerprint,const std::string& stage,
                                     int count,std::uint64_t root=0);
std::string cache_key(const std::string& data,const std::string& model,const std::string& resource,
                      const Parameters&,int particles,std::uint64_t seed);
Json fit_subject(const Subject&,const Catalogue&,const Json& space,const Json& config,
                 const std::filesystem::path& output);
// Callback must be safe for concurrent invocation; explicit model_version is mandatory.
Json fit_subject(const Subject&,const Catalogue&,const Json& space,const Json& config,
                 const std::filesystem::path& output,const Evaluator&);
// Development diagnostic command "filter" is intentionally omitted from public help.
int cli_main(const std::vector<std::string>& arguments);

// Offline evaluation schema gm-evaluation-inputs-v1 (no Python runtime launched).
// NPZ: conditional_category_probability_repeats[R,T,C],
// conditional_category_probabilities/conditional_response_probabilities[T,C],
// conditional_choice_probability_repeats[R,T], conditional_choice_probabilities[T],
// conditional_score_mask[T]; observed_choices/observed_feedback/true_categories[T],
// category_to_response[T,C] (identity row), valid/score_mask[T], trial_ids[T,3],
// session_ids[T], stimuli[T,4], autonomous_choices/feedback/valid[S,T],
// autonomous_category_probabilities[S,T,C], conditional_<state>[T,...],
// autonomous_diagnostic_<state>[T,...]; pf_seeds[R], simulation_seeds[S],
// band_seed scalar, autonomous_diagnostic_seed scalar (all seeds int64, <= INT64_MAX).
// Manifest includes schema_version, subject_id, condition, n_categories, parameters,
// config, data_fingerprint, resource_fingerprint, provenance, oral_reports,
// perception, catalogue_metadata, pf_seeds, simulation_seeds, band_seed, state_keys, arrays[name].shape,
// inputs_sha256. Beta is weighted across repeats by active_probability.
}
