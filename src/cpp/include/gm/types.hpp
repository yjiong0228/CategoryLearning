#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <iosfwd>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace gm {
using Json = nlohmann::json;
using Shape = std::vector<std::size_t>;
struct Array {
    Shape shape;
    std::variant<std::vector<double>, std::vector<std::int64_t>, std::vector<std::uint8_t>> data;
    std::vector<double> doubles() const;
};
using Arrays = std::map<std::string, Array>;
struct Parameters {
    int capacity=4;
    double similarity_width=.25, baseline_probability=.15, error_gain=1, failure_decay=.8;
    double global_search=.2, retention=.8, execution_inertia=1, beta_init=5, learning_rate=.1;
    void validate() const;
    Json json() const;
    static Parameters from_json(const Json&);
};
struct Subject {
    int id=0, condition=0, n_categories=0;
    std::vector<std::array<double,4>> stimuli;
    std::vector<std::int64_t> choices, true_categories, session_ids;
    std::vector<std::array<std::int64_t,3>> trial_ids;
    std::vector<double> feedback;
    std::vector<std::uint8_t> valid, score_mask;
    std::array<double,4> perception_mean{}, perception_scale{};
    std::string perception_mode, fingerprint;
    Json oral_reports=Json::array(), protocol;
    std::size_t size() const { return stimuli.size(); }
};
struct Polytope { Eigen::MatrixXd A; Eigen::VectorXd b; };
struct Rule { std::vector<std::vector<Polytope>> categories; Json metadata; };
struct Catalogue {
    int condition=0, n_categories=0;
    std::vector<Rule> rules;
    std::vector<double> functional_distance, edit_costs;
    std::string fingerprint;
    std::size_t size() const { return rules.size(); }
};
struct RunResult { Arrays arrays; Json summary=Json::object(); Arrays particle_contents; };
std::string sha256(const std::string&);
std::string file_sha256(const std::filesystem::path&);
Json read_json(const std::filesystem::path&);
void write_json_new(const std::filesystem::path&, const Json&);
Arrays read_npz(const std::filesystem::path&);
void write_npz_new(const std::filesystem::path&, const Arrays&);
namespace detail { void encode_npz(std::ostream&, const Arrays&); }
Json read_yaml(const std::filesystem::path&);
Subject load_subject(const std::filesystem::path& data_root, const std::string& dataset,
                     int subject_id, int max_valid_trials=0);
std::shared_ptr<const Catalogue> load_catalogue(const std::filesystem::path& resource_root, int condition);
double task_feedback(const Subject&, std::size_t trial, int choice);
Json provenance(const Catalogue&);
}
