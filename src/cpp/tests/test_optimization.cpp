#include <doctest/doctest.h>
#include "gm/optimization.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>

namespace {
gm::Json tiny_space() {
    gm::Json j = gm::Json::object();
    auto p = gm::Parameters{}.json();
    for (const auto& name : gm::parameter_names) j[name] = gm::Json::array({p.at(name)});
    j["baseline_probability"] = {0.05, 0.35};
    return j;
}
gm::Subject tiny_subject() {
    gm::Subject d;
    d.fingerprint = "optimizer-unit-subject";
    d.n_categories = 2;
    d.stimuli.resize(2);
    d.choices = {0, 0};
    d.score_mask = {1, 1};
    return d;
}
gm::Json tiny_config() {
    return {{"initial_starts",2},{"rounds",0},{"search_particles",2},{"search_seeds",2},
        {"rescore_top",2},{"rescore_diverse",0},{"rescore_particles",3},{"rescore_seeds",2},
        {"audit_top",2},{"audit_particles",4},{"audit_seeds",2},{"model_version","fake-v1"}};
}
gm::RunResult result(const gm::Subject& d, double p) {
    return {{{"category_probabilities",{{2,2},std::vector<double>{p,1-p,p,1-p}}},
             {"choice_probabilities",{{2},std::vector<double>{p,p}}},
             {"score_mask",{{2},d.score_mask}}},gm::Json::object()};
}
struct Temp {
    std::filesystem::path path;
    Temp() {
        static std::atomic<unsigned> n{0};
        path=std::filesystem::temp_directory_path()/
            ("gm-opt-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(n++));
    }
    ~Temp() { std::error_code e; std::filesystem::remove_all(path,e); }
};
}

TEST_CASE("optimizer averages probability before log and preserves zero") {
    CHECK(gm::mean_nll({{0.1,0.2},{0.9,0.8}},{1,1}) == doctest::Approx(-std::log(.5)));
    CHECK(std::isinf(gm::mean_nll({{0.0}},{1})));
    CHECK_THROWS(gm::mean_nll({{1.1}},{1}));
    CHECK_THROWS(gm::mean_nll({{.5}},{0}));
    CHECK_THROWS(gm::mean_nll({{.5}},{2}));
}

TEST_CASE("canonical space and strict settings") {
    auto j=tiny_space();
    CHECK(gm::validate_space(j).size()==10);
    CHECK(gm::parameter_names.size()==10);
    j["retention"]={.8,.8}; CHECK_THROWS(gm::validate_space(j));
    j=tiny_space(); j["capacity"]={2.5}; CHECK_THROWS(gm::validate_space(j));
    j=tiny_space(); j["capacity"]={1,10000}; CHECK_NOTHROW(gm::validate_space(j));
    j=tiny_space(); j["failure_decay"]={0.,1.}; CHECK_NOTHROW(gm::validate_space(j));
    j=tiny_space(); j["similarity_width"]={0.}; CHECK_THROWS(gm::validate_space(j));
    CHECK(gm::optimizer_settings().at("model_version")=="1031-unified-v2:cpp-v1");
    CHECK(std::string(gm::optimizer_rng_identity).find("1031-unified-v2")!=std::string::npos);
    j=tiny_space(); j["beta_init"]={30}; CHECK_THROWS(gm::validate_space(j));
    CHECK_THROWS(gm::optimizer_settings({{"rounds",true}}));
#ifdef _WIN32
    CHECK(gm::optimizer_settings({{"n_jobs",12}}).at("n_jobs")==12);
    CHECK_THROWS(gm::optimizer_settings({{"n_jobs",13}}));
#else
    CHECK(gm::optimizer_settings({{"n_jobs",128}}).at("n_jobs")==128);
#endif
    CHECK_THROWS(gm::optimizer_settings({{"typo",1}}));
}

TEST_CASE("audit chooses winner and cache reuses paired seeds across jobs") {
    Temp t; auto d=tiny_subject(); gm::Catalogue c; c.fingerprint="resource-v1";
    std::atomic<int> calls{0};
    gm::Evaluator evaluator=[&](const gm::Subject& s,const gm::Parameters& p,const gm::Catalogue&,int particles,std::uint64_t,bool keep) {
        ++calls; CHECK_FALSE(keep);
        return result(s, particles==4 ? p.baseline_probability : 1-p.baseline_probability);
    };
    auto config=tiny_config();
    auto a=gm::fit_subject(d,c,tiny_space(),config,t.path,evaluator);
    CHECK(a.at("best_parameters").at("baseline_probability")==.35);
    CHECK(a.at("near_optimal").size()==1);
    CHECK(a.at("candidate_trace").size()==6);
    CHECK(calls==12);
    config["n_jobs"]=2;
    auto b=gm::fit_subject(d,c,tiny_space(),config,t.path,evaluator);
    CHECK(calls==12);
    CHECK(a==b);
    c.fingerprint="resource-v2";
    gm::fit_subject(d,c,tiny_space(),config,t.path,evaluator);
    CHECK(calls==24);
}

TEST_CASE("worker errors are captured and invalid results never published") {
    Temp t; auto d=tiny_subject(); gm::Catalogue c; c.fingerprint="resource-v1";
    auto cfg=tiny_config(); cfg["n_jobs"]=2;
    gm::Evaluator bad=[](const gm::Subject&,const gm::Parameters&,const gm::Catalogue&,int,std::uint64_t,bool)->gm::RunResult {
        throw std::runtime_error("injected-filter-failure");
    };
    CHECK_THROWS_WITH_AS(gm::fit_subject(d,c,tiny_space(),cfg,t.path,bad),doctest::Contains("injected-filter-failure"),std::runtime_error);
    bad=[](const gm::Subject& s,const gm::Parameters&,const gm::Catalogue&,int,std::uint64_t,bool) { return result(s,1.5); };
    CHECK_THROWS(gm::fit_subject(d,c,tiny_space(),cfg,t.path,bad));
    for (const auto& f:std::filesystem::directory_iterator(t.path/"optimization"/"cache")) CHECK(f.path().extension()!=".npz");
}

TEST_CASE("cache identity includes numerical and RNG identities") {
    auto p=gm::Parameters{};
    auto k=gm::cache_key("data","model","resource",p,8,5);
    CHECK(k!=gm::cache_key("data2","model","resource",p,8,5));
    CHECK(k!=gm::cache_key("data","model2","resource",p,8,5));
    CHECK(k!=gm::cache_key("data","model","resource2",p,8,5));
    CHECK(k!=gm::cache_key("data","model","resource",p,9,5));
    CHECK(k!=gm::cache_key("data","model","resource",p,8,6));
    CHECK(gm::stage_seeds("data","search",2,3)!=gm::stage_seeds("data","audit",2,3));
    const auto search=gm::stage_seeds("data","search",8,3);
    const auto rescore=gm::stage_seeds("data","rescore",8,3);
    const auto audit=gm::stage_seeds("data","audit",8,3);
    CHECK(search==gm::stage_seeds("data","search",8,3));
    std::set<std::uint64_t> seeds(search.begin(),search.end());
    seeds.insert(rescore.begin(),rescore.end());seeds.insert(audit.begin(),audit.end());
    CHECK(seeds.size()==24);
    auto values=p.json();
    for(const auto name:gm::parameter_names) {
        auto changed=values;
        if(std::string(name)=="capacity")changed[name]=10000;
        else changed[name]=values.at(name).get<double>()+.001;
        CHECK(k!=gm::cache_key("data","model","resource",gm::Parameters::from_json(changed),8,5));
    }
}

TEST_CASE("balanced proposals are unique with local one and two coordinate moves") {
    Temp t;auto d=tiny_subject();gm::Catalogue c;c.fingerprint="resource-v1";
    auto space=tiny_space();space["baseline_probability"]={.05,.15,.35,.65};
    space["retention"]={0.,.3,.6,.8};space["global_search"]={0.,.1,.25,.5};
    auto cfg=tiny_config();cfg["rounds"]=1;cfg["proposals_per_round"]=9;cfg["top_k"]=2;
    gm::Evaluator evaluator=[](const gm::Subject& s,const gm::Parameters&,const gm::Catalogue&,int,std::uint64_t,bool) {return result(s,.5);};
    auto fitted=gm::fit_subject(d,c,space,cfg,t.path,evaluator);
    std::set<std::array<int,10>> seen;std::vector<std::array<int,10>> anchors;std::map<std::string,int> modes;
    for(const auto& record:fitted.at("candidate_trace"))if(record.at("stage")=="search") {
        auto p=record.at("grid_index").get<std::array<int,10>>();CHECK(seen.insert(p).second);
        if(record.at("round")==0) {anchors.push_back(p);continue;}
        auto mode=record.at("proposal_mode").get<std::string>();++modes[mode];
        if(mode=="global")continue;
        bool adjacent=false;
        for(const auto& a:anchors) {
            int changed=0;bool local=true;
            for(int j=0;j<10;++j)if(a[j]!=p[j]) {++changed;local=local && std::abs(a[j]-p[j])==1;}
            adjacent=adjacent || (local && changed==(mode=="single_neighbor"?1:2));
        }
        CHECK(adjacent);
    }
    CHECK(modes["single_neighbor"]==3);
    CHECK(modes["double_coordinate"]==3);
    CHECK(modes["global"]==3);
}

TEST_CASE("singleton grid caps all stages and zero probability cannot win") {
    Temp t;auto d=tiny_subject();gm::Catalogue c;c.fingerprint="resource-v1";
    auto space=tiny_space();space["baseline_probability"]={.05};auto cfg=tiny_config();
    gm::Evaluator evaluator=[](const gm::Subject& s,const gm::Parameters&,const gm::Catalogue&,int,std::uint64_t,bool) {return result(s,.5);};
    auto fitted=gm::fit_subject(d,c,space,cfg,t.path,evaluator);
    CHECK(fitted.at("candidate_trace").size()==3);
    CHECK(fitted.at("stage_budgets").at("audit").at("evaluated_candidates")==1);
    cfg["model_version"]="zero-v1";
    evaluator=[](const gm::Subject& s,const gm::Parameters&,const gm::Catalogue&,int,std::uint64_t,bool) {return result(s,0);};
    CHECK_THROWS_WITH_AS(gm::fit_subject(d,c,space,cfg,t.path,evaluator),doctest::Contains("no finite"),std::runtime_error);
}

TEST_CASE("nearoptimal is an audited mean NLL tolerance") {
    Temp t;auto d=tiny_subject();gm::Catalogue c;c.fingerprint="resource-v1";auto cfg=tiny_config();
    cfg["nearoptimal_tolerance"]=3.;
    gm::Evaluator evaluator=[](const gm::Subject& s,const gm::Parameters& p,const gm::Catalogue&,int,std::uint64_t,bool) {return result(s,p.baseline_probability);};
    auto fitted=gm::fit_subject(d,c,tiny_space(),cfg,t.path,evaluator);
    CHECK(fitted.at("near_optimal").size()==2);
    for(const auto& record:fitted.at("near_optimal"))CHECK(record.at("stage")=="audit");
}

TEST_CASE("fitted objective scores the mean forecast, not mean seed NLL") {
    Temp t;auto d=tiny_subject();gm::Catalogue c;c.fingerprint="resource-v1";
    auto cfg=tiny_config();auto first=gm::stage_seeds(d.fingerprint,"audit",2).front();
    gm::Evaluator evaluator=[first](const gm::Subject& s,const gm::Parameters&,const gm::Catalogue&,int particles,std::uint64_t seed,bool) {
        return result(s,particles==4?(seed==first?.1:.9):.5);
    };
    auto fitted=gm::fit_subject(d,c,tiny_space(),cfg,t.path,evaluator);
    CHECK(fitted.at("mean_nll").get<double>()==doctest::Approx(-std::log(.5)));
    CHECK(fitted.at("best_parameters").at("baseline_probability")==.05);
}

TEST_CASE("cache fingerprint includes score mask even with unchanged upstream label") {
    Temp t;auto d=tiny_subject();gm::Catalogue c;c.fingerprint="resource-v1";auto cfg=tiny_config();
    std::atomic<int> calls{0};
    gm::Evaluator evaluator=[&](const gm::Subject& s,const gm::Parameters&,const gm::Catalogue&,int,std::uint64_t,bool) {++calls;return result(s,.5);};
    gm::fit_subject(d,c,tiny_space(),cfg,t.path,evaluator);CHECK(calls==12);
    d.score_mask={1,0};
    gm::fit_subject(d,c,tiny_space(),cfg,t.path,evaluator);CHECK(calls==24);
}
