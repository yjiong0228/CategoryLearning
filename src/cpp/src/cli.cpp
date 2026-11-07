#include "gm/optimization.hpp"
#include "gm/rng.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <set>
#include <type_traits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace gm {
namespace {
const std::set<std::string> state_names{
    "prior","filtered_prior","posterior","beta","active_probability","choice_weights",
    "dominant_probability","failure_pressure","search_probability","global_probability",
    "active_count","retained_count","newcomer_count","dropped_count","newcomer_mass",
    "fallback","proposal_source","pairing"};

void known_keys(const Json& value,const std::set<std::string>& allowed,const std::string& context) {
    if(!value.is_object())throw std::invalid_argument(context+" must be a mapping");
    for(auto it=value.begin();it!=value.end();++it)if(!allowed.count(it.key()))throw std::invalid_argument("unknown "+context+" key: "+it.key());
}
std::int64_t integer(const Json& v,const std::string& name,std::int64_t minimum=1,
                     std::int64_t maximum=std::numeric_limits<int>::max()) {
    if(!v.is_number_integer() || (v.is_number_unsigned() && v.get<std::uint64_t>()>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())))
        throw std::invalid_argument(name+" must be an integer");
    auto n=v.get<std::int64_t>();
    if(n<minimum || n>maximum)throw std::invalid_argument(name+" outside supported range");
    return n;
}
std::string text_field(const Json& v,const std::string& name) {
    if(!v.is_string() || v.get<std::string>().empty())throw std::invalid_argument(name+" must be a nonempty string");
    return v.get<std::string>();
}
Parameters validated_parameters(const Json& input) {
    std::set<std::string> names(parameter_names.begin(),parameter_names.end());
    known_keys(input,names,"parameters");
    auto values=Parameters{}.json();values.update(input);
    Json singletons=Json::object();
    for(const auto name:parameter_names)singletons[name]=Json::array({values.at(name)});
    validate_space(singletons);
    return Parameters::from_json(values);
}
void new_destination(const std::filesystem::path& destination) {
    if(std::filesystem::exists(destination))throw std::runtime_error("output already exists: "+destination.string());
    std::filesystem::create_directories(destination.parent_path());
    if(!std::filesystem::create_directory(destination))throw std::runtime_error("output creation race");
}
void prepare_fit_destination(const std::filesystem::path& destination,const Json& manifest,bool resume) {
    if(std::filesystem::exists(destination)) {
        if(!resume)throw std::runtime_error("output already exists: "+destination.string());
        if(std::filesystem::exists(destination/"best_parameters.json"))throw std::runtime_error("fit already completed; refusing overwrite");
        if(!std::filesystem::exists(destination/"fit_manifest.json") || read_json(destination/"fit_manifest.json")!=manifest)throw std::invalid_argument("resume requires matching data/model/resources/space/budget/RNG manifest");
    } else {
        new_destination(destination);
        write_json_new(destination/"fit_manifest.json",manifest);
    }
}
std::int64_t parse_integer(const std::string& s,const std::string& name) {
    std::int64_t n=0;auto [end,error]=std::from_chars(s.data(),s.data()+s.size(),n);
    if(error!=std::errc{} || end!=s.data()+s.size())throw std::invalid_argument(name+" must be an integer");
    return n;
}
std::filesystem::path relative(const std::filesystem::path& config,const std::string& value) {
    std::filesystem::path p(value);
    return std::filesystem::absolute(p.is_absolute()?p:config.parent_path()/p).lexically_normal();
}
std::size_t elements(const Shape& shape) {
    std::size_t size=1;
    for(auto n:shape) {
        if(n && size>std::numeric_limits<std::size_t>::max()/n)throw std::invalid_argument("array shape overflow");
        size*=n;
    }
    return size;
}
void validate_array(const Array& a,const std::string& name) {
    if(std::visit([](const auto& v){return v.size();},a.data)!=elements(a.shape))throw std::invalid_argument(name+": shape/storage mismatch");
}
Array stack(const std::vector<RunResult>& runs,const std::string& key,const Shape& expected) {
    if(runs.empty())throw std::invalid_argument("cannot stack empty repeat list");
    const auto& first=runs.front().arrays.at(key);validate_array(first,key);
    if(first.shape!=expected)throw std::invalid_argument(key+": repeat shape mismatch");
    Array out=first;out.shape.insert(out.shape.begin(),runs.size());
    std::visit([](auto& data){data.clear();},out.data);
    for(const auto& run:runs) {
        const auto& a=run.arrays.at(key);validate_array(a,key);
        if(a.shape!=expected || a.data.index()!=first.data.index())throw std::invalid_argument(key+": repeats disagree on shape or dtype");
        std::visit([&](auto& data){const auto& source=std::get<std::decay_t<decltype(data)>>(a.data);data.insert(data.end(),source.begin(),source.end());},out.data);
    }
    return out;
}
Array average(const std::vector<RunResult>& runs,const std::string& key,const Shape& shape) {
    auto stacked=stack(runs,key,shape).doubles();std::vector<double> mean(elements(shape),0);
    for(std::size_t i=0;i<mean.size();++i)for(std::size_t r=0;r<runs.size();++r)mean[i]+=stacked[r*mean.size()+i]/runs.size();
    return {shape,std::move(mean)};
}
Arrays states_of(const RunResult& run,std::size_t trials) {
    Arrays states;
    const std::set<std::string> nonstates{"category_probabilities","choice_probabilities","score_mask","effective_sample_size","choices","feedback","valid"};
    for(const auto& [name,a]:run.arrays) {
        if(nonstates.count(name)||name=="catalogue_rule_ids"||name=="feature_probability"||name=="feature_set_probability"||
                std::any_of(catalogue_state_names.begin(),catalogue_state_names.end(),[&](auto key){return name==key;}))continue;
        if(!state_names.count(name))throw std::invalid_argument("unrecognized diagnostic state: "+name);
        validate_array(a,name);
        if(a.shape.empty() || a.shape.front()!=trials)throw std::invalid_argument(name+": state does not align with trials");
        const bool matrix=name=="prior" || name=="filtered_prior" || name=="posterior" || name=="beta" ||
            name=="active_probability" || name=="choice_weights" || name=="dominant_probability" || name=="proposal_source";
        const Shape expected=name=="pairing"?Shape{trials,4,3}:matrix?Shape{trials,4}:Shape{trials};
        if(a.shape!=expected)throw std::invalid_argument(name+": diagnostic shape differs from semantic-kind contract");
        states.emplace(name,a);
    }
    return states;
}
Arrays mean_filter_states(const std::vector<RunResult>& runs,std::size_t trials) {
    if(runs.empty())throw std::invalid_argument("PF repeats are empty");
    auto first=states_of(runs.front(),trials);std::vector<Arrays> repeats;
    for(const auto& run:runs) {
        auto states=states_of(run,trials);
        if(states.size()!=first.size())throw std::invalid_argument("PF repeats disagree on state keys");
        for(const auto& [name,a]:first)if(!states.count(name) || states.at(name).shape!=a.shape)throw std::invalid_argument("PF repeats disagree on state shapes/keys");
        repeats.push_back(std::move(states));
    }
    Arrays out;
    for(const auto& [name,a]:first) {
        std::vector<double> sum(elements(a.shape),0),count(sum.size(),0);
        for(const auto& run:repeats) {
            auto values=run.at(name).doubles();
            for(std::size_t i=0;i<sum.size();++i)if(std::isfinite(values[i])) {sum[i]+=values[i];count[i]+=1;}
        }
        for(std::size_t i=0;i<sum.size();++i)sum[i]=count[i]?sum[i]/count[i]:std::numeric_limits<double>::quiet_NaN();
        out[name]={a.shape,std::move(sum)};
    }
    if(first.count("beta")) {
        if(!first.count("active_probability") || first.at("active_probability").shape!=first.at("beta").shape)throw std::invalid_argument("beta averaging requires matching active_probability");
        auto presence=out.at("active_probability").doubles();std::vector<double> numerator(presence.size(),0),denominator(presence.size(),0);
        for(const auto& run:repeats) {
            auto p=run.at("active_probability").doubles(),b=run.at("beta").doubles();
            for(std::size_t i=0;i<p.size();++i)if(std::isfinite(p[i]) && p[i]>0 && std::isfinite(b[i])) {numerator[i]+=p[i]*b[i];denominator[i]+=p[i];}
        }
        for(std::size_t i=0;i<numerator.size();++i)numerator[i]=!std::isfinite(presence[i])?std::numeric_limits<double>::quiet_NaN():denominator[i]>0?numerator[i]/denominator[i]:0;
        out["beta"]={first.at("beta").shape,std::move(numerator)};
    }
    return out;
}
Arrays mean_catalogue_states(const std::vector<RunResult>& runs,std::size_t trials,std::size_t rules) {
    if(runs.empty()||!rules)throw std::invalid_argument("Catalogue diagnostics require rules and runs");
    for(const auto& run:runs) {
        const auto& ids=run.arrays.at("catalogue_rule_ids");
        const auto* values=std::get_if<std::vector<std::int64_t>>(&ids.data);
        if(ids.shape!=Shape{rules}||!values||values->size()!=rules)throw std::invalid_argument("Invalid catalogue rule IDs");
        for(std::size_t h=0;h<rules;++h)if((*values)[h]!=static_cast<std::int64_t>(h))throw std::invalid_argument("Catalogue ID order differs");
    }
    Arrays out;
    for(auto name:catalogue_state_names) {
        std::vector<double> sum(trials*rules,0.);std::vector<int> count(sum.size(),0);
        for(const auto& run:runs) {
            const auto& a=run.arrays.at(name);validate_array(a,name);
            const auto* v=std::get_if<std::vector<double>>(&a.data);
            if(a.shape!=Shape{trials,rules}||!v)throw std::invalid_argument("Invalid catalogue diagnostic shape/dtype");
            for(std::size_t j=0;j<v->size();++j)if(std::isfinite((*v)[j])) {sum[j]+=(*v)[j];++count[j];}
        }
        for(std::size_t j=0;j<sum.size();++j)sum[j]=count[j]?sum[j]/count[j]:std::numeric_limits<double>::quiet_NaN();
        out[name]={{trials,rules},std::move(sum)};
    }
    return out;
}
Json evaluation_settings(const Json& config) {
    Json defaults={{"particle_count",32},{"pf_repeats",1},{"simulation_repeats",16},{"band_draws",256},
                   {"window",32},{"seed",0},{"jobs",1},{"plots",true},{"keep_states",false},{"export_particles",false}};
    std::set<std::string> allowed;for(auto it=defaults.begin();it!=defaults.end();++it)allowed.insert(it.key());
    known_keys(config,allowed,"evaluation");
    for(auto it=config.begin();it!=config.end();++it)defaults[it.key()]=it.value();
    for(auto it=defaults.begin();it!=defaults.end();++it) {
        if(it.key()=="plots" || it.key()=="keep_states" || it.key()=="export_particles") {if(!it.value().is_boolean())throw std::invalid_argument(it.key()+" must be boolean");}
        else if(it.key()=="seed")integer(it.value(),it.key(),0,std::numeric_limits<std::int64_t>::max());
        else integer(it.value(),it.key(),1,it.key()=="jobs"?maximum_jobs:std::numeric_limits<int>::max()/4);
    }
    return defaults;
}
void probability_arrays(const RunResult& run,const Subject& data,bool filter) {
    const auto n=data.size(),c=static_cast<std::size_t>(data.n_categories);
    const auto& a=run.arrays.at("category_probabilities");validate_array(a,"category_probabilities");
    if(a.shape!=Shape{n,c})throw std::invalid_argument("category_probabilities shape mismatch");
    auto values=a.doubles();
    const auto& mask=filter?data.valid:std::get<std::vector<std::uint8_t>>(run.arrays.at("valid").data);
    if(mask.size()!=n)throw std::invalid_argument("valid mask length mismatch");
    for(std::size_t t=0;t<n;++t)if(mask[t]) {
        double sum=0;for(std::size_t j=0;j<c;++j) {
            auto v=values[t*c+j];if(!std::isfinite(v) || v<0 || v>1)throw std::invalid_argument("invalid forecast probability");sum+=v;
        }
        if(std::abs(sum-1)>1.01e-7)throw std::invalid_argument("forecast probabilities do not normalize");
    }
    if(filter) {
        const auto& m=run.arrays.at("score_mask");validate_array(m,"score_mask");
        if(m.shape!=Shape{n} || !std::holds_alternative<std::vector<std::uint8_t>>(m.data) || std::get<std::vector<std::uint8_t>>(m.data)!=data.score_mask)throw std::invalid_argument("PF score mask differs from observed score mask");
        const auto& p=run.arrays.at("choice_probabilities");validate_array(p,"choice_probabilities");
        if(p.shape!=Shape{n})throw std::invalid_argument("choice_probabilities shape mismatch");
        auto choice=p.doubles();
        for(std::size_t t=0;t<n;++t)if(data.score_mask[t]) {
            if(data.choices[t]<0 || data.choices[t]>=data.n_categories || !std::isfinite(choice[t]) || choice[t]<0 || choice[t]>1 || std::abs(choice[t]-values[t*c+data.choices[t]])>1e-9)throw std::invalid_argument("PF observed choice probability mismatch");
        }
    } else {
        for(auto key:{"choices","feedback","valid"}) {const auto& v=run.arrays.at(key);validate_array(v,key);if(v.shape!=Shape{n})throw std::invalid_argument(std::string(key)+": simulation shape mismatch");}
        const auto& choices=std::get<std::vector<std::int64_t>>(run.arrays.at("choices").data);
        auto feedback=run.arrays.at("feedback").doubles();
        for(std::size_t t=0;t<n;++t) {
            if(mask[t]>1 || mask[t]!=data.valid[t])throw std::invalid_argument("simulation changed valid schedule");
            if(mask[t] && (choices[t]<0 || choices[t]>=data.n_categories || !std::isfinite(feedback[t])))throw std::invalid_argument("invalid simulated choice/feedback");
        }
    }
}
void validate_subject(const Subject& d) {
    auto n=d.size();
    if(!n || d.n_categories<1 || d.fingerprint.empty())throw std::invalid_argument("subject must contain trials, categories and a fingerprint");
    if(d.choices.size()!=n || d.true_categories.size()!=n || d.feedback.size()!=n || d.session_ids.size()!=n || d.trial_ids.size()!=n || d.valid.size()!=n || d.score_mask.size()!=n)throw std::invalid_argument("subject arrays do not align");
    for(std::size_t t=0;t<n;++t) {
        if(d.valid[t]>1 || d.score_mask[t]>1 || (d.score_mask[t] && !d.valid[t]))throw std::invalid_argument("invalid subject masks");
        if(d.score_mask[t] && (d.choices[t]<0 || d.choices[t]>=d.n_categories))throw std::invalid_argument("scored observed choice outside category range");
        if(d.valid[t])for(double x:d.stimuli[t])if(!std::isfinite(x))throw std::invalid_argument("nonfinite valid stimulus");
    }
}
std::vector<RunResult> run_tasks(const Subject& data,const Catalogue& catalogue,const Parameters& p,
                               const std::vector<std::uint64_t>& filter_seeds,const std::vector<std::uint64_t>& simulation_seeds,
                               int particles,int jobs,bool keep_states,bool export_particles=false) {
    const auto count=filter_seeds.size()+simulation_seeds.size();
    if(count>static_cast<std::size_t>(std::numeric_limits<int>::max()))throw std::invalid_argument("repeat budget overflow");
#ifdef _OPENMP
    if(omp_in_parallel())throw std::invalid_argument("CLI tasks cannot run in a nested OpenMP pool");
#endif
    std::vector<RunResult> results(count);std::vector<std::exception_ptr> errors(count);
    int workers=static_cast<int>(std::min<std::size_t>(jobs,count));
    #pragma omp parallel for schedule(dynamic,1) num_threads(workers)
    for(int i=0;i<static_cast<int>(count);++i) {
        try {
            if(static_cast<std::size_t>(i)<filter_seeds.size())results[i]=particle_filter(data,p,catalogue,particles,filter_seeds[i],keep_states,.5,export_particles&&i==0);
            else {auto r=static_cast<std::size_t>(i)-filter_seeds.size();results[i]=simulate(data,p,catalogue,simulation_seeds[r],keep_states && r==0);}
            probability_arrays(results[i],data,static_cast<std::size_t>(i)<filter_seeds.size());
        } catch(...) {errors[i]=std::current_exception();}
    }
    for(std::size_t i=0;i<count;++i)if(errors[i]) {
        try {std::rethrow_exception(errors[i]);}
        catch(const std::exception& e) {throw std::runtime_error("repeat "+std::to_string(i)+": "+e.what());}
        catch(...) {throw std::runtime_error("repeat "+std::to_string(i)+": nonstandard worker exception");}
    }
    return results;
}
std::vector<std::int64_t> signed_seeds(const std::vector<std::uint64_t>& seeds) {
    std::vector<std::int64_t> out;for(auto s:seeds)out.push_back(static_cast<std::int64_t>(s));return out;
}
Arrays observed_arrays(const Subject& d) {
    auto n=d.size(),c=static_cast<std::size_t>(d.n_categories);
    std::vector<double> stimuli;std::vector<std::int64_t> ids,mapping;
    stimuli.reserve(n*4);ids.reserve(n*3);mapping.reserve(n*c);
    for(std::size_t t=0;t<n;++t) {
        stimuli.insert(stimuli.end(),d.stimuli[t].begin(),d.stimuli[t].end());
        ids.insert(ids.end(),d.trial_ids[t].begin(),d.trial_ids[t].end());
        for(std::size_t j=0;j<c;++j)mapping.push_back(static_cast<std::int64_t>(j));
    }
    return {{"observed_choices",{{n},d.choices}},{"observed_feedback",{{n},d.feedback}},
        {"true_categories",{{n},d.true_categories}},{"category_to_response",{{n,c},std::move(mapping)}},
        {"valid",{{n},d.valid}},{"score_mask",{{n},d.score_mask}},
        {"trial_ids",{{n,3},std::move(ids)}},{"session_ids",{{n},d.session_ids}},{"stimuli",{{n,4},std::move(stimuli)}}};
}
Json base_summary(const Subject& d,const Catalogue& c,const Parameters& p,const Json& config) {
    return {{"subject_id",d.id},{"condition",d.condition},{"n_categories",d.n_categories},{"trials",d.size()},
        {"scientific_model_contract","1031-unified-v2"},
        {"parameters",p.json()},{"config",config},{"data_fingerprint",d.fingerprint},{"resource_fingerprint",c.fingerprint},
        {"provenance",provenance(c)},{"rng",optimizer_rng_identity},
        {"oral_reports",d.oral_reports},{"protocol",d.protocol},
        {"perception",{{"mode",d.perception_mode},{"mean",d.perception_mean},{"scale",d.perception_scale}}}};
}
void write_evaluation(const Subject& d,const Catalogue& c,const Parameters& p,const Json& cfg,const std::filesystem::path& destination) {
    auto seed=cfg.at("seed").get<std::uint64_t>();
    auto pf_seeds=stage_seeds(d.fingerprint,"evaluation-pf",cfg.at("pf_repeats").get<int>(),seed);
    auto sim_seeds=stage_seeds(d.fingerprint,"evaluation-autonomous",cfg.at("simulation_repeats").get<int>(),seed);
    auto band_seed=stage_seeds(d.fingerprint,"evaluation-bands",1,seed).front();
    auto results=run_tasks(d,c,p,pf_seeds,sim_seeds,cfg.at("particle_count").get<int>(),cfg.at("jobs").get<int>(),true,cfg.at("export_particles").get<bool>());
    std::vector<RunResult> filters(std::make_move_iterator(results.begin()),std::make_move_iterator(results.begin()+pf_seeds.size()));
    std::vector<RunResult> simulations(std::make_move_iterator(results.begin()+pf_seeds.size()),std::make_move_iterator(results.end()));
    const auto n=d.size(),categories=static_cast<std::size_t>(d.n_categories);
    auto arrays=observed_arrays(d);
    arrays["conditional_category_probability_repeats"]=stack(filters,"category_probabilities",{n,categories});
    arrays["conditional_category_probabilities"]=average(filters,"category_probabilities",{n,categories});
    arrays["conditional_response_probabilities"]=arrays.at("conditional_category_probabilities");
    arrays["conditional_choice_probability_repeats"]=stack(filters,"choice_probabilities",{n});
    arrays["conditional_choice_probabilities"]=average(filters,"choice_probabilities",{n});
    arrays["conditional_score_mask"]={{n},d.score_mask};
    arrays["conditional_feature_set_probability"]=average(filters,"feature_set_probability",{n,16});
    arrays["conditional_feature_probability"]=average(filters,"feature_probability",{n,4});
    arrays["autonomous_category_probabilities"]=stack(simulations,"category_probabilities",{n,categories});
    for(const auto name:{"choices","feedback","valid"})arrays[std::string("autonomous_")+name]=stack(simulations,name,{n});
    arrays["pf_seeds"]={{pf_seeds.size()},signed_seeds(pf_seeds)};
    arrays["simulation_seeds"]={{sim_seeds.size()},signed_seeds(sim_seeds)};
    arrays["band_seed"]={{},std::vector<std::int64_t>{static_cast<std::int64_t>(band_seed)}};
    arrays["autonomous_diagnostic_seed"]={{},std::vector<std::int64_t>{static_cast<std::int64_t>(sim_seeds.front())}};
    auto states=mean_filter_states(filters,n);
    if(states.empty())throw std::runtime_error("PF keep_states produced no diagnostics");
    auto autonomous_states=states_of(simulations.front(),n);
    if(autonomous_states.empty())throw std::runtime_error("first autonomous path produced no diagnostics");
    Json state_keys=Json::array(),autonomous_keys=Json::array();
    for(const auto& [name,a]:states) {arrays["conditional_"+name]=a;state_keys.push_back(name);}
    for(const auto& [name,a]:autonomous_states) {arrays["autonomous_diagnostic_"+name]=a;autonomous_keys.push_back(name);}
    auto catalogue_states=mean_catalogue_states(filters,n,c.size());
    arrays["catalogue_rule_ids"]=filters.front().arrays.at("catalogue_rule_ids");
    Json catalogue_keys=Json::array();
    for(const auto& [name,a]:catalogue_states) {
        arrays["conditional_"+name]=a;
        arrays["autonomous_diagnostic_"+name]=simulations.front().arrays.at(name);
        catalogue_keys.push_back(name);
    }
    auto effective_config=cfg;effective_config["keep_states"]=true;
    auto manifest=base_summary(d,c,p,effective_config);
    manifest["schema_version"]="gm1031-evaluation-inputs-v1";
    manifest["pf_seeds"]=pf_seeds;manifest["simulation_seeds"]=sim_seeds;manifest["band_seed"]=band_seed;
    manifest["state_keys"]=state_keys;manifest["autonomous_state_keys"]=autonomous_keys;
    manifest["feature_axis"]={{"codes",16},{"bit_order","bit d is protocol.feature_order[d]"},
        {"conditioning","current choice filtered, before current feedback; equal mean across PF seeds"},
        {"uncommitted_mask",0},{"rule_mask","union of nonzero columns of category region constraints"},
        {"scope","dimension-set use; not geometric region IoU"}};
    manifest["conditional_particle_contents"]={{"status","not_requested"}};
    if(cfg.at("export_particles").get<bool>()) {
        auto packet=std::move(filters.front().particle_contents);
        auto observed=observed_arrays(d);for(auto& [name,a]:observed)packet.emplace(name,std::move(a));
        const auto file=destination/"conditional_particle_contents.npz";write_npz_new(file,packet);
        manifest["conditional_particle_contents"]={{"status","available"},{"file",file.filename().string()},
            {"sha256",file_sha256(file)},{"seed",pf_seeds.front()},{"pf_repeat_index",0},
            {"phase","current_choice_filtered_before_feedback_before_resampling"},
            {"scope","one prespecified conditional PF seed; all its particles, not autonomous and not future-data smoothed"},
            {"particle_count",cfg.at("particle_count")},{"capacity",p.capacity},
            {"representation_kind",{"uncommitted","template","catalogue_rule"}},
            {"semantic_kind",{"uncommitted","example","generalization","catalogue_rule"}},
            {"pairing_axis",d.condition==3?3:1},{"inactive","active=0; IDs=-1; floating content fields=NaN; choice weight=0"},
            {"weights","particle_weights conditions on current choice; particle_predictive_weights excludes it"},
            {"identity","slot/uid are particle-local, not cross-particle rule IDs"},{"arrays",Json::object()}};
        for(const auto& [name,a]:packet)manifest["conditional_particle_contents"]["arrays"][name]={{"shape",a.shape},
            {"dtype",a.data.index()==0?"float64":a.data.index()==1?"int64":"uint8"}};
    }
    manifest["catalogue_state_keys"]=catalogue_keys;
    manifest["catalogue_metadata"]=Json::array();
    for(std::size_t h=0;h<c.size();++h)manifest["catalogue_metadata"].push_back({{"index",h},
        {"label","H"+std::to_string(h)},{"family",c.rules[h].metadata.at("family")},
        {"program",c.rules[h].metadata.at("program")}});
    manifest["catalogue_axis"]={{"axis",1},{"rule_count",c.size()},{"identity","stable zero-based catalogue rule ID, not workspace slot or path-local uid"},
        {"catalogue_prior","pre-choice particle weights times pre-feedback content belief"},
        {"catalogue_filtered_prior","current-choice filtered particle weights times pre-feedback content belief"},
        {"catalogue_posterior","current-choice filtered particle weights times post-feedback content belief"},
        {"catalogue_predictive_execution_probability","pre-choice particle weights times realized execution indicator"},
        {"catalogue_execution_probability","current-choice filtered particle weights times realized execution indicator; before feedback conditioning"},
        {"catalogue_active_probability","pre-choice probability of rule presence"},
        {"normalization","templates/uncommitted retain separate mass; catalogue rows need not sum to one"},
        {"skipped_trials","NaN rows; absent rule on a valid trial contributes zero"}};
    manifest["catalogue_state_aggregation"]="equal mean across PF repeats; autonomous_diagnostic is fixed path 0 only; no future-data smoothing";
    manifest["state_metadata"]=Json::array();
    for(const auto kind:{"uncommitted","example","generalization","catalogue_rule"})
        manifest["state_metadata"].push_back({{"name",kind},{"kind",kind}});
    manifest["semantic_axis"]={{"axis",1},{"kinds",{"uncommitted","example","generalization","catalogue_rule"}},
        {"identity","representation kind; contents and workspace uids are path-local"},
        {"active_probability","expected number of active contents of each kind; may exceed one"},
        {"beta","pre-feedback per-active-item mean within kind"},
        {"pairing","filtered joint kind/pairing mass [T,4,3]"},
        {"proposal_source",{"none","encode","drop","rule"}}};
    manifest["category_mapping"]="identity category_to_response[T,C]; probabilities are already response-coordinate forecasts";
    manifest["state_aggregation"]="equal-repeat finite mean; beta=sum(mean_beta*expected_active_count)/sum(expected_active_count); first fixed autonomous path, never selected";
    manifest["seed_scheme"]="role-tagged derive_seed; separate PF/autonomous/band families; nonnegative signed int64";
    manifest["arrays"]=Json::object();
    for(const auto& [name,a]:arrays)manifest["arrays"][name]={{"shape",a.shape},{"dtype",a.data.index()==0?"float64":a.data.index()==1?"int64":"uint8"}};
    manifest["artifacts"]={"evaluation_inputs.npz","evaluation_manifest.json"};
    if(cfg.at("export_particles").get<bool>())manifest["artifacts"].push_back("conditional_particle_contents.npz");
    manifest["content_path"]={{"status","not_available"},{"scope","autonomous path index 0 only; PF summaries are marginal distributions"}};
    if(simulations.front().summary.contains("content_path")) {
        write_json_new(destination/"content_path.json",simulations.front().summary.at("content_path"));
        manifest["artifacts"].push_back("content_path.json");
        manifest["content_path"].update(Json{{"status","available"},{"file","content_path.json"},
            {"path_index",0},{"seed",sim_seeds.front()},{"sha256",file_sha256(destination/"content_path.json")}});
    }
    write_npz_new(destination/"evaluation_inputs.npz",arrays);
    manifest["inputs_sha256"]=file_sha256(destination/"evaluation_inputs.npz");
    write_json_new(destination/"evaluation_manifest.json",manifest);
}
struct Arguments {
    std::string command;
    std::filesystem::path config,output,fit_dir;
    Json subjects,jobs,max_trials;
    bool smoke=false,resume=false;
};
Arguments parse_arguments(const std::vector<std::string>& args) {
    if(args.empty())throw std::invalid_argument("command required: fit, simulate, evaluate");
    Arguments out;out.command=args.front();
    if(out.command!="fit" && out.command!="simulate" && out.command!="evaluate" && out.command!="filter")throw std::invalid_argument("unknown command: "+out.command);
    std::set<std::string> seen;
    for(std::size_t i=1;i<args.size();++i) {
        const auto& option=args[i];
        if(!seen.insert(option).second)throw std::invalid_argument("duplicate option: "+option);
        if(option=="--smoke") {out.smoke=true;continue;}
        if(option=="--resume") {out.resume=true;continue;}
        if(option!="--config" && option!="--subjects" && option!="--jobs" && option!="--max-trials" && option!="--output" && option!="--fit-dir")throw std::invalid_argument("unknown option: "+option);
        if(i+1>=args.size() || args[i+1].starts_with("--"))throw std::invalid_argument("missing value for "+option);
        if(option=="--subjects") {
            out.subjects=Json::array();
            while(i+1<args.size() && !args[i+1].starts_with("--"))out.subjects.push_back(parse_integer(args[++i],option));
        } else {
            const auto& value=args[++i];
            if(value.empty())throw std::invalid_argument("empty value for "+option);
            if(option=="--config")out.config=value;
            else if(option=="--output")out.output=value;
            else if(option=="--fit-dir")out.fit_dir=value;
            else if(option=="--jobs")out.jobs=parse_integer(value,option);
            else out.max_trials=parse_integer(value,option);
        }
    }
    if(out.config.empty())throw std::invalid_argument("--config is required");
    if(out.resume && out.command!="fit")throw std::invalid_argument("--resume is supported only for fit");
    if(!out.fit_dir.empty() && out.command=="fit")throw std::invalid_argument("--fit-dir is not supported for fit");
    return out;
}
}

int cli_main(const std::vector<std::string>& args) {
    if(args==std::vector<std::string>{"--help"} || (args.size()==2 && args[1]=="--help")) {
        std::cout<<"Usage: genmodel_cpp {fit|simulate|evaluate} --config FILE [--subjects ID ...] [--jobs N] [--max-trials N] [--output DIR] [--fit-dir DIR] [--smoke] [--resume]\n"
            <<"evaluate writes inputs for offline Python evaluation; it never launches Python.\n";
        return 0;
    }
    auto a=parse_arguments(args);a.config=std::filesystem::absolute(a.config);
    auto cfg=read_yaml(a.config);
    known_keys(cfg,{"dataset","data_root","resource_root","subjects","parameters","parameter_space","seed","jobs","max_trials","fit","evaluation","simulation_repeats","output"},"config");
    auto dataset=text_field(cfg.at("dataset"),"dataset");
    auto subjects=a.subjects.is_null()?cfg.at("subjects"):a.subjects;
    if(!subjects.is_array() || subjects.empty())throw std::invalid_argument("subjects must be a nonempty unique integer list");
    std::set<int> ids;for(const auto& s:subjects)if(!ids.insert(static_cast<int>(integer(s,"subject ID"))).second)throw std::invalid_argument("duplicate subject ID");
    int jobs=static_cast<int>(integer(a.jobs.is_null()?cfg.value("jobs",Json(1)):a.jobs,"jobs",1,maximum_jobs));
    auto seed=integer(cfg.value("seed",Json(0)),"seed",0,std::numeric_limits<std::int64_t>::max());
    int max_trials=0;
    if(!a.max_trials.is_null() || cfg.contains("max_trials"))max_trials=static_cast<int>(integer(a.max_trials.is_null()?cfg.at("max_trials"):a.max_trials,"max_trials"));
    if(a.smoke)max_trials=max_trials?std::min(max_trials,16):16;
    auto output=a.output.empty()?relative(a.config,text_field(cfg.value("output",Json("../results/run")),"output")):std::filesystem::absolute(a.output);
    auto data_root=relative(a.config,text_field(cfg.at("data_root"),"data_root"));
    auto resource_root=relative(a.config,text_field(cfg.value("resource_root",Json("../resources")),"resource_root"));
    auto eval=evaluation_settings(cfg.value("evaluation",Json::object()));eval["seed"]=seed;eval["jobs"]=jobs;
    auto fit=optimizer_settings(cfg.value("fit",Json::object()));fit["seed"]=seed;fit["n_jobs"]=jobs;
    auto repeats=integer(cfg.value("simulation_repeats",Json(256)),"simulation_repeats",1,std::numeric_limits<int>::max()/4);
    auto parameters=validated_parameters(cfg.value("parameters",Json::object()));
    Json space;
    if(cfg.contains("parameter_space"))space=validate_space(read_yaml(relative(a.config,text_field(cfg.at("parameter_space"),"parameter_space"))));
    if(a.command=="fit" && space.is_null())throw std::invalid_argument("fit requires parameter_space");
    if(a.smoke) {
        Json small={{"initial_starts",2},{"rounds",1},{"proposals_per_round",4},{"top_k",2},{"search_particles",4},{"search_seeds",1},
            {"rescore_top",2},{"rescore_diverse",1},{"rescore_particles",8},{"rescore_seeds",1},{"audit_top",2},{"audit_particles",8},{"audit_seeds",2}};
        fit.update(small);
        eval.update(Json{{"particle_count",8},{"pf_repeats",1},{"simulation_repeats",4},{"band_draws",32},{"window",8}});repeats=4;
    }
    for(const auto& id_json:subjects) {
        int id=id_json.get<int>();auto d=load_subject(data_root,dataset,id,max_trials);validate_subject(d);
        if(d.id!=id)throw std::invalid_argument("loaded subject ID mismatch");
        auto catalogue=load_catalogue(resource_root,d.condition);
        if(!catalogue || catalogue->n_categories!=d.n_categories || catalogue->fingerprint.empty())throw std::invalid_argument("catalogue and subject categories differ or resource fingerprint missing");
        auto prov=provenance(*catalogue);auto model=prov.at("scientific_fingerprint").get<std::string>();
        auto destination=output/("subject_"+std::to_string(id));
        auto p=parameters;
        if(!a.fit_dir.empty()) {
            auto fitted=read_json(a.fit_dir/("subject_"+std::to_string(id))/"best_parameters.json");
            if(fitted.at("subject_id")!=id || fitted.at("condition")!=d.condition)throw std::invalid_argument("fit subject/condition mismatch");
            if(fitted.at("model_version")!=model || fitted.at("resource_fingerprint")!=catalogue->fingerprint)throw std::invalid_argument("fit code/resource identity differs; use frozen implementation or refit");
            if(fitted.contains("provenance") && fitted.at("provenance")!=prov)throw std::invalid_argument("fit provenance differs");
            p=validated_parameters(fitted.at("best_parameters"));
        }
        if(a.command=="fit") {
            fit["model_version"]=model;
            auto scientific_fit=fit;scientific_fit.erase("n_jobs");
            Json manifest={{"schema_version","gm1031-fit-v1"},{"subject_id",id},{"condition",d.condition},
                {"scientific_model_contract","1031-unified-v2"},
                {"data_fingerprint",d.fingerprint},{"model_fingerprint",model},{"resource_fingerprint",catalogue->fingerprint},
                {"rng",optimizer_rng_identity},{"space",space},{"options",scientific_fit},{"smoke",a.smoke}};
            prepare_fit_destination(destination,manifest,a.resume);
            auto result=fit_subject(d,*catalogue,space,fit,destination);
            result.erase("candidate_trace");result["subject_id"]=id;result["condition"]=d.condition;result["provenance"]=prov;result["smoke"]=a.smoke;
            write_json_new(destination/"best_parameters.json",result);
        } else {
            new_destination(destination);
            if(a.command=="evaluate")write_evaluation(d,*catalogue,p,eval,destination);
            else if(a.command=="filter") {
                // Development-only raw filter: the configured root seed is used literally.
                auto result=particle_filter(d,p,*catalogue,eval.at("particle_count").get<int>(),static_cast<std::uint64_t>(seed),eval.at("keep_states").get<bool>());
                probability_arrays(result,d,true);
                auto summary=base_summary(d,*catalogue,p,eval);summary["schema_version"]="gm1031-filter-diagnostic-v1";summary["filter"]=result.summary;summary["seed"]=seed;
                write_npz_new(destination/"filter.npz",result.arrays);write_json_new(destination/"summary.json",summary);
            } else {
                auto seeds=stage_seeds(d.fingerprint,"autonomous",static_cast<int>(repeats),static_cast<std::uint64_t>(seed));
                auto runs=run_tasks(d,*catalogue,p,{},seeds,eval.at("particle_count").get<int>(),jobs,false);
                auto arrays=observed_arrays(d);auto n=d.size(),c=static_cast<std::size_t>(d.n_categories);
                arrays["category_probabilities"]=stack(runs,"category_probabilities",{n,c});
                for(auto name:{"choices","feedback"})arrays[name]=stack(runs,name,{n});
                arrays["seeds"]={{seeds.size()},signed_seeds(seeds)};
                auto summary=base_summary(d,*catalogue,p,{{"simulation_repeats",repeats},{"seed",seed},{"jobs",jobs}});
                summary["seeds"]=seeds;summary["meaning"]="independent autonomous choices and feedback on fixed stimulus schedule";
                write_npz_new(destination/"simulation.npz",arrays);write_json_new(destination/"summary.json",summary);
            }
        }
        std::cout<<Json{{"command",a.command},{"subject",id},{"output",destination.string()},{"smoke",a.smoke}}.dump()<<'\n';
    }
    return 0;
}
}

#ifndef GM_CLI_NO_MAIN
int main(int argc,char** argv) {
    try {return gm::cli_main(std::vector<std::string>(argv+1,argv+argc));}
    catch(const std::exception& e) {std::cerr<<"genmodel_cpp: "<<e.what()<<'\n';return 1;}
}
#endif
