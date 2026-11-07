#include "gm/inference.hpp"
#include "gm/model.hpp"
#include "gm/content_diagnostics.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gm {
namespace {
constexpr double missing=std::numeric_limits<double>::quiet_NaN();
void add(Arrays& a,const std::string& key,Shape shape) {
    std::size_t n=1;for(auto s:shape)n*=s;
    a[key]={std::move(shape),std::vector<double>(n,missing)};
}
std::vector<double>& values(Arrays& a,const std::string& key) {return std::get<std::vector<double>>(a.at(key).data);}
void allocate(Arrays& a,std::size_t trials,int condition,std::size_t rules) {
    for(const char* k:{"prior","filtered_prior","posterior","beta","active_probability","choice_weights","dominant_probability"})add(a,k,{trials,4});
    for(const char* k:{"failure_pressure","search_probability","global_probability","active_count","retained_count","newcomer_count","dropped_count","newcomer_mass","fallback"})add(a,k,{trials});
    add(a,"proposal_source",{trials,4});if(condition==3)add(a,"pairing",{trials,4,3});
    for(auto name:catalogue_state_names)add(a,name,{trials,rules});
    std::vector<std::int64_t> ids(rules);std::iota(ids.begin(),ids.end(),0);
    a["catalogue_rule_ids"]={{rules},std::move(ids)};
}
void record(Arrays& a,std::size_t t,const std::vector<ParticleState>& states,const std::vector<TrialTape>& tapes,
            const std::vector<TransitionLog>& logs,const std::vector<double>& predictive,const std::vector<double>& filtered,
            const std::vector<double>& priors,const std::vector<double>& betas) {
    for(const char* k:{"prior","filtered_prior","posterior","beta","active_probability","choice_weights","dominant_probability","proposal_source"})std::fill_n(values(a,k).begin()+t*4,4,0.);
    for(const char* k:{"failure_pressure","search_probability","global_probability","active_count","retained_count","newcomer_count","dropped_count","newcomer_mass","fallback"})values(a,k)[t]=0;
    if(a.count("pairing"))std::fill_n(values(a,"pairing").begin()+t*12,12,0.);
    const auto rules=a.at("catalogue_prior").shape.at(1);
    for(auto name:catalogue_state_names)std::fill_n(values(a,name).begin()+t*rules,rules,0.);
    auto& rule_prior=values(a,"catalogue_prior");auto& rule_filtered=values(a,"catalogue_filtered_prior");
    auto& rule_posterior=values(a,"catalogue_posterior");auto& rule_predictive=values(a,"catalogue_predictive_execution_probability");
    auto& rule_execution=values(a,"catalogue_execution_probability");auto& rule_presence=values(a,"catalogue_active_probability");
    for(std::size_t i=0;i<states.size();++i) {
        const auto& s=states[i];const auto& log=logs[i];auto cap=s.slots.size();double wp=predictive[i],wf=filtered[i];
        for(int j=0;j<s.count;++j) {
            const auto& slot=s.slots[j];auto kind=semantic_kind(slot);auto at=t*4+kind;auto pos=i*cap+j;
            values(a,"prior")[at]+=wp*priors[pos];values(a,"filtered_prior")[at]+=wf*priors[pos];
            values(a,"posterior")[at]+=wf*slot.belief;values(a,"beta")[at]+=wp*betas[pos];
            values(a,"active_probability")[at]+=wp;values(a,"choice_weights")[at]+=wp*tapes[i].choice_weights[j];
            if(a.count("pairing"))for(int g=0;g<3;++g)values(a,"pairing")[at*3+g]+=wf*slot.belief*slot.pairing[g];
            if(slot.kind==ContentKind::rule) {
                if(slot.id<0||static_cast<std::size_t>(slot.id)>=rules)throw std::runtime_error("Invalid diagnostic catalogue rule ID");
                auto h=t*rules+slot.id;
                rule_prior[h]+=wp*priors[pos];
                rule_filtered[h]+=wf*priors[pos];
                rule_posterior[h]+=wf*slot.belief;
                rule_predictive[h]+=wp*tapes[i].choice_weights[j];
                rule_execution[h]+=wf*tapes[i].choice_weights[j];
                rule_presence[h]+=wp;
            }
        }
        values(a,"dominant_probability")[t*4+semantic_kind(s.slots[s.executed])]+=wp;
        values(a,"failure_pressure")[t]+=wp*s.failure;
        values(a,"search_probability")[t]+=wp*log.search_probability;
        values(a,"global_probability")[t]+=wp*log.global_probability;
        values(a,"active_count")[t]+=wp*s.count;
        values(a,"retained_count")[t]+=wp*log.retained_count;
        values(a,"newcomer_count")[t]+=wp*log.newcomer_count;
        values(a,"dropped_count")[t]+=wp*log.dropped_count;
        values(a,"newcomer_mass")[t]+=wp*log.newcomer_mass;
        values(a,"proposal_source")[t*4+log.proposal_source]+=wp;
    }
    for(int k=0;k<4;++k) {auto at=t*4+k;double n=values(a,"active_probability")[at];if(n>0)values(a,"beta")[at]/=n;}
}
struct Streams {
    Rng perception,search,execution,resample,choice;
    explicit Streams(std::uint64_t seed):perception(derive_seed(seed,"1031-perception")),search(derive_seed(seed,"1031-search")),
        execution(derive_seed(seed,"1031-execution")),resample(derive_seed(seed,"1031-resample")),choice(derive_seed(seed,"1031-choice")) {}
};
RunResult run(const Subject& d,const Parameters& p,const Catalogue& cat,int particles,std::uint64_t seed,bool keep,double threshold,bool simulation,bool export_particles=false) {
    p.validate();
    if(particles<1||threshold<=0||threshold>1||!std::isfinite(threshold)||d.n_categories!=cat.n_categories||d.condition!=cat.condition)
        throw std::invalid_argument("Invalid filter/subject/catalogue configuration");
    if(d.size()==0||d.valid.size()!=d.size()||d.score_mask.size()!=d.size()||d.choices.size()!=d.size()||d.feedback.size()!=d.size())
        throw std::invalid_argument("Invalid trial shapes");
    Geometry geom(cat);Streams rng(seed);
    std::vector<ParticleState> states,resampled;std::vector<TrialTape> tapes;
    for(int i=0;i<particles;++i) {states.emplace_back(p.capacity,d.n_categories,p.beta_init);resampled.emplace_back(p.capacity,d.n_categories,p.beta_init);tapes.emplace_back(p.capacity);}
    std::vector<double> weights(particles,1./particles),predictive(particles),priors(keep?particles*p.capacity:0),betas(priors.size());
    std::vector<TransitionLog> logs(particles);
    RunResult out;auto& a=out.arrays;auto n=d.size();
    add(a,"category_probabilities",{n,static_cast<std::size_t>(d.n_categories)});
    if(simulation) {a["choices"]={{n},std::vector<std::int64_t>(n,-1)};add(a,"feedback",{n});a["valid"]={{n},d.valid};}
    else {add(a,"choice_probabilities",{n});add(a,"effective_sample_size",{n});a["score_mask"]={{n},d.score_mask};}
    if(keep)allocate(a,n,d.condition,cat.size());
    std::vector<unsigned> rule_masks;
    if(keep) {rule_masks=catalogue_feature_masks(cat);add(a,"feature_set_probability",{n,16});add(a,"feature_probability",{n,4});}
    if(export_particles)allocate_particle_contents(out.particle_contents,n,particles,p.capacity,d.n_categories,d.condition);
    int resamples=0;Json content_path=Json::array();
    for(std::size_t t=0;t<n;++t) {
        if(!d.valid[t])continue;
        predictive=weights;Json snapshot;
        for(int i=0;i<particles;++i) {
            const auto previous_uid=states[i].executed_uid;
            logs[i]=prepare_trial(states[i],p,cat,geom,rng.search,rng.execution);
            decide(states[i],d.stimuli[t],p,geom,d,rng.perception,tapes[i]);
            if(export_particles) {
                auto& selection=tapes[i].selection_probabilities;
                std::fill(selection.begin(),selection.end(),0.);
                double maximum=-std::numeric_limits<double>::infinity(),sum=0;
                for(int j=0;j<states[i].count;++j) {
                    const auto& slot=states[i].slots[j];
                    selection[j]=slot.belief>0?std::log(slot.belief)+(slot.uid==previous_uid?p.execution_inertia:0):-std::numeric_limits<double>::infinity();
                    maximum=std::max(maximum,selection[j]);
                }
                for(int j=0;j<states[i].count;++j)sum+=(selection[j]=std::exp(selection[j]-maximum));
                for(int j=0;j<states[i].count;++j)selection[j]/=sum;
            }
            if(keep)for(int j=0;j<states[i].count;++j) {priors[i*p.capacity+j]=states[i].slots[j].belief;betas[i*p.capacity+j]=states[i].slots[j].beta;}
        }
        if(keep&&simulation) {
            snapshot={{"trial_index",t},{"executed_uid",states[0].executed_uid},{"perceived_stimulus",tapes[0].perceived_stimulus},{"contents_pre",Json::array()}};
            for(int j=0;j<states[0].count;++j)snapshot["contents_pre"].push_back(export_content(states[0].slots[j],d.n_categories,d.condition));
        }
        auto base=t*d.n_categories;auto& forecast=values(a,"category_probabilities");std::fill_n(forecast.begin()+base,d.n_categories,0.);
        for(int i=0;i<particles;++i)for(int c=0;c<d.n_categories;++c)forecast[base+c]+=weights[i]*tapes[i].choice_probabilities[c];
        int choice;double feedback;
        if(simulation) {
            choice=static_cast<int>(rng.choice.categorical(std::span(forecast.data()+base,d.n_categories)));feedback=task_feedback(d,t,choice);
            std::get<std::vector<std::int64_t>>(a["choices"].data)[t]=choice;values(a,"feedback")[t]=feedback;
        } else {
            choice=static_cast<int>(d.choices[t]);feedback=d.feedback[t];
            if(choice<0||choice>=d.n_categories)throw std::invalid_argument("Invalid observed choice");
            values(a,"choice_probabilities")[t]=forecast[base+choice];double sum=0;
            for(int i=0;i<particles;++i) {weights[i]*=tapes[i].choice_probabilities[choice];sum+=weights[i];}
            if(!(sum>0)||!std::isfinite(sum))throw std::runtime_error("Observed choice has zero particle support");
            for(auto& w:weights)w/=sum;
        }
        // Both diagnostics condition on the observed choice, never its feedback.
        // Snapshot before learning and resampling; no extra emission/RNG work.
        if(export_particles)record_particle_contents(out.particle_contents,t,states,tapes,predictive,weights,rule_masks);
        if(keep) {
            auto& sets=values(a,"feature_set_probability");auto& features=values(a,"feature_probability");
            std::fill_n(sets.begin()+t*16,16,0.);std::fill_n(features.begin()+t*4,4,0.);
            for(int i=0;i<particles;++i)for(int j=0;j<states[i].count;++j) {
                const auto code=content_feature_mask(states[i].slots[j],rule_masks);
                const double mass=weights[i]*tapes[i].choice_weights[j];sets[t*16+code]+=mass;
                for(int dimension=0;dimension<4;++dimension)if(code&(1u<<dimension))features[t*4+dimension]+=mass;
            }
        }
        for(int i=0;i<particles;++i)assimilate_feedback(states[i],tapes[i],choice,feedback,d.condition,p);
        if(keep)record(a,t,states,tapes,logs,predictive,weights,priors,betas);
        if(keep&&simulation) {
            snapshot["choice"]=choice;snapshot["feedback"]=feedback;snapshot["contents_post"]=Json::array();
            for(int j=0;j<states[0].count;++j)snapshot["contents_post"].push_back(export_content(states[0].slots[j],d.n_categories,d.condition));
            content_path.push_back(std::move(snapshot));
        }
        if(!simulation) {
            double squares=0;for(auto w:weights)squares+=w*w;double ess=1/squares;values(a,"effective_sample_size")[t]=ess;
            if(ess<threshold*particles) {
                double offset=rng.resample.uniform(),cumulative=weights[0];int ancestor=0;
                for(int i=0;i<particles;++i) {
                    double position=(offset+i)/particles;
                    while(ancestor+1<particles&&cumulative<=position)cumulative+=weights[++ancestor];
                    resampled[i]=states[ancestor];
                }
                states.swap(resampled);std::fill(weights.begin(),weights.end(),1./particles);++resamples;
            }
        }
    }
    out.summary={{"model_version","1031-unified-v2"},{"resample_count",resamples},{"particle_count",particles},{"seed",seed},
        {"condition",d.condition},{"parameters",p.json()},{"mode",simulation?"simulation":"filter"},
        {"semantic_axis",{"uncommitted","example","generalization","catalogue_rule"}},
        {"valid_trials",std::count(d.valid.begin(),d.valid.end(),1)}};
    if(keep&&simulation)out.summary["content_path"]=std::move(content_path);
    return out;
}
}
RunResult particle_filter(const Subject& d,const Parameters& p,const Catalogue& cat,int count,std::uint64_t seed,bool keep,double threshold,bool export_particles) {
    if(export_particles&&!keep)throw std::invalid_argument("Particle export requires keep_states");
    return run(d,p,cat,count,seed,keep,threshold,false,export_particles);
}
RunResult simulate(const Subject& d,const Parameters& p,const Catalogue& cat,std::uint64_t seed,bool keep) {return run(d,p,cat,1,seed,keep,.5,true);}
}
