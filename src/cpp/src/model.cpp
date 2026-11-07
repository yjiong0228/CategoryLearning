#include "gm/model.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gm {
namespace {
void normalize(std::span<double> a) {
    double sum=std::accumulate(a.begin(),a.end(),0.);
    if(!(sum>0)||!std::isfinite(sum))throw std::runtime_error("Invalid probability mass");
    for(auto& v:a)v/=sum;
}
bool allowed(int category,int choice,double feedback,int condition,int pairing) {
    if(feedback==1)return category==choice;
    if(condition!=3)return category!=choice;
    int sibling=pairing_siblings[pairing][choice];
    return feedback==.5?category==sibling:category!=choice&&category!=sibling;
}
Slot example(const ParticleState& state,int categories,int condition) {
    Slot node;node.kind=ContentKind::templ;node.anchor=state.previous_stimulus;
    for(int g=0;g<3;++g) {
        for(int c=0;c<categories;++c)node.labels[g][c]=allowed(c,state.previous_choice,state.previous_feedback,condition,g)?1.:0.;
        normalize(std::span(node.labels[g].data(),categories));
    }
    return node;
}
int choose_parent(const ParticleState& state,const std::vector<int>& parents,Rng& rng) {
    std::vector<double> weights;for(int j:parents)weights.push_back(state.slots[j].belief);
    if(std::accumulate(weights.begin(),weights.end(),0.)==0)std::fill(weights.begin(),weights.end(),1.);
    return parents[rng.categorical(weights)];
}
}

ParticleState::ParticleState(int capacity,int categories,double initial_beta):slots(capacity) {
    if(capacity<1||(categories!=2&&categories!=4))throw std::invalid_argument("Invalid workspace shape");
    slots[0].belief=1;slots[0].beta=initial_beta;
    for(auto& s:slots)for(auto& row:s.labels)for(int c=0;c<categories;++c)row[c]=1./categories;
}
TrialTape::TrialTape(int capacity):conditional_probabilities(capacity),beta_derivatives(capacity),
    probabilities(capacity),choice_weights(capacity),selection_probabilities(capacity) {}
int semantic_kind(const Slot& s) {
    return s.kind==ContentKind::uncommitted?0:s.kind==ContentKind::rule?3:std::popcount(s.mask)==4?1:2;
}
CategoryVector category_emission(const CategoryVector& d,double beta,int categories) {
    CategoryVector p{};double nearest=*std::min_element(d.begin(),d.begin()+categories);
    for(int c=0;c<categories;++c)p[c]=std::exp(-beta*(d[c]-nearest));
    normalize(std::span(p.data(),categories));return p;
}
std::pair<double,double> template_gain(const Slot& s,const CategoryVector& x,double beta,double width,int cats) {
    if(!(width>0)||beta<0||!std::isfinite(beta))throw std::invalid_argument("Invalid emission parameters");
    double squared=0;for(int k=0;k<4;++k)if(s.mask&(1u<<k))squared+=(x[k]-s.anchor[k])*(x[k]-s.anchor[k]);
    const double a=std::exp(-squared/(2*width*width)),e=std::exp(-beta),den=1+(cats-1)*e;
    return {a*(-std::expm1(-beta))/den,a*cats*e/(den*den)};
}
ConditionalCategories conditional_emission(const Slot& s,const CategoryVector& x,double beta,double width,int cats,const Geometry& geom) {
    ConditionalCategories p{};
    if(s.kind==ContentKind::rule) {
        auto row=category_emission(geom.distances(s.id,x),beta,cats);p.fill(row);
    } else {
        double gain=s.kind==ContentKind::templ?template_gain(s,x,beta,width,cats).first:0;
        for(int g=0;g<3;++g)for(int c=0;c<cats;++c)p[g][c]=(1-gain)/cats+gain*s.labels[g][c];
    }
    return p;
}
std::array<double,3> feedback_kernel(const ConditionalCategories& probabilities,int choice,double feedback,int condition) {
    int cats=condition==1?2:4;
    if(choice<0||choice>=cats||!(feedback==0||feedback==1||(condition==3&&feedback==.5)))
        throw std::invalid_argument("Invalid feedback event");
    std::array<double,3> kernel{};
    for(int g=0;g<3;++g)for(int c=0;c<cats;++c)if(allowed(c,choice,feedback,condition,g))kernel[g]+=probabilities[g][c];
    return kernel;
}
bool duplicate_content(const Slot& a,const Slot& b,int cats,int condition) {
    if(a.kind!=b.kind)return false;
    if(a.kind==ContentKind::rule)return a.id==b.id;
    if(a.kind==ContentKind::uncommitted)return true;
    if(a.mask!=b.mask)return false;
    for(int k=0;k<4;++k)if((a.mask&(1u<<k))&&a.anchor[k]!=b.anchor[k])return false;
    for(int g=0;g<(condition==3?3:1);++g)for(int c=0;c<cats;++c)if(a.labels[g][c]!=b.labels[g][c])return false;
    return true;
}
std::vector<double> rule_proposal_weights(const Slot& parent,std::span<const int> candidates,const Geometry& geometry,const Parameters&,int condition) {
    const int cats=condition==1?2:4;
    CategoryVector theta{};
    for(int c=0;c<cats;++c)for(int g=0;g<(condition==3?3:1);++g)
        theta[c]+=(condition==3?parent.pairing[g]:1.)*parent.labels[g][c];
    std::vector<int> ignored;for(int k=0;k<4;++k)if(!(parent.mask&(1u<<k)))ignored.push_back(k);
    int probes=1<<ignored.size();std::vector<double> scores(candidates.size());
    for(std::size_t i=0;i<candidates.size();++i)for(int probe=0;probe<probes;++probe) {
        auto x=parent.anchor;for(std::size_t k=0;k<ignored.size();++k)x[ignored[k]]=(probe>>k)&1;
        auto row=category_emission(geometry.distances(candidates[i],x),parent.beta,cats);
        double agreement=0;for(int c=0;c<cats;++c)agreement+=theta[c]*row[c];
        scores[i]+=std::log(std::max(agreement,std::numeric_limits<double>::min()))/probes;
    }
    if(scores.empty())throw std::invalid_argument("No rule candidates");
    double highest=*std::max_element(scores.begin(),scores.end());
    for(auto& v:scores)v=std::exp(v-highest);normalize(scores);return scores;
}
TransitionLog prepare_trial(ParticleState& state,const Parameters& params,const Catalogue& cat,const Geometry& geom,Rng& search,Rng& execution) {
    TransitionLog log;const int cats=cat.n_categories,condition=cat.condition;
    bool replaced_uncommitted=false;
    const int old_count=state.count;const int old_uid=state.executed_uid;
    log.global_probability=params.global_search;
    if(state.pending) {
        state.failure=state.previous_feedback==1?state.failure*params.failure_decay:state.failure+1;
        const double z=std::log(params.baseline_probability)-std::log1p(-params.baseline_probability)+params.error_gain*state.failure;
        double hazard=z>=0?1/(1+std::exp(-z)):std::exp(z)/(1+std::exp(z));
        const bool first=state.slots[0].kind==ContentKind::uncommitted;
        log.search_probability=first?1:hazard;
        if(first||search.uniform()<hazard) {
            Slot candidate;bool proposed=false;int parent_index=-1;
            if(first) {candidate=example(state,cats,condition);log.proposal_source=1;proposed=true;}
            else {
                std::vector<int> generalized,reducible;
                for(int j=0;j<state.count;++j)if(state.slots[j].kind==ContentKind::templ) {
                    if(std::popcount(state.slots[j].mask)<4)generalized.push_back(j);
                    if(std::popcount(state.slots[j].mask)>1)reducible.push_back(j);
                }
                if(!generalized.empty()&&search.uniform()<params.global_search) {
                    parent_index=choose_parent(state,generalized,search);
                    std::vector<int> rules;
                    for(int h=0;h<static_cast<int>(cat.size());++h) {
                        bool occupied=false;for(int j=0;j<state.count;++j)if(state.slots[j].kind==ContentKind::rule&&state.slots[j].id==h)occupied=true;
                        if(!occupied)rules.push_back(h);
                    }
                    if(!rules.empty()) {
                        auto weights=rule_proposal_weights(state.slots[parent_index],rules,geom,params,condition);
                        candidate.kind=ContentKind::rule;candidate.id=rules[search.categorical(weights)];
                        candidate.parent_uid=state.slots[parent_index].uid;log.proposal_source=3;proposed=true;
                    }
                }
                if(!proposed) {
                    parent_index=-1;
                    if(!reducible.empty()&&search.uniform()<.5) {
                        parent_index=choose_parent(state,reducible,search);candidate=state.slots[parent_index];
                        std::vector<int> dims;for(int k=0;k<4;++k)if(candidate.mask&(1u<<k))dims.push_back(k);
                        candidate.mask &= ~(1u<<dims[search.integer(dims.size())]);
                        candidate.parent_uid=state.slots[parent_index].uid;log.proposal_source=2;
                    } else {candidate=example(state,cats,condition);log.proposal_source=1;}
                    proposed=true;
                }
            }
            bool duplicate=false;for(int j=0;j<state.count;++j)if(duplicate_content(candidate,state.slots[j],cats,condition))duplicate=true;
            if(proposed&&!duplicate) {
                std::array<double,3> pairing{};double beta=0;
                for(int j=0;j<state.count;++j) {
                    beta+=state.slots[j].belief*state.slots[j].beta;
                    for(int g=0;g<3;++g)pairing[g]+=state.slots[j].belief*state.slots[j].pairing[g];
                }
                if(parent_index>=0)pairing=state.slots[parent_index].pairing;
                int slot=state.count;const bool full=state.count==params.capacity;
                if(first) {slot=0;replaced_uncommitted=true;}
                else if(full) {
                    double low=1;for(int j=0;j<state.count;++j)low=std::min(low,state.slots[j].belief);
                    std::vector<int> tied;for(int j=0;j<state.count;++j)if(state.slots[j].belief==low)tied.push_back(j);
                    slot=tied[search.integer(tied.size())];
                }
                double mass=first?1:full?(state.previous_feedback==1?state.slots[slot].belief:1./params.capacity):1./(state.count+1);
                if(full&&!first)log.dropped_count=1;
                if(!first&&!full)++state.count;
                state.slots[slot].belief=0;
                double support=0;for(int j=0;j<state.count;++j)support+=state.slots[j].belief;
                if(support>0)for(int j=0;j<state.count;++j)state.slots[j].belief*=((1-mass)/support);
                candidate.uid=state.next_uid++;candidate.born_trial=state.trial;
                candidate.belief=mass;candidate.beta=std::clamp(beta,.1,25.);candidate.pairing=pairing;
                state.slots[slot]=candidate;log.newcomer_count=1;log.newcomer_mass=mass;
            }
        }
        state.pending=false;
    }
    std::vector<double> logits(state.count),weights(state.count);
    for(int j=0;j<state.count;++j)logits[j]=state.slots[j].belief>0?std::log(state.slots[j].belief)+(state.slots[j].uid==old_uid?params.execution_inertia:0):-std::numeric_limits<double>::infinity();
    double maximum=*std::max_element(logits.begin(),logits.end());
    for(int j=0;j<state.count;++j)weights[j]=std::exp(logits[j]-maximum);
    state.executed=static_cast<int>(execution.categorical(weights));state.executed_uid=state.slots[state.executed].uid;
    log.execution_switch=state.executed_uid!=old_uid;log.active_count=state.count;
    log.retained_count=old_count-log.dropped_count-int(replaced_uncommitted);
    ++state.trial;return log;
}
void decide_perceived(ParticleState& state,const CategoryVector& x,const Parameters& p,const Geometry& geom,int condition,TrialTape& tape) {
    int cats=condition==1?2:4;tape.perceived_stimulus=x;tape.choice_probabilities={};
    std::fill(tape.choice_weights.begin(),tape.choice_weights.end(),0.);
    for(int j=0;j<state.count;++j) {
        const auto& s=state.slots[j];auto cp=conditional_emission(s,x,s.beta,p.similarity_width,cats,geom);
        tape.conditional_probabilities[j]=cp;tape.probabilities[j]={};tape.beta_derivatives[j]={};
        if(s.kind==ContentKind::rule) {
            auto d=geom.distances(s.id,x);double mean=0;for(int c=0;c<cats;++c)mean+=cp[0][c]*d[c];
            for(int g=0;g<3;++g)for(int c=0;c<cats;++c)tape.beta_derivatives[j][g][c]=cp[g][c]*(mean-d[c]);
        } else if(s.kind==ContentKind::templ) {
            double dg=template_gain(s,x,s.beta,p.similarity_width,cats).second;
            for(int g=0;g<3;++g)for(int c=0;c<cats;++c)tape.beta_derivatives[j][g][c]=dg*(s.labels[g][c]-1./cats);
        }
        for(int c=0;c<cats;++c)for(int g=0;g<(condition==3?3:1);++g)tape.probabilities[j][c]+=(condition==3?s.pairing[g]:1.)*cp[g][c];
    }
    tape.choice_weights[state.executed]=1;tape.choice_probabilities=tape.probabilities[state.executed];
}
void decide(ParticleState& s,const CategoryVector& stimulus,const Parameters& p,const Geometry& geometry,const Subject& subject,Rng& rng,TrialTape& tape) {
    auto x=stimulus;
    for(int k=0;k<4;++k) {
        double noise=subject.perception_mode=="normal"?subject.perception_mean[k]+subject.perception_scale[k]*rng.normal():(2*rng.uniform()-1)*subject.perception_scale[k];
        x[k]=std::clamp(x[k]+noise,0.,1.);
    }
    decide_perceived(s,x,p,geometry,subject.condition,tape);
}
void assimilate_feedback(ParticleState& state,const TrialTape& tape,int choice,double feedback,int condition,const Parameters& params) {
    const int cats=condition==1?2:4;std::vector<double> posterior(state.count);
    const double content_retention=feedback==1?1:params.retention;
    for(int j=0;j<state.count;++j) {
        auto& s=state.slots[j];const auto original_beta=s.beta;
        auto q=s.pairing;
        if(condition==3) {for(auto& v:q)v=params.retention==0?1:std::pow(v,params.retention);normalize(q);}
        auto kernel=feedback_kernel(tape.conditional_probabilities[j],choice,feedback,condition);
        double evidence=0,derivative=0;
        for(int g=0;g<(condition==3?3:1);++g) {
            double w=condition==3?q[g]:1;evidence+=w*kernel[g];
            for(int c=0;c<cats;++c)if(allowed(c,choice,feedback,condition,g))derivative+=w*tape.beta_derivatives[j][g][c];
        }
        if(!(evidence>0))throw std::runtime_error("Feedback has zero emission support");
        // Normalize log masses before exponentiating, preserving recoverable
        // subnormal beliefs without introducing a probability floor.
        posterior[j]=(content_retention==0?0:content_retention*std::log(s.belief))+std::log(evidence);
        s.beta=std::clamp(original_beta+10*params.learning_rate*derivative/evidence,.1,25.);
        if(condition==3) {for(int g=0;g<3;++g)s.pairing[g]=q[g]*kernel[g]/evidence;}
        if(s.kind==ContentKind::templ) {
            double gain=template_gain(s,tape.perceived_stimulus,original_beta,params.similarity_width,cats).first;
            for(int g=0;g<(condition==3?3:1);++g) {
                double chance=0;for(int c=0;c<cats;++c)chance+=allowed(c,choice,feedback,condition,g)?1./cats:0;
                for(int c=0;c<cats;++c)s.labels[g][c]*=(1-gain)*chance+gain*(allowed(c,choice,feedback,condition,g)?1:0);
                normalize(std::span(s.labels[g].data(),cats));
            }
        }
    }
    const double maximum=*std::max_element(posterior.begin(),posterior.end());
    for(auto& v:posterior)v=std::exp(v-maximum);
    normalize(posterior);for(int j=0;j<state.count;++j) {
        state.slots[j].belief=posterior[j];
        if(condition==3&&posterior[j]==0)state.slots[j].pairing.fill(1./3);
    }
    state.previous_choice=choice;state.previous_feedback=feedback;state.previous_stimulus=tape.perceived_stimulus;state.pending=true;
}
Json export_content(const Slot& s,int cats,int condition) {
    Json j={{"kind",s.kind==ContentKind::uncommitted?"uncommitted":s.kind==ContentKind::templ?"template":"rule"},
        {"uid",s.uid},{"parent_uid",s.parent_uid},{"born_trial",s.born_trial},{"belief",s.belief},{"beta",s.beta}};
    if(s.kind==ContentKind::rule)j["rule_id"]=s.id;
    if(s.kind==ContentKind::templ) {
        j["anchor"]=s.anchor;j["mask"]=Json::array();for(int d=0;d<4;++d)j["mask"].push_back(bool(s.mask&(1u<<d)));
        Json rows=Json::array();for(int g=0;g<(condition==3?3:1);++g)rows.push_back(std::vector<double>(s.labels[g].begin(),s.labels[g].begin()+cats));
        j["category_belief"]=condition==3?rows:rows[0];
    }
    if(condition==3)j["pairing"]=s.pairing;return j;
}
}
