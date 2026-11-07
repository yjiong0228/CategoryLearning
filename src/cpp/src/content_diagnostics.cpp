#include "gm/content_diagnostics.hpp"
#include <algorithm>
#include <limits>

namespace gm {
namespace {
template<class T> void allocate(Arrays& a,const char* name,Shape shape,T fill) {
    std::size_t count=1;for(auto n:shape)count*=n;
    a[name]={std::move(shape),std::vector<T>(count,fill)};
}
template<class T> std::vector<T>& v(Arrays& a,const char* name) {return std::get<std::vector<T>>(a.at(name).data);}
}
std::vector<unsigned> catalogue_feature_masks(const Catalogue& cat) {
    std::vector<unsigned> out(cat.size());
    for(std::size_t h=0;h<cat.size();++h)for(const auto& category:cat.rules[h].categories)
        for(const auto& component:category)for(int d=0;d<4;++d)
            if((component.A.col(d).array().abs()>1e-12).any())out[h]|=1u<<d;
    return out;
}
unsigned content_feature_mask(const Slot& slot,const std::vector<unsigned>& masks) {
    return slot.kind==ContentKind::uncommitted?0u:slot.kind==ContentKind::rule?masks.at(slot.id):slot.mask;
}
void allocate_particle_contents(Arrays& a,std::size_t n,int p,int k,int c,int condition) {
    const Shape slots{n,std::size_t(p),std::size_t(k)};const double nan=std::numeric_limits<double>::quiet_NaN();
    for(auto name:{"particle_representation_kind","particle_semantic_kind","particle_representation_uid",
            "particle_representation_parent","particle_catalogue_rule","particle_feature_mask","particle_born_trial"})
        allocate(a,name,slots,std::int64_t(-1));
    allocate(a,"particle_active",slots,std::uint8_t(0));
    allocate(a,"particle_template_mask",{n,std::size_t(p),std::size_t(k),4},std::uint8_t(0));
    for(auto name:{"particle_belief","particle_beta"})allocate(a,name,slots,nan);
    allocate(a,"particle_choice_weights",slots,0.);
    allocate(a,"particle_selection_probabilities",slots,0.);
    for(auto name:{"particle_weights","particle_predictive_weights"})allocate(a,name,{n,std::size_t(p)},nan);
    allocate(a,"particle_perceived_stimulus",{n,std::size_t(p),4},nan);
    allocate(a,"particle_template_anchor",{n,std::size_t(p),std::size_t(k),4},nan);
    const auto g=std::size_t(condition==3?3:1);
    allocate(a,"particle_template_category_belief",{n,std::size_t(p),std::size_t(k),g,std::size_t(c)},nan);
    allocate(a,"particle_pairing",{n,std::size_t(p),std::size_t(k),g},nan);
    allocate(a,"particle_rule_choice_probabilities",{n,std::size_t(p),std::size_t(k),std::size_t(c)},nan);
}
void record_particle_contents(Arrays& a,std::size_t t,const std::vector<ParticleState>& states,
        const std::vector<TrialTape>& tapes,const std::vector<double>& predictive,
        const std::vector<double>& filtered,const std::vector<unsigned>& masks) {
    const auto p=states.size(),k=states.front().slots.size();
    const auto g=a.at("particle_pairing").shape.at(3),c=a.at("particle_template_category_belief").shape.at(4);
    for(std::size_t i=0;i<p;++i) {
        const auto particle=t*p+i;
        v<double>(a,"particle_weights")[particle]=filtered[i];
        v<double>(a,"particle_predictive_weights")[particle]=predictive[i];
        for(int d=0;d<4;++d)v<double>(a,"particle_perceived_stimulus")[particle*4+d]=tapes[i].perceived_stimulus[d];
        for(int j=0;j<states[i].count;++j) {
            const auto& s=states[i].slots[j];const auto at=particle*k+j;
            v<std::uint8_t>(a,"particle_active")[at]=1;
            for(auto item:std::initializer_list<std::pair<const char*,std::int64_t>>{
                {"particle_representation_kind",int(s.kind)},{"particle_semantic_kind",semantic_kind(s)},
                {"particle_representation_uid",s.uid},{"particle_representation_parent",s.parent_uid},
                {"particle_catalogue_rule",s.kind==ContentKind::rule?s.id:-1},
                {"particle_feature_mask",content_feature_mask(s,masks)},{"particle_born_trial",s.born_trial}})
                v<std::int64_t>(a,item.first)[at]=item.second;
            v<double>(a,"particle_belief")[at]=s.belief;v<double>(a,"particle_beta")[at]=s.beta;
            v<double>(a,"particle_choice_weights")[at]=tapes[i].choice_weights[j];
            v<double>(a,"particle_selection_probabilities")[at]=tapes[i].selection_probabilities[j];
            for(int d=0;d<4;++d)if(s.kind==ContentKind::templ) {
                v<std::uint8_t>(a,"particle_template_mask")[at*4+d]=(s.mask>>d)&1u;
                v<double>(a,"particle_template_anchor")[at*4+d]=s.anchor[d];
            }
            for(std::size_t h=0;h<g;++h) {
                v<double>(a,"particle_pairing")[at*g+h]=g==1?1.:s.pairing[h];
                if(s.kind==ContentKind::templ)for(std::size_t d=0;d<c;++d)
                    v<double>(a,"particle_template_category_belief")[(at*g+h)*c+d]=s.labels[h][d];
            }
            for(std::size_t d=0;d<c;++d)v<double>(a,"particle_rule_choice_probabilities")[at*c+d]=tapes[i].probabilities[j][d];
        }
    }
}
}
