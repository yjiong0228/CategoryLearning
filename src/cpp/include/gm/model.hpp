#pragma once
#include "types.hpp"
#include "rng.hpp"
#include <span>

namespace gm {
using CategoryVector = std::array<double,4>;
using ConditionalCategories=std::array<CategoryVector,3>;
inline constexpr std::array<std::array<int,4>,3> pairing_siblings{{
    {{1,0,3,2}}, {{2,3,0,1}}, {{3,2,1,0}}
}};

// Compiled constraints are immutable and shared by catalogue content, never by a
// dangling Catalogue address. A Geometry owns no reference to its input object.
class Geometry {
    struct Compiled;
    std::shared_ptr<const Compiled> compiled_;
public:
    explicit Geometry(const Catalogue&);
    int n_categories() const;
    CategoryVector distances(int hypothesis, const std::array<double,4>&) const;
};

enum class ContentKind { uncommitted, templ, rule };
struct Slot {
    ContentKind kind=ContentKind::uncommitted;
    int id=-1,uid=0,parent_uid=-1,born_trial=0;
    unsigned mask=15;
    CategoryVector anchor{};
    ConditionalCategories labels{};
    std::array<double,3> pairing{1./3,1./3,1./3};
    double belief=0,beta=5;
};
struct ParticleState {
    std::vector<Slot> slots;
    int count=1,executed=0,executed_uid=0,next_uid=1,trial=0;
    double failure=0,previous_feedback=1;
    bool pending=false;
    CategoryVector previous_stimulus{};
    int previous_choice=0;
    ParticleState(int capacity,int categories,double beta_init);
};
struct TrialTape {
    CategoryVector perceived_stimulus{},choice_probabilities{};
    std::vector<ConditionalCategories> conditional_probabilities,beta_derivatives;
    std::vector<CategoryVector> probabilities;
    std::vector<double> choice_weights,selection_probabilities;
    explicit TrialTape(int capacity);
};
struct TransitionLog {
    double retained_count=0,newcomer_count=0,active_count=0,dropped_count=0;
    double newcomer_mass=0,search_probability=0,global_probability=0;
    int proposal_source=0;
    bool execution_switch=false;
};
int semantic_kind(const Slot&);
CategoryVector category_emission(const CategoryVector&,double beta,int categories);
std::pair<double,double> template_gain(const Slot&,const CategoryVector&,double beta,double width,int categories);
ConditionalCategories conditional_emission(const Slot&,const CategoryVector&,double beta,double width,int categories,const Geometry&);
std::array<double,3> feedback_kernel(const ConditionalCategories&,int choice,double feedback,int condition);
std::vector<double> rule_proposal_weights(const Slot&,std::span<const int>,const Geometry&,const Parameters&,int condition);
bool duplicate_content(const Slot&,const Slot&,int categories,int condition);
TransitionLog prepare_trial(ParticleState&,const Parameters&,const Catalogue&,const Geometry&,Rng& search,Rng& execution);
void decide_perceived(ParticleState&,const CategoryVector&,const Parameters&,const Geometry&,int condition,TrialTape&);
void decide(ParticleState&,const CategoryVector&,const Parameters&,const Geometry&,const Subject&,Rng&,TrialTape&);
void assimilate_feedback(ParticleState&,const TrialTape&,int choice,double feedback,int condition,const Parameters&);
Json export_content(const Slot&,int categories,int condition);
}
