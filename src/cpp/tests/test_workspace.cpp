#include <doctest/doctest.h>
#include "gm/model.hpp"
#include "gm/inference.hpp"
#include <cmath>
#include <numeric>
TEST_CASE("1031 templates broaden when a dimension is dropped") {
    gm::Slot s;s.kind=gm::ContentKind::templ;s.anchor={.2,.4,.6,.8};
    auto x=s.anchor;x[0]=.9;
    auto full=gm::template_gain(s,x,5,.25,2).first;s.mask &= ~1u;
    CHECK(gm::template_gain(s,x,5,.25,2).first>full);
    CHECK(gm::template_gain(s,x,5,.25,2).first==doctest::Approx(std::tanh(2.5)));
    CHECK(gm::semantic_kind(s)==2);
}
TEST_CASE("1031 starts uncommitted and allows capacity one") {
    gm::ParticleState s(1,2,5);CHECK(s.count==1);CHECK(s.slots[0].belief==1);
    CHECK(s.slots[0].kind==gm::ContentKind::uncommitted);CHECK(s.slots[0].uid==0);
}
TEST_CASE("1031 duplicate ignores deleted coordinates but retains labels") {
    gm::Slot a;a.kind=gm::ContentKind::templ;a.mask=1;a.labels[0]={1,0,0,0};
    auto b=a;b.anchor[1]=.75;CHECK(gm::duplicate_content(a,b,2,1));
    b.labels[0]={0,1,0,0};CHECK_FALSE(gm::duplicate_content(a,b,2,1));
}

TEST_CASE("1031 deterministic emissions and gradients match frozen Python methods") {
    auto reference=gm::read_json(std::filesystem::path(GM_TEST_ROOT)/"tests/fixtures/reference_1031.json");
    for(const auto& item:reference.at("cases")) {
        INFO(item.at("id").get<std::string>());
        int condition=item.at("condition"),cats=condition==1?2:4;
        auto catalogue=gm::load_catalogue(std::filesystem::path(GM_TEST_ROOT)/"resources",condition);
        gm::Geometry geometry(*catalogue);gm::Parameters params;params.capacity=1;params.retention=1;params.learning_rate=0;
        params.similarity_width=reference.at("similarity_width");
        gm::ParticleState state(1,cats,5);auto& node=state.slots[0];const auto& input=item.at("node");
        if(!input.is_null()) {
            auto kind=input.at("kind").get<std::string>();
            node.kind=kind=="template"?gm::ContentKind::templ:kind=="rule"?gm::ContentKind::rule:gm::ContentKind::uncommitted;
            if(node.kind==gm::ContentKind::rule)node.id=input.at("rule_id");
            if(node.kind==gm::ContentKind::templ) {
                node.anchor=input.at("anchor").get<gm::CategoryVector>();node.mask=0;
                for(int k=0;k<4;++k)if(input.at("mask")[k].get<bool>())node.mask|=1u<<k;
                for(int g=0;g<(condition==3?3:1);++g)for(int c=0;c<cats;++c)
                    node.labels[g][c]=(condition==3?input.at("category_belief")[g][c]:input.at("category_belief")[c]).get<double>();
            }
        }
        if(item.contains("pairing"))node.pairing=item.at("pairing").get<std::array<double,3>>();
        node.beta=item.at("beta");auto stimulus=item.at("stimulus").get<gm::CategoryVector>();
        gm::TrialTape tape(1);gm::decide_perceived(state,stimulus,params,geometry,condition,tape);
        const auto& expected=item.at("expected");
        for(int c=0;c<cats;++c)CHECK(tape.probabilities[0][c]==doctest::Approx(expected.at("category_probabilities")[c].get<double>()).epsilon(1e-9).scale(1.));
        auto kernel=gm::feedback_kernel(tape.conditional_probabilities[0],item.at("choice"),item.at("feedback"),condition);
        auto derivative=gm::feedback_kernel(tape.beta_derivatives[0],item.at("choice"),item.at("feedback"),condition);
        double probability=0,gradient=0;
        for(int g=0;g<(condition==3?3:1);++g) {
            CHECK(kernel[g]==doctest::Approx(expected.at("feedback_kernel")[g].get<double>()).epsilon(1e-9).scale(1.));
            double q=condition==3?node.pairing[g]:1;probability+=q*kernel[g];gradient+=q*derivative[g];
        }
        CHECK(gradient/probability==doctest::Approx(expected.at("log_feedback_gradient").get<double>()).epsilon(1e-9).scale(1.));
        gm::assimilate_feedback(state,tape,item.at("choice"),item.at("feedback"),condition,params);
        if(!expected.at("updated_template_labels").is_null())for(int g=0;g<(condition==3?3:1);++g)for(int c=0;c<cats;++c) {
            auto value=(condition==3?expected.at("updated_template_labels")[g][c]:expected.at("updated_template_labels")[c]).get<double>();
            CHECK(node.labels[g][c]==doctest::Approx(value).epsilon(1e-9).scale(1.));
        }
    }
}

TEST_CASE("1031 admission uses completed feedback and preserves newcomer opportunity") {
    auto cat=gm::load_catalogue(std::filesystem::path(GM_TEST_ROOT)/"resources",1);gm::Geometry geom(*cat);
    for(double feedback:{0.,1.}) {
        gm::Parameters p;p.capacity=2;p.baseline_probability=.999999999;p.error_gain=0;p.global_search=0;
        gm::ParticleState s(2,2,5);s.count=2;s.pending=true;s.previous_feedback=feedback;s.previous_stimulus={.9,.9,.9,.9};
        for(int j=0;j<2;++j) {
            s.slots[j].kind=gm::ContentKind::templ;s.slots[j].uid=j+1;s.slots[j].anchor.fill(j*.2);
            s.slots[j].belief=j?.9:.1;s.slots[j].beta=j?10:2;
        }
        s.next_uid=3;gm::Rng search(72),execution(73);
        auto log=gm::prepare_trial(s,p,*cat,geom,search,execution);
        REQUIRE(log.newcomer_count==1);CHECK(log.newcomer_mass==doctest::Approx(feedback==1?.1:.5));
        CHECK(s.slots[0].beta==doctest::Approx(9.2));CHECK(s.count==2);
        CHECK(s.slots[0].belief+s.slots[1].belief==doctest::Approx(1));
    }
}

TEST_CASE("1031 feedback learning applies outcome retention and ADD10 before label update") {
    auto cat=gm::load_catalogue(std::filesystem::path(GM_TEST_ROOT)/"resources",1);gm::Geometry geometry(*cat);
    for(double feedback:{0.,1.}) {
        gm::Parameters params;params.capacity=2;params.retention=.4;params.learning_rate=.04;
        gm::ParticleState state(2,2,1);state.count=2;
        for(int j=0;j<2;++j) {auto& n=state.slots[j];n.kind=gm::ContentKind::templ;n.beta=1;n.belief=j?.8:.2;n.labels[0]={.25,.75,0,0};}
        gm::TrialTape tape(2);gm::decide_perceived(state,{0,0,0,0},params,geometry,1,tape);
        double probability=gm::feedback_kernel(tape.conditional_probabilities[0],0,feedback,1)[0];
        double derivative=gm::feedback_kernel(tape.beta_derivatives[0],0,feedback,1)[0];
        double ratio=feedback==1?.2/.8:std::pow(.2/.8,.4);
        gm::assimilate_feedback(state,tape,0,feedback,1,params);
        CHECK(state.slots[0].belief/state.slots[1].belief==doctest::Approx(ratio));
        CHECK(state.slots[0].beta==doctest::Approx(1+.4*derivative/probability));
        CHECK(state.pending);CHECK(state.previous_feedback==feedback);
    }
}

TEST_CASE("1031 conditional pairing retains dependence on template labels") {
    auto cat=gm::load_catalogue(std::filesystem::path(GM_TEST_ROOT)/"resources",3);gm::Geometry geometry(*cat);
    gm::Parameters params;params.capacity=1;params.retention=1;params.learning_rate=0;
    gm::ParticleState state(1,4,5);auto& n=state.slots[0];n.kind=gm::ContentKind::templ;
    n.labels[0]={0,1,0,0};n.labels[1]={0,0,1,0};n.labels[2]={0,0,0,1};
    gm::TrialTape tape(1);gm::decide_perceived(state,{0,0,0,0},params,geometry,3,tape);
    gm::assimilate_feedback(state,tape,1,1,3,params);
    CHECK(n.pairing[0]>.98);CHECK(n.pairing[1]<.01);CHECK(n.pairing[2]<.01);
}

TEST_CASE("shared four-category geometry does not overwrite task condition") {
    auto root=std::filesystem::path(GM_TEST_ROOT)/"resources";
    auto c2=gm::load_catalogue(root,2),c3=gm::load_catalogue(root,3);
    CHECK(c2->condition==2);CHECK(c3->condition==3);
    CHECK(c2->fingerprint==c3->fingerprint);
}

TEST_CASE("1031 success preserves tiny recoverable content mass") {
    auto cat=gm::load_catalogue(std::filesystem::path(GM_TEST_ROOT)/"resources",1);gm::Geometry geometry(*cat);
    gm::Parameters params;params.capacity=2;params.retention=.1;params.learning_rate=0;
    gm::ParticleState state(2,2,25);state.count=2;
    for(auto& s:state.slots) {s.kind=gm::ContentKind::templ;s.beta=25;s.labels[0]={0,1,0,0};}
    state.slots[0].belief=1e-320;state.slots[1].belief=1;
    gm::TrialTape tape(2);gm::decide_perceived(state,{0,0,0,0},params,geometry,1,tape);
    gm::assimilate_feedback(state,tape,0,1,1,params);
    CHECK(state.slots[0].belief>0);
    gm::assimilate_feedback(state,tape,0,0,1,params);
    CHECK(state.slots[0].belief>1e-33);
}

TEST_CASE("1031 rule proposals match all rules in the independent Python catalogue") {
    auto reference=gm::read_json(std::filesystem::path(GM_TEST_ROOT)/"tests/fixtures/reference_1031.json");
    for(const auto& item:reference.at("proposals")) {
        INFO(item.at("id").get<std::string>());
        int condition=item.at("condition"),cats=condition==1?2:4;
        auto cat=gm::load_catalogue(std::filesystem::path(GM_TEST_ROOT)/"resources",condition);gm::Geometry geom(*cat);
        gm::Slot parent;parent.kind=gm::ContentKind::templ;parent.beta=item.at("beta");parent.mask=0;
        const auto& node=item.at("parent");parent.anchor=node.at("anchor").get<gm::CategoryVector>();
        for(int k=0;k<4;++k)if(node.at("mask")[k].get<bool>())parent.mask|=1u<<k;
        for(int g=0;g<(condition==3?3:1);++g)for(int c=0;c<cats;++c)
            parent.labels[g][c]=(condition==3?node.at("category_belief")[g][c]:node.at("category_belief")[c]).get<double>();
        if(condition==3)parent.pairing=item.at("pairing").get<std::array<double,3>>();
        auto ids=item.at("rule_ids").get<std::vector<int>>();
        auto weights=gm::rule_proposal_weights(parent,ids,geom,gm::Parameters{},condition);
        const auto& expected=item.at("expected");
        for(std::size_t h=0;h<ids.size();++h) {
            CHECK(weights[h]==doctest::Approx(expected.at("rule_proposal_weights")[h].get<double>()).epsilon(1e-9).scale(1.));
            for(std::size_t x=0;x<item.at("probes").size();++x) {
                auto stimulus=item.at("probes")[x].get<gm::CategoryVector>();
                auto probabilities=gm::category_emission(geom.distances(ids[h],stimulus),parent.beta,cats);
                for(int c=0;c<cats;++c)CHECK(probabilities[c]==doctest::Approx(expected.at("catalogue_category_probabilities")[h][x][c].get<double>()).epsilon(1e-9).scale(1.));
            }
        }
    }
}

TEST_CASE("1031 filter and autonomous paths preserve state and omission invariants") {
    auto root=std::filesystem::path(GM_TEST_ROOT);
    for(int subject:{122,206,314}) {
        auto data=gm::load_subject(root.parent_path().parent_path()/"data","exp123",subject,32);
        auto cat=gm::load_catalogue(root/"resources",data.condition);gm::Parameters params;
        auto a=gm::particle_filter(data,params,*cat,16,81,true),b=gm::particle_filter(data,params,*cat,16,81,false);
        CHECK(a.arrays.at("category_probabilities").doubles()==b.arrays.at("category_probabilities").doubles());
        CHECK(a.summary.at("resample_count").get<int>()>0);
        auto mass=a.arrays.at("posterior").doubles(),counts=a.arrays.at("active_count").doubles();
        for(std::size_t t=0;t<data.size();++t) {
            CHECK(std::accumulate(mass.begin()+4*t,mass.begin()+4*t+4,0.)==doctest::Approx(1));
            CHECK(counts[t]>=1-1e-12);CHECK(counts[t]<=params.capacity+1e-12);
        }
        auto simulation=gm::simulate(data,params,*cat,34,true);
        CHECK(simulation.summary.at("content_path").size()==data.size());
        auto choices=std::get<std::vector<std::int64_t>>(simulation.arrays.at("choices").data);
        auto feedback=simulation.arrays.at("feedback").doubles();
        for(std::size_t t=0;t<data.size();++t)CHECK(feedback[t]==gm::task_feedback(data,t,int(choices[t])));
        // A skipped physical row must consume neither a state transition nor RNG.
        auto skipped=data;
        skipped.stimuli.insert(skipped.stimuli.begin()+1,data.stimuli[0]);
        skipped.valid.insert(skipped.valid.begin()+1,0);skipped.score_mask.insert(skipped.score_mask.begin()+1,0);
        skipped.choices.insert(skipped.choices.begin()+1,-1);skipped.feedback.insert(skipped.feedback.begin()+1,NAN);
        auto omitted=gm::particle_filter(skipped,params,*cat,16,81,false).arrays.at("category_probabilities").doubles();
        auto original=b.arrays.at("category_probabilities").doubles();
        for(std::size_t t=0;t<data.size();++t)for(int c=0;c<data.n_categories;++c)
            CHECK(omitted[(t+(t>0))*data.n_categories+c]==original[t*data.n_categories+c]);
    }
}

TEST_CASE("catalogue diagnostics retain stable rule identity without changing predictions") {
    const std::array<const char*,6> names{"catalogue_prior","catalogue_filtered_prior","catalogue_posterior",
        "catalogue_predictive_execution_probability","catalogue_execution_probability","catalogue_active_probability"};
    auto root=std::filesystem::path(GM_TEST_ROOT);
    for(int subject:{122,206,314}) {
        auto data=gm::load_subject(root.parent_path().parent_path()/"data","exp123",subject,32);
        auto cat=gm::load_catalogue(root/"resources",data.condition);gm::Parameters p;
        p.baseline_probability=.99;p.error_gain=0;p.global_search=1;
        auto kept=gm::particle_filter(data,p,*cat,16,60,true),plain=gm::particle_filter(data,p,*cat,16,60,false);
        CHECK(kept.arrays.at("category_probabilities").doubles()==plain.arrays.at("category_probabilities").doubles());
        REQUIRE(kept.arrays.contains("catalogue_rule_ids"));
        auto ids=std::get<std::vector<std::int64_t>>(kept.arrays.at("catalogue_rule_ids").data);
        REQUIRE(ids.size()==cat->size());
        for(std::size_t h=0;h<ids.size();++h)CHECK(ids[h]==h);
        for(auto name:names) {
            REQUIRE(kept.arrays.contains(name));CHECK_FALSE(plain.arrays.contains(name));
            CHECK(kept.arrays.at(name).shape==gm::Shape{data.size(),cat->size()});
            auto v=kept.arrays.at(name).doubles();
            for(double x:v) {CHECK(std::isfinite(x));CHECK(x>=0);CHECK(x<=1+1e-12);}
        }
        for(auto pair:std::vector<std::pair<std::string,std::string>>{{"catalogue_prior","prior"},
                {"catalogue_filtered_prior","filtered_prior"},{"catalogue_posterior","posterior"},
                {"catalogue_predictive_execution_probability","choice_weights"},{"catalogue_active_probability","active_probability"}}) {
            auto rules=kept.arrays.at(pair.first).doubles(),kinds=kept.arrays.at(pair.second).doubles();
            for(std::size_t t=0;t<data.size();++t)CHECK(std::accumulate(rules.begin()+t*cat->size(),
                rules.begin()+(t+1)*cat->size(),0.)==doctest::Approx(kinds[t*4+3]).epsilon(1e-12).scale(1.));
        }
        auto path=gm::simulate(data,p,*cat,90,true);bool saw_rule=false;
        auto path_prior=path.arrays.at("catalogue_prior").doubles(),path_filtered=path.arrays.at("catalogue_filtered_prior").doubles(),
             path_posterior=path.arrays.at("catalogue_posterior").doubles(),path_execution=path.arrays.at("catalogue_execution_probability").doubles(),
             path_presence=path.arrays.at("catalogue_active_probability").doubles();
        for(const auto& trial:path.summary.at("content_path")) {
            const auto t=trial.at("trial_index").get<std::size_t>();
            std::vector<double> prior(cat->size()),posterior(cat->size()),execution(cat->size()),presence(cat->size());
            for(const auto& node:trial.at("contents_pre"))if(node.at("kind")=="rule") {
                auto h=node.at("rule_id").get<std::size_t>();saw_rule=true;
                prior[h]+=node.at("belief").get<double>();presence[h]=1;
                if(node.at("uid")==trial.at("executed_uid"))execution[h]=1;
            }
            for(const auto& node:trial.at("contents_post"))if(node.at("kind")=="rule")
                posterior[node.at("rule_id").get<std::size_t>()]+=node.at("belief").get<double>();
            for(std::size_t h=0;h<cat->size();++h) {
                auto at=t*cat->size()+h;
                CHECK(path_prior[at]==prior[h]);
                CHECK(path_filtered[at]==prior[h]);
                CHECK(path_posterior[at]==posterior[h]);
                CHECK(path_execution[at]==execution[h]);
                CHECK(path_presence[at]==presence[h]);
            }
        }
        CHECK(saw_rule);
        data.valid[2]=0;data.score_mask[2]=0;
        auto skipped=gm::particle_filter(data,p,*cat,4,60,true);
        for(auto name:names) {
            auto v=skipped.arrays.at(name).doubles();
            for(std::size_t h=0;h<cat->size();++h)CHECK(std::isnan(v[2*cat->size()+h]));
        }
    }
}

TEST_CASE("catalogue execution conditions on current choice rather than current feedback") {
    auto root=std::filesystem::path(GM_TEST_ROOT);
    auto data=gm::load_subject(root.parent_path().parent_path()/"data","exp123",122,32);
    data.perception_scale.fill(0);data.perception_mean.fill(0);
    auto cat=gm::load_catalogue(root/"resources",1);gm::Geometry geometry(*cat);gm::Parameters p;
    p.baseline_probability=.99;p.error_gain=0;p.global_search=1;p.learning_rate=0;
    auto run=gm::particle_filter(data,p,*cat,32,60,true);
    auto predictive=run.arrays.at("catalogue_predictive_execution_probability").doubles();
    auto filtered=run.arrays.at("catalogue_execution_probability").doubles();
    auto forecast=run.arrays.at("choice_probabilities").doubles();bool changed=false;
    for(std::size_t t=0;t<data.size();++t)for(std::size_t h=0;h<cat->size();++h) {
        auto at=t*cat->size()+h;
        if(predictive[at]>0) {
            auto emission=gm::category_emission(geometry.distances(int(h),data.stimuli[t]),p.beta_init,2);
            // With fixed precision and no perception noise the rule emission is
            // identical in every particle: Bayes' rule gives an independent oracle.
            double expected=predictive[at]*emission[data.choices[t]]/forecast[t];
            CHECK(filtered[at]==doctest::Approx(expected).epsilon(1e-10).scale(1.));
            changed=changed||std::abs(filtered[at]-predictive[at])>1e-8;
        }
    }
    CHECK(changed);
}

TEST_CASE("conditional particle export is causal and does not consume randomness") {
    auto root=std::filesystem::path(GM_TEST_ROOT);
    for(int subject:{122,206,314}) {
        auto data=gm::load_subject(root.parent_path().parent_path()/"data","exp123",subject,24);
        auto cat=gm::load_catalogue(root/"resources",data.condition);gm::Parameters p;
        auto kept=gm::particle_filter(data,p,*cat,8,60,true,.5,true);
        auto plain=gm::particle_filter(data,p,*cat,8,60,true);
        CHECK(plain.particle_contents.empty());
        CHECK(kept.arrays.at("category_probabilities").doubles()==plain.arrays.at("category_probabilities").doubles());
        CHECK(kept.arrays.at("posterior").doubles()==plain.arrays.at("posterior").doubles());
        auto changed=data;changed.feedback.back()=data.feedback.back()==1?0:1;
        auto other=gm::particle_filter(changed,p,*cat,8,60,true,.5,true);
        REQUIRE(kept.particle_contents.size()==other.particle_contents.size());
        for(const auto& [name,a]:kept.particle_contents) {
            const auto b=other.particle_contents.at(name).doubles();auto v=a.doubles();
            REQUIRE(v.size()==b.size());
            for(std::size_t i=0;i<v.size();++i)CHECK((v[i]==b[i]||(std::isnan(v[i])&&std::isnan(b[i]))));
        }
        data.valid[2]=0;data.score_mask[2]=0;
        auto skipped=gm::particle_filter(data,p,*cat,8,60,true,.5,true);
        auto weights=skipped.particle_contents.at("particle_weights").doubles();
        for(int i=0;i<8;++i)CHECK(std::isnan(weights[2*8+i]));
        auto active=skipped.particle_contents.at("particle_active").doubles();
        for(int i=0;i<8*p.capacity;++i)CHECK(active[2*8*p.capacity+i]==0);
        CHECK_THROWS_AS(gm::particle_filter(data,p,*cat,8,60,false,.5,true),std::invalid_argument);
    }
}
