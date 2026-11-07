#include "gm/optimization.hpp"
#include "gm/rng.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace gm {
namespace {
constexpr int dimensions=static_cast<int>(parameter_names.size());
using Point=std::array<int,parameter_names.size()>;
using Points=std::vector<Point>;
using Seen=std::set<Point>;
using Proposal=std::pair<Point,std::string>;
constexpr double infinity=std::numeric_limits<double>::infinity();

std::filesystem::path unique_directory(const std::filesystem::path& root,const std::string& prefix) {
    std::filesystem::create_directories(root);
    static std::atomic<std::uint64_t> counter{0};
    for (int attempt=0;attempt<10000;++attempt) {
        auto name=prefix+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+"-"+std::to_string(counter++);
        auto path=root/name;
        std::error_code ec;
        if(std::filesystem::create_directory(path,ec)) return path;
        if(ec && ec!=std::errc::file_exists) throw std::filesystem::filesystem_error("allocate directory",path,ec);
    }
    throw std::runtime_error("cannot allocate unique output directory");
}
void mask_valid(const std::vector<std::uint8_t>& mask) {
    if(mask.empty() || std::none_of(mask.begin(),mask.end(),[](auto m){return m==1;}) ||
       std::any_of(mask.begin(),mask.end(),[](auto m){return m>1;}))
        throw std::invalid_argument("score_mask must be a nonempty boolean vector with scored trials");
}
std::vector<double> validate_result(const Arrays& a,const Subject& d) {
    const auto& category=a.at("category_probabilities");
    const auto& choice=a.at("choice_probabilities");
    const auto& mask=a.at("score_mask");
    if(category.shape!=Shape{d.size(),static_cast<std::size_t>(d.n_categories)} ||
       choice.shape!=Shape{d.size()} || mask.shape!=Shape{d.size()})
        throw std::invalid_argument("filter probability arrays have invalid shapes");
    const auto* bits=std::get_if<std::vector<std::uint8_t>>(&mask.data);
    if(!bits || *bits!=d.score_mask) throw std::invalid_argument("filter score_mask differs from subject score_mask");
    auto c=category.doubles(),p=choice.doubles();
    if(c.size()!=d.size()*d.n_categories || p.size()!=d.size()) throw std::invalid_argument("filter array storage length mismatch");
    for(std::size_t t=0;t<d.size();++t) if(d.score_mask[t]) {
        if(!std::isfinite(p[t]) || p[t]<0 || p[t]>1) throw std::invalid_argument("invalid choice probability on scored trial");
        double sum=0;
        for(int j=0;j<d.n_categories;++j) {
            double v=c[t*d.n_categories+j];
            if(!std::isfinite(v) || v<0 || v>1) throw std::invalid_argument("invalid category probability on scored trial");
            sum+=v;
        }
        if(std::abs(sum-1)>1.01e-7) throw std::invalid_argument("category probabilities are not normalized");
        if(d.choices[t]<0 || d.choices[t]>=d.n_categories ||
           std::abs(p[t]-c[t*d.n_categories+d.choices[t]])>1e-9)
            throw std::invalid_argument("choice probability differs from category forecast");
    }
    return p;
}
std::vector<double> evaluate_cached(const Subject& d,const Catalogue& c,const Parameters& p,
                                  int particles,std::uint64_t seed,const std::string& version,
                                  const std::string& data_identity,const std::filesystem::path& cache,const Evaluator& evaluator) {
    auto path=cache/(cache_key(data_identity,version,c.fingerprint,p,particles,seed)+".npz");
    if(std::filesystem::exists(path)) return validate_result(read_npz(path),d);
    auto result=evaluator(d,p,c,particles,seed,false);
    auto probabilities=validate_result(result.arrays,d);
    Arrays saved;
    for(auto name:{"category_probabilities","choice_probabilities","score_mask"}) saved.emplace(name,result.arrays.at(name));
    auto temporary=unique_directory(cache,".pending-");
    try {
        auto file=temporary/"probabilities.npz";
        write_npz_new(file,saved);
        // Hard-link publication is atomic and never replaces a racing writer.
        std::error_code ec;
        std::filesystem::create_hard_link(file,path,ec);
        if(ec) {
            if(ec!=std::errc::file_exists) throw std::filesystem::filesystem_error("atomic cache publication",file,path,ec);
            validate_result(read_npz(path),d);
        }
        std::filesystem::remove_all(temporary);
    } catch(...) { std::error_code ec;std::filesystem::remove_all(temporary,ec);throw; }
    return probabilities;
}
void shuffle(Points& p,Rng& rng) {
    for(std::size_t i=p.size();i>1;--i) std::swap(p[i-1],p[rng.integer(i)]);
}
Points global_pool(const Point& shape,int count,Rng& rng,const Seen& excluded) {
    std::uint64_t total=1;
    for(auto n:shape) {
        if(total>std::numeric_limits<std::uint64_t>::max()/static_cast<unsigned>(n)) {total=std::numeric_limits<std::uint64_t>::max();break;}
        total*=n;
    }
    Points pool;
    if(total<=std::max<std::uint64_t>(4096,static_cast<std::uint64_t>(count)*4)) {
        Point point{};
        for(std::uint64_t i=0;i<total;++i) {
            if(!excluded.count(point)) pool.push_back(point);
            for(int j=dimensions-1;j>=0;--j) {if(++point[j]<shape[j]) break;point[j]=0;}
        }
        shuffle(pool,rng);
        if(pool.size()>static_cast<std::size_t>(count)) pool.resize(count);
    } else {
        auto seen=excluded;
        for(std::int64_t i=0;i<static_cast<std::int64_t>(count)*20 && pool.size()<static_cast<std::size_t>(count);++i) {
            Point p{};for(int j=0;j<dimensions;++j)p[j]=static_cast<int>(rng.integer(shape[j]));
            if(seen.insert(p).second)pool.push_back(p);
        }
    }
    return pool;
}
double distance(const Point& a,const Point& b,const Point& shape) {
    double result=0;
    for(int j=0;j<dimensions;++j) {double x=double(a[j]-b[j])/std::max(shape[j]-1,1);result+=x*x;}
    return result;
}
Points diverse(const Points& pool,const Points& selected,int count,const Point& shape) {
    Seen remaining(pool.begin(),pool.end());for(const auto& p:selected)remaining.erase(p);
    Points chosen;auto anchors=selected;
    while(!remaining.empty() && chosen.size()<static_cast<std::size_t>(count)) {
        auto best=remaining.begin();double greatest=-1;
        for(auto it=remaining.begin();it!=remaining.end();++it) {
            double nearest=infinity;for(const auto& a:anchors)nearest=std::min(nearest,distance(*it,a,shape));
            if(nearest>greatest) {greatest=nearest;best=it;}
        }
        chosen.push_back(*best);anchors.push_back(*best);remaining.erase(best);
    }
    return chosen;
}
Points neighbors(const Points& anchors,const Point& shape,int coordinates,const Seen& excluded) {
    Seen candidates;
    for(const auto& a:anchors) for(int i=0;i<dimensions;++i) if(shape[i]>1) for(int di:{-1,1}) {
        Point p=a;p[i]+=di;
        if(p[i]<0 || p[i]>=shape[i])continue;
        if(coordinates==1) {if(!excluded.count(p))candidates.insert(p);continue;}
        for(int j=i+1;j<dimensions;++j)if(shape[j]>1)for(int dj:{-1,1}) {
            Point q=p;q[j]+=dj;
            if(q[j]>=0 && q[j]<shape[j] && !excluded.count(q))candidates.insert(q);
        }
    }
    return Points(candidates.begin(),candidates.end());
}
std::vector<Proposal> proposals(const Points& anchors,const Point& shape,int count,Rng& rng,const Seen& seen) {
    std::array<Points,3> pools{neighbors(anchors,shape,1,seen),neighbors(anchors,shape,2,seen),
                             global_pool(shape,std::max(count*4,128),rng,seen)};
    for(auto& p:pools)shuffle(p,rng);
    const std::array<std::string,3> names{"single_neighbor","double_coordinate","global"};
    Seen used=seen;std::vector<Proposal> out;
    for(int i=0;i<count;++i) {
        bool found=false;
        for(int offset=0;offset<3;++offset) {
            int mode=(i+offset)%3;auto& pool=pools[mode];
            while(!pool.empty() && used.count(pool.back()))pool.pop_back();
            if(!pool.empty()) {out.emplace_back(pool.back(),names[mode]);used.insert(pool.back());pool.pop_back();found=true;break;}
        }
        if(!found)break;
    }
    return out;
}
bool rank_less(const Json& a,const Json& b) {
    double x=a.at("mean_nll").is_null()?infinity:a.at("mean_nll").get<double>();
    double y=b.at("mean_nll").is_null()?infinity:b.at("mean_nll").get<double>();
    return x<y || (x==y && a.at("grid_index").get<Point>()<b.at("grid_index").get<Point>());
}
Points top_points(std::vector<Json> records,int count) {
    std::sort(records.begin(),records.end(),rank_less);Points points;
    for(std::size_t i=0;i<records.size() && i<static_cast<std::size_t>(count);++i)points.push_back(records[i].at("grid_index").get<Point>());
    return points;
}
std::vector<Proposal> label(const Points& points,const std::string& mode) {
    std::vector<Proposal> out;for(const auto& p:points)out.emplace_back(p,mode);return out;
}
}

Json validate_space(const Json& input) {
    if(!input.is_object() || input.size()!=parameter_names.size())throw std::invalid_argument("space must contain exactly the 10 canonical parameters");
    Json out=Json::object();
    for(const auto name:parameter_names) {
        if(!input.contains(name) || !input.at(name).is_array() || input.at(name).empty())throw std::invalid_argument(std::string(name)+": nonempty support required");
        out[name]=Json::array();double previous=-infinity;
        for(const auto& v:input.at(name)) {
            if(!v.is_number() || !std::isfinite(v.get<double>()))throw std::invalid_argument(std::string(name)+": finite numeric support required");
            double x=v.get<double>();
            if(x<=previous)throw std::invalid_argument(std::string(name)+": supports must be strictly increasing");
            previous=x;
            Json p=Parameters{}.json();
            if(std::string_view(name)=="capacity") {
                if(x<1 || x!=std::floor(x) || x>std::numeric_limits<int>::max())throw std::invalid_argument("capacity must contain integers >= 1");
                p[name]=static_cast<int>(x);out[name].push_back(static_cast<int>(x));
            } else {p[name]=x;out[name].push_back(x);}
            Parameters::from_json(p).validate();
        }
    }
    return out;
}
Json optimizer_settings(const Json& config) {
    Json s={{"initial_starts",32},{"rounds",4},{"proposals_per_round",64},{"top_k",4},
        {"search_particles",32},{"search_seeds",4},{"rescore_top",8},{"rescore_diverse",4},
        {"rescore_particles",128},{"rescore_seeds",16},{"audit_top",4},{"audit_particles",256},
        {"audit_seeds",32},{"nearoptimal_tolerance",.005},{"seed",0},{"n_jobs",1},
        {"model_version","1031-unified-v2:cpp-v1"}};
    if(!config.is_object())throw std::invalid_argument("optimizer config must be a mapping");
    for(auto it=config.begin();it!=config.end();++it) {
        if(!s.contains(it.key()))throw std::invalid_argument("unknown optimizer option: "+it.key());
        s[it.key()]=it.value();
    }
    for(auto it=s.begin();it!=s.end();++it) {
        auto key=it.key();const auto& v=it.value();
        if(key=="model_version") {if(!v.is_string() || v.get<std::string>().empty())throw std::invalid_argument("model_version must be nonempty");continue;}
        if(key=="nearoptimal_tolerance") {if(!v.is_number() || !std::isfinite(v.get<double>()) || v.get<double>()<0)throw std::invalid_argument("invalid nearoptimal_tolerance");continue;}
        auto min=(key=="rounds" || key=="rescore_diverse" || key=="seed")?0:1;
        if(!v.is_number_integer() || (v.is_number_unsigned() && v.get<std::uint64_t>()>static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) || v.get<std::int64_t>()<min)
            throw std::invalid_argument(key+" must be an integer >= "+std::to_string(min));
        if(key!="seed" && v.get<std::int64_t>()>std::numeric_limits<int>::max()/32)throw std::invalid_argument(key+" exceeds supported budget range");
    }
    if(s.at("n_jobs").get<int>()>maximum_jobs)throw std::invalid_argument("n_jobs exceeds this platform's worker limit");
    return s;
}
double mean_nll(const std::vector<std::vector<double>>& probabilities,const std::vector<std::uint8_t>& mask) {
    mask_valid(mask);
    if(probabilities.empty())throw std::invalid_argument("at least one filter seed required");
    for(const auto& p:probabilities)if(p.size()!=mask.size())throw std::invalid_argument("choice probabilities must have shape [seeds,trials]");
    double total=0;std::size_t count=0;
    for(std::size_t t=0;t<mask.size();++t)if(mask[t]) {
        double sum=0;for(const auto& p:probabilities) {
            if(!std::isfinite(p[t]) || p[t]<0 || p[t]>1)throw std::invalid_argument("invalid probability on scored trial");
            sum+=p[t];
        }
        total-=std::log(sum/probabilities.size());++count;
    }
    return total/count;
}
std::vector<std::uint64_t> stage_seeds(const std::string& fingerprint,const std::string& stage,int count,std::uint64_t root) {
    if(fingerprint.empty() || stage.empty() || count<1)throw std::invalid_argument("invalid stage seed inputs");
    std::vector<std::uint64_t> seeds;
    for(int i=0;i<count;++i)seeds.push_back(derive_seed(root,Json::array({optimizer_rng_identity,fingerprint,stage,i}).dump()) & 0x7fffffffffffffffULL);
    return seeds;
}
std::string cache_key(const std::string& data,const std::string& model,const std::string& resource,const Parameters& p,int particles,std::uint64_t seed) {
    p.validate();
    return sha256(Json{{"schema","1031-unified-v2:probability-cache-cpp-v1"},{"data",data},{"model",model},{"resource",resource},
        {"parameters",p.json()},{"particle_count",particles},{"ess_fraction",.5},{"seed",seed},{"rng",optimizer_rng_identity}}.dump());
}
Json fit_subject(const Subject& data,const Catalogue& catalogue,const Json& space,const Json& config,const std::filesystem::path& output) {
    auto options=config;
    options["model_version"]=provenance(catalogue).at("scientific_fingerprint");
    return fit_subject(data,catalogue,space,options,output,
        [](const Subject& d,const Parameters& p,const Catalogue& c,int n,std::uint64_t seed,bool keep) {return particle_filter(d,p,c,n,seed,keep);});
}
Json fit_subject(const Subject& data,const Catalogue& catalogue,const Json& input,const Json& config,
                 const std::filesystem::path& output,const Evaluator& evaluator) {
    auto space=validate_space(input),settings=optimizer_settings(config);
    if(!config.contains("model_version") || !evaluator)throw std::invalid_argument("callback evaluator requires an explicit model_version");
    if(data.fingerprint.empty() || catalogue.fingerprint.empty())throw std::invalid_argument("data/resource fingerprints must be nonempty");
    mask_valid(data.score_mask);
    if(data.size()!=data.score_mask.size() || data.size()!=data.choices.size() || data.n_categories<1)throw std::invalid_argument("subject arrays do not align");
#ifdef _OPENMP
    if(omp_in_parallel())throw std::invalid_argument("optimizer cannot run inside a nested OpenMP pool");
#endif
    auto scientific_settings=settings;scientific_settings.erase("n_jobs");
    auto version=settings.at("model_version").get<std::string>();
    auto cache_version=sha256(Json::array({version,scientific_settings,optimizer_rng_identity}).dump());
    const auto data_identity=sha256(Json{{"fingerprint",data.fingerprint},{"id",data.id},{"condition",data.condition},
        {"n_categories",data.n_categories},{"stimuli",data.stimuli},{"choices",data.choices},{"feedback",data.feedback},
        {"true_categories",data.true_categories},{"session_ids",data.session_ids},{"trial_ids",data.trial_ids},
        {"valid",data.valid},{"score_mask",data.score_mask},{"perception_mean",data.perception_mean},
        {"perception_scale",data.perception_scale},{"perception_mode",data.perception_mode},{"protocol",data.protocol}}.dump());
    Point shape{};for(int i=0;i<dimensions;++i)shape[i]=static_cast<int>(space.at(parameter_names[i]).size());
    Rng rng(derive_seed(settings.at("seed").get<std::uint64_t>(),std::string(optimizer_rng_identity)+":grid-search-v1:"+data.fingerprint));
    Json plan=Json::object(),budgets=Json::object(),trace=Json::array();
    for(const std::string stage:{"search","rescore","audit"}) {
        plan[stage]=stage_seeds(data.fingerprint,stage,settings.at(stage+"_seeds").get<int>(),settings.at("seed").get<std::uint64_t>());
        budgets[stage]={{"particles",settings.at(stage+"_particles")},{"seeds",settings.at(stage+"_seeds")},
                        {"evaluated_candidates",0},{"filter_evaluations",0}};
    }
    budgets["search"]["requested_candidates"]=settings.at("initial_starts").get<std::int64_t>()+settings.at("rounds").get<std::int64_t>()*settings.at("proposals_per_round").get<std::int64_t>();
    budgets["rescore"]["requested_candidates"]=settings.at("rescore_top").get<int>()+settings.at("rescore_diverse").get<int>();
    budgets["audit"]["requested_candidates"]=settings.at("audit_top");
    auto root=output/"optimization",cache=root/"cache";
    std::filesystem::create_directories(cache);auto trace_dir=unique_directory(root/"traces","run-");
    write_json_new(trace_dir/"manifest.json",{{"data_fingerprint",data.fingerprint},{"filter_data_fingerprint",data_identity},{"resource_fingerprint",catalogue.fingerprint},
        {"model_version",version},{"rng",optimizer_rng_identity},{"space",space},{"config",scientific_settings},{"seed_plan",plan}});
    auto evaluate=[&](const std::vector<Proposal>& candidates,const std::string& stage,int round) {
        std::vector<Json> records;if(candidates.empty())return records;
        auto seeds=plan.at(stage).get<std::vector<std::uint64_t>>();int particles=settings.at(stage+"_particles").get<int>();
        std::vector<Parameters> parameters;
        for(const auto& [point,mode]:candidates) {
            Json p=Json::object();for(int j=0;j<dimensions;++j)p[parameter_names[j]]=space.at(parameter_names[j]).at(point[j]);
            parameters.push_back(Parameters::from_json(p));
        }
        std::size_t tasks=candidates.size()*seeds.size();
        if(tasks>static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))throw std::invalid_argument("task budget overflow");
        std::vector<std::vector<double>> results(tasks);std::vector<std::exception_ptr> errors(tasks);
        int workers=static_cast<int>(std::min<std::size_t>(settings.at("n_jobs").get<int>(),tasks));
        #pragma omp parallel for schedule(dynamic,1) num_threads(workers)
        for(std::int64_t task=0;task<static_cast<std::int64_t>(tasks);++task) {
            try {results[task]=evaluate_cached(data,catalogue,parameters[task/seeds.size()],particles,seeds[task%seeds.size()],cache_version,data_identity,cache,evaluator);}
            catch(...) {errors[task]=std::current_exception();}
        }
        for(std::size_t task=0;task<tasks;++task)if(errors[task]) {
            try {std::rethrow_exception(errors[task]);}
            catch(const std::exception& e) {throw std::runtime_error(stage+" candidate "+std::to_string(task/seeds.size())+" seed "+std::to_string(seeds[task%seeds.size()])+": "+e.what());}
            catch(...) {throw std::runtime_error(stage+": nonstandard worker exception");}
        }
        std::ofstream stream(trace_dir/(stage+".jsonl"),std::ios::app);
        if(!stream)throw std::runtime_error("cannot open optimizer trace");
        for(std::size_t i=0;i<candidates.size();++i) {
            std::vector<std::vector<double>> repeats(results.begin()+i*seeds.size(),results.begin()+(i+1)*seeds.size());
            double score=mean_nll(repeats,data.score_mask);
            Json record={{"stage",stage},{"round",round},{"proposal_mode",candidates[i].second},{"grid_index",candidates[i].first},
                {"parameters",parameters[i].json()},{"mean_nll",std::isfinite(score)?Json(score):Json(nullptr)},
                {"particles",particles},{"seeds",seeds},{"data_fingerprint",data.fingerprint},{"model_version",version},
                {"resource_fingerprint",catalogue.fingerprint},{"rng",optimizer_rng_identity}};
            stream<<record.dump()<<'\n';records.push_back(record);trace.push_back(record);
        }
        stream.flush();if(!stream)throw std::runtime_error("optimizer trace write failed");
        budgets[stage]["evaluated_candidates"]=budgets[stage]["evaluated_candidates"].get<std::size_t>()+candidates.size();
        budgets[stage]["filter_evaluations"]=budgets[stage]["filter_evaluations"].get<std::size_t>()+tasks;
        return records;
    };
    int initial=settings.at("initial_starts").get<int>();
    auto starts=diverse(global_pool(shape,std::max(256,initial*16),rng,{}),{},initial,shape);
    auto search=evaluate(label(starts,"dispersed_start"),"search",0);Seen seen(starts.begin(),starts.end());
    for(int round=1;round<=settings.at("rounds").get<int>();++round) {
        auto candidates=proposals(top_points(search,settings.at("top_k").get<int>()),shape,settings.at("proposals_per_round").get<int>(),rng,seen);
        if(candidates.empty())break;
        auto records=evaluate(candidates,"search",round);search.insert(search.end(),records.begin(),records.end());
        for(const auto& p:candidates)seen.insert(p.first);
    }
    std::sort(search.begin(),search.end(),rank_less);
    if(search.empty() || search.front().at("mean_nll").is_null())throw std::runtime_error("search found no finite mean NLL");
    auto top=top_points(search,settings.at("rescore_top").get<int>());
    auto extra=diverse(top_points(search,static_cast<int>(search.size())),top,settings.at("rescore_diverse").get<int>(),shape);
    auto nominees=label(top,"top"),other=label(extra,"diverse");nominees.insert(nominees.end(),other.begin(),other.end());
    auto rescored=evaluate(nominees,"rescore",0);
    auto audited=evaluate(label(top_points(rescored,settings.at("audit_top").get<int>()),"top"),"audit",0);
    std::sort(audited.begin(),audited.end(),rank_less);
    if(audited.empty() || audited.front().at("mean_nll").is_null())throw std::runtime_error("audit found no finite mean NLL");
    const auto& best=audited.front();Json near=Json::array();
    for(const auto& record:audited)if(!record.at("mean_nll").is_null() && record.at("mean_nll").get<double>()<=best.at("mean_nll").get<double>()+settings.at("nearoptimal_tolerance").get<double>())near.push_back(record);
    return {{"best_parameters",best.at("parameters")},{"mean_nll",best.at("mean_nll")},{"near_optimal",near},
        {"stage_budgets",budgets},{"seed_plan",plan},{"candidate_trace",trace},{"data_fingerprint",data.fingerprint},
        {"resource_fingerprint",catalogue.fingerprint},{"model_version",version},{"space",space},{"config",scientific_settings}};
}
}
