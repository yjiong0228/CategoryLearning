#include "gm/types.hpp"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
namespace gm {
std::shared_ptr<const Catalogue> load_catalogue(const std::filesystem::path& root,int condition){
    if(condition<1||condition>3)throw std::invalid_argument("Condition must be 1,2,3");
    static std::mutex mutex;static std::map<std::string,std::shared_ptr<const Catalogue>> cache;
    const int code=condition==1?1:2;
    const auto key=std::filesystem::weakly_canonical(root).string()+"/"+std::to_string(condition);
    std::lock_guard lock(mutex);if(auto it=cache.find(key);it!=cache.end())return it->second;
    const auto base=root/("catalogue_c"+std::to_string(code));
    auto j=read_json(base.string()+".json");auto arrays=read_npz(base.string()+".npz");
    auto c=std::make_shared<Catalogue>();c->condition=condition;c->n_categories=j.at("n_categories");
    for(const auto& r:j.at("rules")){
        Rule rule;rule.metadata=r;
        for(const auto& category:r.at("regions")){
            std::vector<Polytope> parts;
            for(const auto& region:category){
                const auto rows=region.at("b").size();Polytope poly;
                poly.A.resize(rows,4);poly.b.resize(rows);
                if(region.at("A").size()!=rows)throw std::runtime_error("Region row mismatch");
                for(std::size_t i=0;i<rows;++i){if(region["A"][i].size()!=4)throw std::runtime_error("Region dimension mismatch");poly.b[i]=region["b"][i];for(int k=0;k<4;++k)poly.A(i,k)=region["A"][i][k];}
                parts.push_back(std::move(poly));
            }
            if(parts.empty())throw std::runtime_error("Empty category region");
            rule.categories.push_back(std::move(parts));
        }
        if(rule.categories.size()!=static_cast<std::size_t>(c->n_categories))throw std::runtime_error("Category count mismatch");
        c->rules.push_back(std::move(rule));
    }
    const auto shape=Shape{c->size(),c->size()};
    if(arrays.at("functional_distance").shape!=shape||arrays.at("edit_costs").shape!=shape)throw std::runtime_error("Catalogue matrix shape mismatch");
    c->functional_distance=arrays.at("functional_distance").doubles();c->edit_costs=arrays.at("edit_costs").doubles();
    for(auto& value:c->functional_distance){if(!std::isfinite(value)||value< -1e-12||value>1+1e-12)throw std::runtime_error("Invalid functional distance");value=std::clamp(value,0.0,1.0);}
    for(auto value:c->edit_costs)if(std::isnan(value)||value<0)throw std::runtime_error("Invalid edit cost");
    c->fingerprint=sha256(file_sha256(base.string()+".json")+file_sha256(base.string()+".npz"));
    cache.emplace(key,c);return c;
}
}
