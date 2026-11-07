#include "gm/model.hpp"
#include "gm/inference.hpp"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace gm {
namespace {
struct Constraint { CategoryVector a{}; double b=0; };
struct ActiveSet {
    std::array<int,4> rows{};
    std::array<double,16> inverse{};
    int size=0;
};
struct CompiledPolytope {
    std::vector<Constraint> constraints;
    std::vector<ActiveSet> cheap, exact;
};
double dot(const CategoryVector& a,const CategoryVector& b) {
    double result=0; for(int d=0;d<4;++d) result+=a[d]*b[d]; return result;
}
std::vector<ActiveSet> enumerate(const std::vector<Constraint>& constraints,int rows) {
    std::vector<ActiveSet> result(1);
    for(int n=1;n<=std::min(rows,4);++n) {
        ActiveSet set; set.size=n;
        std::function<void(int,int)> recurse=[&](int depth,int begin) {
            if(depth<n) {
                for(int r=begin;r<=rows-(n-depth);++r) {
                    set.rows[depth]=r; recurse(depth+1,r+1);
                }
                return;
            }
            Eigen::MatrixXd matrix(n,4);
            for(int j=0;j<n;++j) for(int d=0;d<4;++d)
                matrix(j,d)=constraints[set.rows[j]].a[d];
            const Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
            if((svd.singularValues().array()>1e-9).count()!=n) return;
            const Eigen::MatrixXd gram=matrix*matrix.transpose();
            const Eigen::MatrixXd inverse=gram.inverse();
            if(!inverse.allFinite()) return;
            set.inverse.fill(0);
            for(int j=0;j<n;++j) for(int k=0;k<n;++k) set.inverse[j*4+k]=inverse(j,k);
            result.push_back(set);
        };
        recurse(0,0);
    }
    return result;
}
CompiledPolytope compile_polytope(const Polytope& source) {
    if(source.A.cols()!=4 || source.A.rows()!=source.b.size() || !source.A.allFinite() || !source.b.allFinite())
        throw std::invalid_argument("invalid polytope matrix");
    CompiledPolytope p;
    for(Eigen::Index r=0;r<source.A.rows();++r) {
        Constraint c; c.b=source.b[r];
        for(int d=0;d<4;++d) c.a[d]=source.A(r,d);
        p.constraints.push_back(c);
    }
    for(int sign : {-1,1}) for(int d=0;d<4;++d) {
        Constraint c; c.a[d]=sign; c.b=sign<0?0:1; p.constraints.push_back(c);
    }
    p.cheap=enumerate(p.constraints,static_cast<int>(source.A.rows()));
    p.exact=enumerate(p.constraints,static_cast<int>(p.constraints.size()));
    return p;
}
double best_squared(const CategoryVector& x,const CompiledPolytope& p,
                    const std::vector<ActiveSet>& sets,bool clamp) {
    double best=std::numeric_limits<double>::infinity();
    for(const auto& set:sets) {
        std::array<double,4> multipliers{};
        for(int j=0;j<set.size;++j) for(int k=0;k<set.size;++k) {
            const auto& c=p.constraints[set.rows[k]];
            multipliers[j]+=set.inverse[j*4+k]*(dot(c.a,x)-c.b);
        }
        CategoryVector y=x;
        for(int d=0;d<4;++d) {
            for(int j=0;j<set.size;++j) y[d]-=multipliers[j]*p.constraints[set.rows[j]].a[d];
            if(clamp) y[d]=std::clamp(y[d],0.,1.);
        }
        bool feasible=true;
        for(const auto& c:p.constraints) if(dot(c.a,y)>c.b+1e-7) { feasible=false; break; }
        if(!feasible) continue;
        double squared=0; for(int d=0;d<4;++d) squared+=(x[d]-y[d])*(x[d]-y[d]);
        best=std::min(best,squared);
    }
    return best;
}
double distance(const CategoryVector& x,const CompiledPolytope& p) {
    bool inside=true;
    for(const auto& c:p.constraints) if(dot(c.a,x)>c.b+1e-9) { inside=false; break; }
    if(inside) return 0;
    double squared=best_squared(x,p,p.cheap,true);
    // Frozen full-v3: exact enumeration is a fallback, not a refinement of an
    // already feasible cheap candidate after clamping to the unit cube.
    if(!std::isfinite(squared)) squared=best_squared(x,p,p.exact,false);
    if(!std::isfinite(squared)) throw std::runtime_error("exact boundary projection found no feasible candidate");
    return std::sqrt(squared);
}
void append_polytope(std::string& key,const Polytope& p) {
    const auto rows=static_cast<std::int64_t>(p.A.rows());
    const auto cols=static_cast<std::int64_t>(p.A.cols());
    const auto bounds=static_cast<std::int64_t>(p.b.size());
    key.append(reinterpret_cast<const char*>(&rows),sizeof(rows));
    key.append(reinterpret_cast<const char*>(&cols),sizeof(cols));
    key.append(reinterpret_cast<const char*>(&bounds),sizeof(bounds));
    for(Eigen::Index r=0;r<p.A.rows();++r) for(Eigen::Index d=0;d<p.A.cols();++d) {
        double x=p.A(r,d); key.append(reinterpret_cast<const char*>(&x),sizeof(x));
    }
    for(Eigen::Index r=0;r<p.b.size();++r) {
        double x=p.b[r]; key.append(reinterpret_cast<const char*>(&x),sizeof(x));
    }
}
}
struct Geometry::Compiled {
    int categories=0;
    std::vector<CompiledPolytope> polytopes;
    std::vector<std::array<std::vector<int>,4>> components;
};
Geometry::Geometry(const Catalogue& cat) {
    if((cat.n_categories!=2 && cat.n_categories!=4) || cat.rules.empty())
        throw std::invalid_argument("invalid geometry catalogue");
    // Content identity handles temporary catalogues and allocator address reuse.
    std::string identity=std::to_string(cat.n_categories)+":";
    for(const auto& rule:cat.rules) {
        if(rule.categories.size()!=static_cast<std::size_t>(cat.n_categories))
            throw std::invalid_argument("rule category count mismatch");
        identity.push_back('R');
        for(const auto& category:rule.categories) {
            identity.push_back('C'); const auto n=category.size();
            identity.append(reinterpret_cast<const char*>(&n),sizeof(n));
            if(!n) throw std::invalid_argument("empty category region");
            for(const auto& p:category) append_polytope(identity,p);
        }
    }
    static std::mutex mutex;
    static std::unordered_map<std::string,std::shared_ptr<const Compiled>> cache;
    std::lock_guard lock(mutex);
    const auto found=cache.find(identity);
    if(found!=cache.end()) { compiled_=found->second; return; }
    auto data=std::make_shared<Compiled>(); data->categories=cat.n_categories;
    std::unordered_map<std::string,int> poly_lookup;
    for(const auto& rule:cat.rules) {
        std::array<std::vector<int>,4> categories;
        for(int c=0;c<cat.n_categories;++c) for(const auto& poly:rule.categories[c]) {
            std::string key; append_polytope(key,poly);
            auto it=poly_lookup.find(key);
            if(it==poly_lookup.end()) {
                const auto id=static_cast<int>(data->polytopes.size());
                data->polytopes.push_back(compile_polytope(poly));
                it=poly_lookup.emplace(std::move(key),id).first;
            }
            categories[c].push_back(it->second);
        }
        data->components.push_back(std::move(categories));
    }
    compiled_=data; cache.emplace(std::move(identity),std::move(data));
}
int Geometry::n_categories() const { return compiled_->categories; }
CategoryVector Geometry::distances(int hypothesis,const CategoryVector& x) const {
    if(hypothesis<0 || static_cast<std::size_t>(hypothesis)>=compiled_->components.size())
        throw std::invalid_argument("invalid active hypothesis");
    for(double v:x) if(!std::isfinite(v) || v<0 || v>1)
        throw std::invalid_argument("perceived stimulus must be finite in the unit cube");
    CategoryVector result{};
    for(int c=0;c<compiled_->categories;++c) {
        double best=std::numeric_limits<double>::infinity();
        for(int k:compiled_->components[hypothesis][c]) best=std::min(best,distance(x,compiled_->polytopes[k]));
        result[c]=best;
    }
    return result;
}
std::vector<double> category_distances(const Catalogue& cat,int h,const CategoryVector& x) {
    const Geometry geometry(cat); const auto result=geometry.distances(h,x);
    return {result.begin(),result.begin()+cat.n_categories};
}
}
