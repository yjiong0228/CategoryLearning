#include "gm/rng.hpp"
#include "gm/types.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
namespace gm {
std::uint64_t derive_seed(std::uint64_t root,const std::string& purpose){
    return std::stoull(sha256(Json::array({"gm-rng-v1",root,purpose}).dump()).substr(0,16),nullptr,16);
}
double Rng::uniform(){return static_cast<double>(engine_()>>11)*0x1.0p-53;}
double Rng::normal(){
    const double u=1.0-uniform(),v=uniform();
    return std::sqrt(-2.0*std::log(u))*std::cos(6.283185307179586476925286766559*v);
}
std::size_t Rng::integer(std::size_t upper){
    if(!upper)throw std::invalid_argument("Empty integer range");
    const auto bound=static_cast<std::uint64_t>(upper),threshold=-bound%bound;
    std::uint64_t draw;do{draw=engine_();}while(draw<threshold);return static_cast<std::size_t>(draw%bound);
}
std::size_t Rng::categorical(std::span<const double> q){
    double total=0;for(auto p:q){if(!std::isfinite(p)||p<0)throw std::invalid_argument("Invalid categorical weights");total+=p;}
    if(!(total>0)||!std::isfinite(total))throw std::invalid_argument("Zero categorical mass");
    const auto u=uniform()*total;double cumulative=0;std::size_t last=0;
    for(std::size_t i=0;i<q.size();++i){if(q[i]>0)last=i;cumulative+=q[i];if(u<cumulative)return i;}
    return last;
}
}
