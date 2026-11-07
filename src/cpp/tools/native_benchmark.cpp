#include "gm/inference.hpp"
#include <algorithm>
#include <chrono>
#include <exception>
#include <iostream>
#include <numeric>
#include <omp.h>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
int main(int argc,char** argv){
    try{
        if(argc!=7)throw std::invalid_argument("Usage: gm_benchmark ROOT SUBJECT TRIALS PARTICLES SEEDS JOBS");
        const std::filesystem::path root=std::filesystem::absolute(argv[1]).lexically_normal();const int id=std::stoi(argv[2]),trials=std::stoi(argv[3]),particles=std::stoi(argv[4]),seeds=std::stoi(argv[5]),jobs=std::stoi(argv[6]);
        if(trials<2||trials>256||particles<1||particles>256||seeds<1||seeds>64||jobs<1||jobs>64)throw std::invalid_argument("Benchmark budget outside small-test bounds");
        auto start=Clock::now();auto data=gm::load_subject((root/"../../data").lexically_normal(),id>=333?"meg":"exp123",id,trials);auto catalogue=gm::load_catalogue(root/"resources",data.condition);const auto load_seconds=seconds(start);
        gm::Parameters p;start=Clock::now();auto first=gm::particle_filter(data,p,*catalogue,particles,999,false);const auto cold=seconds(start);
        std::vector<double> elapsed;std::vector<gm::RunResult> results(seeds);std::vector<std::exception_ptr> errors(seeds);
        for(int repeat=0;repeat<3;++repeat){
            start=Clock::now();
            #pragma omp parallel for schedule(static) num_threads(jobs)
            for(int s=0;s<seeds;++s){try{results[s]=gm::particle_filter(data,p,*catalogue,particles,static_cast<std::uint64_t>(s),false);}catch(...){errors[s]=std::current_exception();}}
            for(const auto& error:errors)if(error)std::rethrow_exception(error);
            elapsed.push_back(seconds(start));
        }
        double peak=0;
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS mem{};GetProcessMemoryInfo(GetCurrentProcess(),&mem,sizeof(mem));peak=static_cast<double>(mem.PeakWorkingSetSize)/1048576.;
#else
        rusage mem{};getrusage(RUSAGE_SELF,&mem);peak=static_cast<double>(mem.ru_maxrss)/1024.;
#endif
        const auto values=results[0].arrays.at("choice_probabilities").doubles();double nll=0,n=0;for(std::size_t t=0;t<data.size();++t)if(data.score_mask[t]){nll-=std::log(values[t]);++n;}
        std::cout<<gm::Json{{"subject",id},{"condition",data.condition},{"trials",data.size()},{"particles",particles},{"seeds",seeds},{"jobs",jobs},{"keep_states",false},{"load_seconds",load_seconds},{"cold_filter_seconds",cold},{"warm_batch_seconds",elapsed},{"peak_process_rss_mib",peak},{"first_seed_nll",nll/n},{"parameters",p.json()},{"provenance",gm::provenance(*catalogue)}}.dump(2)<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
