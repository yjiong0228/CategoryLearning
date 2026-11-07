#include "gm/types.hpp"
#include <csv.hpp>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace gm {
namespace {
using Row=std::map<std::string,std::string>;
std::string stripped(std::string s){
    const auto a=s.find_first_not_of(" \t\r\n"),b=s.find_last_not_of(" \t\r\n");
    if(a==std::string::npos)return "";s=s.substr(a,b-a+1);
    std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});return s;
}
std::vector<Row> read_rows(const std::filesystem::path& path,int subject_id){
    if(!std::filesystem::is_regular_file(path))throw std::runtime_error("Missing data: "+path.string());
    csv::CSVFormat format;format.delimiter(',').quote('"').variable_columns(csv::VariableColumnPolicy::THROW);
    csv::CSVReader reader(path.string(),format);auto columns=reader.get_col_names();
    // Filter while streaming; do not allocate maps/strings for every other subject.
    std::vector<Row> rows;for(auto& input:reader){
        const auto id=input["iSub"].get<std::string>();std::size_t consumed=0;const auto value=std::stod(id,&consumed);
        if(!std::isfinite(value)||value!=std::floor(value)||!stripped(id.substr(consumed)).empty())throw std::runtime_error("Invalid subject identifier");
        if(value!=subject_id)continue;
        Row row;for(const auto& name:columns)row[name]=input[name].get<std::string>();rows.push_back(std::move(row));
    }return rows;
}
double number(const Row& row,const std::string& key,bool missing=false){
    auto it=row.find(key);if(it==row.end())throw std::runtime_error("Missing column: "+key);
    if(stripped(it->second).empty()||stripped(it->second)=="nan"){
        if(missing)return std::numeric_limits<double>::quiet_NaN();throw std::runtime_error("Missing numeric value: "+key);
    }
    std::size_t n=0;const auto value=std::stod(it->second,&n);
    if(n!=it->second.size()&&!stripped(it->second.substr(n)).empty())throw std::runtime_error("Invalid number: "+key);
    return value;
}
std::int64_t integer(const Row& row,const std::string& key){
    const auto x=number(row,key);if(!std::isfinite(x)||x!=std::floor(x)||std::abs(x)>9e15)throw std::runtime_error("Invalid integer: "+key);return static_cast<std::int64_t>(x);
}
bool uniform_measurement(int s){return s>=333||(s>=101&&s<=124)||(s>=201&&s<=224)||(s>=301&&s<=324);}
}
Subject load_subject(const std::filesystem::path& data_root,const std::string& dataset,int id,int max_valid_trials){
    if(dataset!="exp123"&&dataset!="meg")throw std::invalid_argument("Unknown dataset");
    if(max_valid_trials<0)throw std::invalid_argument("max_trials cannot be negative");
    const auto directory=data_root/dataset/"processed";
    const auto source=directory/(dataset=="exp123"?"Task2_processed.csv":"Task3b_processed.csv");
    const auto table=read_rows(source,id);std::vector<Row> rows;
    for(const auto& row:table)if(integer(row,"iSub")==id)rows.push_back(row);
    if(rows.empty())throw std::invalid_argument("Subject not found: "+std::to_string(id));
    Subject d;d.id=id;d.condition=static_cast<int>(integer(rows.front(),"condition"));
    if(d.condition<1||d.condition>3)throw std::runtime_error("Invalid condition");
    d.n_categories=d.condition==1?2:4;
    std::array<std::string,4> order;
    for(int k=0;k<4;++k)order[k]=stripped(rows.front().at("feature"+std::to_string(k+1)+"_name"));
    const std::set<std::string> expected=dataset=="exp123"?std::set<std::string>{"neck","head","leg","tail"}:std::set<std::string>{"green","yellow","pink","blue"};
    if(std::set<std::string>(order.begin(),order.end())!=expected)throw std::runtime_error("Invalid feature order");
    std::set<std::array<std::int64_t,3>> seen;std::array<std::int64_t,3> previous{-1,-1,-1};
    std::map<std::int64_t,std::int64_t> keymap,reverse_keymap;
    for(const auto& row:rows){
        if(integer(row,"condition")!=d.condition)throw std::runtime_error("Inconsistent condition");
        for(int k=0;k<4;++k)if(stripped(row.at("feature"+std::to_string(k+1)+"_name"))!=order[k])throw std::runtime_error("Inconsistent feature order");
        const std::array<std::int64_t,3> trial{integer(row,"iSession"),integer(row,dataset=="exp123"?"iBlock":"iRun"),integer(row,"iTrial")};
        if(trial<previous||!seen.insert(trial).second)throw std::runtime_error("Duplicate/unordered trial");previous=trial;
        bool valid=true;if(dataset=="meg"&&!row.contains("miss"))throw std::runtime_error("MEG requires miss");
        if(row.contains("miss")){auto miss=integer(row,"miss");if(miss!=0&&miss!=1)throw std::runtime_error("Invalid miss");valid=miss==0;}
        const auto truth=integer(row,"category")-1;if(truth<0||truth>=d.n_categories)throw std::runtime_error("Invalid category");
        const auto choice=valid?integer(row,"choice")-1:-1;
        const auto feedback=number(row,"feedback",!valid);
        if(valid){
            if(choice<0||choice>=d.n_categories)throw std::runtime_error("Invalid choice");
            if(feedback!=0&&feedback!=1&&!(d.condition==3&&feedback==.5))throw std::runtime_error("Invalid feedback");
            const auto key=integer(row,"presskey");
            if((keymap.contains(choice)&&keymap.at(choice)!=key)||(reverse_keymap.contains(key)&&reverse_keymap.at(key)!=choice))throw std::runtime_error("Inconsistent response key mapping");
            keymap[choice]=key;reverse_keymap[key]=choice;
        }
        std::array<double,4> x;for(int k=0;k<4;++k){x[k]=number(row,"feature"+std::to_string(k+1));if(!std::isfinite(x[k]))throw std::runtime_error("Nonfinite stimulus");}
        d.stimuli.push_back(x);d.choices.push_back(choice);d.true_categories.push_back(truth);d.feedback.push_back(feedback);
        d.trial_ids.push_back(trial);d.session_ids.push_back(trial[0]);d.valid.push_back(valid);
        Json oral=Json::object();for(const auto& name:{"text","oral_center","oral_A","oral_b","feature1_use","feature2_use","feature3_use","feature4_use"})if(row.contains(name))oral[name]=row.at(name).empty()?Json(nullptr):Json(row.at(name));d.oral_reports.push_back(oral);
    }
    if(std::count(d.valid.begin(),d.valid.end(),1)==0)throw std::runtime_error("No valid trials");
    std::size_t stop=d.size();if(max_valid_trials){int count=0;for(std::size_t t=0;t<d.size();++t)if(d.valid[t]&&++count==max_valid_trials){stop=t+1;break;}}
    d.stimuli.resize(stop);d.choices.resize(stop);d.true_categories.resize(stop);d.feedback.resize(stop);d.trial_ids.resize(stop);d.session_ids.resize(stop);d.valid.resize(stop);
    while(d.oral_reports.size()>stop)d.oral_reports.erase(d.oral_reports.size()-1);
    d.score_mask=d.valid;for(auto& flag:d.score_mask)if(flag){flag=0;break;}
    const bool uniform=uniform_measurement(id);
    const auto measurement=directory/(dataset=="meg"?"Task1b_errorsummary.csv":uniform?"Task1b_errorsummary_72.csv":"Task1b_errorsummary_24.csv");
    const auto measures=read_rows(measurement,id);bool found=false;
    std::array<double,4> sums{},scale_sums{};std::array<int,4> counts{},scale_counts{};
    std::set<std::string> present;
    for(const auto& row:measures)if(integer(row,"iSub")==id){
        found=true;const auto feature=stripped(row.at("feature_name"));auto it=std::find(order.begin(),order.end(),feature);
        if(it==order.end())throw std::runtime_error("Unknown perception feature");present.insert(feature);auto k=static_cast<std::size_t>(it-order.begin());
        double mean=number(row,uniform?"threshold_mean_mean":"error_mean",true);
        if(!std::isnan(mean)){sums[k]+=mean;++counts[k];}
        if(!uniform){double scale=number(row,"error_std",true);if(!std::isnan(scale)){scale_sums[k]+=scale;++scale_counts[k];}}
    }
    if(!found||(!uniform&&present.size()!=4))throw std::runtime_error("Missing perception measurements");
    for(int k=0;k<4;++k){
        const auto mean=counts[k]?sums[k]/counts[k]:0;
        const auto scale=scale_counts[k]?scale_sums[k]/scale_counts[k]:0;
        if(uniform){d.perception_mean[k]=0;d.perception_scale[k]=std::isfinite(mean)?std::abs(mean):0;}
        else{d.perception_mean[k]=std::isfinite(mean)?mean:0;d.perception_scale[k]=std::isfinite(scale)?std::abs(scale):0;}
    }
    d.perception_mode=uniform?"uniform":"normal";
    Json pairs=Json::array();for(auto [choice,key]:keymap)pairs.push_back({choice+1,key});
    d.protocol={{"dataset",dataset},{"feature_order",order},{"choice_coordinate","category_coded_response"},{"choice_to_presskey",pairs}};
    // Hash only selected scientific inputs; raw source hashes remain separately auditable.
    Json identity={{"id",id},{"condition",d.condition},{"stimuli",d.stimuli},{"choices",d.choices},{"feedback",d.feedback},{"true_categories",d.true_categories},{"valid",d.valid},{"trial_ids",d.trial_ids},{"sessions",d.session_ids},{"perception_mean",d.perception_mean},{"perception_scale",d.perception_scale},{"perception_mode",d.perception_mode},{"protocol",d.protocol}};
    d.fingerprint=sha256(identity.dump());return d;
}
double task_feedback(const Subject& d,std::size_t t,int choice){
    if(t>=d.size()||choice<0||choice>=d.n_categories)throw std::invalid_argument("Invalid task response");
    auto truth=d.true_categories[t];if(choice==truth)return 1;
    return d.condition==3&&choice/2==truth/2?.5:0;
}
}
