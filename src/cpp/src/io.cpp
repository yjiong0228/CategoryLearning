#include "gm/types.hpp"
#include <cnpy.h>
#include <yaml-cpp/yaml.h>
#include <picosha2.h>
#include <zlib.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <limits>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// cnpy's upstream header writer indexes shape[0] unconditionally. Preserve
// genuine NPY scalar shapes instead of changing the public result schema.
namespace gm_npz_detail {
template<class T> std::vector<char> header(const std::vector<std::size_t>& shape) {
    std::string text="{'descr': '";text+=cnpy::BigEndianTest();text+=cnpy::map_type(typeid(T));
    text+=std::to_string(sizeof(T))+"', 'fortran_order': False, 'shape': (";
    for(std::size_t i=0;i<shape.size();++i){if(i)text+=", ";text+=std::to_string(shape[i]);}
    if(shape.size()==1)text+=",";text+="), }";
    text.append(16-(10+text.size())%16,' ');text.back()='\n';
    if(text.size()>65535)throw std::runtime_error("NPY v1 header too large");
    std::vector<char> out{'\x93','N','U','M','P','Y',1,0,static_cast<char>(text.size()&255),static_cast<char>((text.size()>>8)&255)};
    out.insert(out.end(),text.begin(),text.end());return out;
}
}
namespace cnpy {
template<> std::vector<char> create_npy_header<double>(const std::vector<std::size_t>& s){return gm_npz_detail::header<double>(s);}
template<> std::vector<char> create_npy_header<std::int64_t>(const std::vector<std::size_t>& s){return gm_npz_detail::header<std::int64_t>(s);}
template<> std::vector<char> create_npy_header<std::uint8_t>(const std::vector<std::size_t>& s){return gm_npz_detail::header<std::uint8_t>(s);}
}

namespace gm {
std::vector<double> Array::doubles() const {
    return std::visit([](const auto& v){return std::vector<double>(v.begin(),v.end());},data);
}
std::string sha256(const std::string& bytes) { return picosha2::hash256_hex_string(bytes); }
std::string file_sha256(const std::filesystem::path& p) {
    std::ifstream f(p,std::ios::binary); if(!f) throw std::runtime_error("Cannot read "+p.string());
    return sha256(std::string(std::istreambuf_iterator<char>(f),{}));
}
Json read_json(const std::filesystem::path& p) {
    std::ifstream f(p); if(!f) throw std::runtime_error("Cannot read "+p.string());
    Json j; f>>j; return j;
}
namespace {
std::atomic<std::uint64_t> serial{0};
std::filesystem::path temporary(const std::filesystem::path& p) {
    std::filesystem::create_directories(p.parent_path());
    return p.string()+".tmp-"+std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count())+"-"+std::to_string(serial.fetch_add(1));
}
void publish(const std::filesystem::path& tmp,const std::filesystem::path& p) {
    // Detect delayed write errors before publishing a completed checkpoint.
#ifdef _WIN32
    HANDLE h=CreateFileW(tmp.c_str(),GENERIC_WRITE,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(h==INVALID_HANDLE_VALUE)throw std::runtime_error("Cannot open artifact for flush");
    const auto flushed=FlushFileBuffers(h);const auto closed=CloseHandle(h);
    if(!flushed||!closed)throw std::runtime_error("Artifact flush/close failed");
#else
    int fd=::open(tmp.c_str(),O_RDWR);if(fd<0)throw std::runtime_error("Cannot open artifact for flush");
    int flushed=::fsync(fd),closed=::close(fd);if(flushed||closed)throw std::runtime_error("Artifact flush/close failed");
#endif
    // Hard-link publication is atomic and refuses existing destinations on both OSes.
    try { std::filesystem::create_hard_link(tmp,p); }
    catch(...) { std::filesystem::remove(tmp); throw; }
    std::filesystem::remove(tmp);
}
Json yaml_value(const YAML::Node& n) {
    if(n.IsNull()) return nullptr;
    if(n.IsSequence()){Json a=Json::array();for(const auto& v:n)a.push_back(yaml_value(v));return a;}
    if(n.IsMap()){Json j=Json::object();for(const auto& e:n)j[e.first.as<std::string>()]=yaml_value(e.second);return j;}
    std::string s=n.Scalar(); if(n.Tag()=="!") return s;
    if(s=="true")return true; if(s=="false")return false;
    static const std::regex integer("[-+]?[0-9]+");
    static const std::regex real("[-+]?(?:[0-9]+\\.?[0-9]*|\\.[0-9]+)(?:[eE][-+]?[0-9]+)?");
    if(std::regex_match(s,integer))return std::stoll(s);
    if(std::regex_match(s,real))return std::stod(s);
    return s;
}
std::uint16_t u16(const std::vector<unsigned char>& b,std::size_t p) {
    if(p+2>b.size()) throw std::runtime_error("Truncated NPZ");
    return b[p]|(std::uint16_t(b[p+1])<<8);
}
std::uint32_t u32(const std::vector<unsigned char>& b,std::size_t p) {
    return u16(b,p)|(std::uint32_t(u16(b,p+2))<<16);
}
}
void write_json_new(const std::filesystem::path& p,const Json& j) {
    if(std::filesystem::exists(p))throw std::runtime_error("Refusing overwrite: "+p.string());
    auto t=temporary(p);
    try{{std::ofstream f(t,std::ios::binary);f<<j.dump(2)<<'\n';f.flush();if(!f)throw std::runtime_error("JSON flush failed");f.close();if(f.fail())throw std::runtime_error("JSON close failed");}publish(t,p);}
    catch(...){std::filesystem::remove(t);throw;}
}
Json read_yaml(const std::filesystem::path& p) { return yaml_value(YAML::LoadFile(p.string())); }
void detail::encode_npz(std::ostream& stream,const Arrays& arrays) {
    // cnpy supplies numeric NPY headers; checked streaming replaces its unchecked
    // fopen/fwrite ZIP wrapper. ZIP64 is intentionally excluded from this schema.
    using Bytes=std::vector<unsigned char>;Bytes central;std::uint64_t offset=0;
    auto append16=[](Bytes& b,std::uint16_t x){b.push_back(x&255);b.push_back((x>>8)&255);};
    auto append32=[&](Bytes& b,std::uint32_t x){append16(b,x&65535);append16(b,x>>16);};
    auto write=[&](const void* p,std::size_t n){if(n)stream.write(static_cast<const char*>(p),static_cast<std::streamsize>(n));if(!stream)throw std::runtime_error("NPZ stream write failed");offset+=n;};
    if(arrays.empty()||arrays.size()>65535)throw std::runtime_error("Unsupported NPZ entry count");
    for(const auto& [name,a]:arrays){
        const auto filename=name+".npy";if(filename.size()>65535)throw std::runtime_error("NPZ filename too long");
        std::visit([&](const auto& values){
            using T=typename std::decay_t<decltype(values)>::value_type;
            std::size_t count=1;for(auto d:a.shape){if(d&&count>std::numeric_limits<std::size_t>::max()/d)throw std::runtime_error("NPZ shape overflow");count*=d;}
            if(values.size()!=count)throw std::runtime_error("NPZ shape mismatch: "+name);
            auto header=cnpy::create_npy_header<T>(a.shape);auto bytes=values.size()*sizeof(T);
            if(bytes>0xffffffffULL-header.size()||offset>0xffffffffULL)throw std::runtime_error("ZIP64 arrays unsupported");
            auto payload=static_cast<std::uint32_t>(header.size()+bytes),entry_offset=static_cast<std::uint32_t>(offset);
            auto crc=crc32(0,reinterpret_cast<const Bytef*>(header.data()),static_cast<uInt>(header.size()));
            if(bytes)crc=crc32(crc,reinterpret_cast<const Bytef*>(values.data()),static_cast<uInt>(bytes));
            Bytes local;append32(local,0x04034b50);append16(local,20);append16(local,0);append16(local,0);append16(local,0);append16(local,0);
            append32(local,crc);append32(local,payload);append32(local,payload);append16(local,static_cast<std::uint16_t>(filename.size()));append16(local,0);
            local.insert(local.end(),filename.begin(),filename.end());
            write(local.data(),local.size());write(header.data(),header.size());write(values.data(),bytes);
            append32(central,0x02014b50);append16(central,20);append16(central,20);append16(central,0);append16(central,0);append16(central,0);append16(central,0);
            append32(central,crc);append32(central,payload);append32(central,payload);append16(central,static_cast<std::uint16_t>(filename.size()));
            append16(central,0);append16(central,0);append16(central,0);append16(central,0);append32(central,0);append32(central,entry_offset);
            central.insert(central.end(),filename.begin(),filename.end());
        },a.data);
    }
    if(offset>0xffffffffULL||central.size()>0xffffffffULL)throw std::runtime_error("ZIP64 directory unsupported");
    const auto directory_offset=static_cast<std::uint32_t>(offset);write(central.data(),central.size());
    Bytes footer;append32(footer,0x06054b50);append16(footer,0);append16(footer,0);append16(footer,static_cast<std::uint16_t>(arrays.size()));append16(footer,static_cast<std::uint16_t>(arrays.size()));append32(footer,static_cast<std::uint32_t>(central.size()));append32(footer,directory_offset);append16(footer,0);
    write(footer.data(),footer.size());
}
void write_npz_new(const std::filesystem::path& p,const Arrays& arrays) {
    if(std::filesystem::exists(p))throw std::runtime_error("Refusing overwrite: "+p.string());
    if(arrays.empty())throw std::invalid_argument("Cannot write empty NPZ");
    auto t=temporary(p);
    try {
        {std::ofstream f(t,std::ios::binary);if(!f)throw std::runtime_error("Cannot open NPZ temporary file");detail::encode_npz(f,arrays);f.flush();if(!f)throw std::runtime_error("NPZ flush failed");f.close();if(f.fail())throw std::runtime_error("NPZ close failed");}
        const auto restored=read_npz(t);
        if(restored.size()!=arrays.size())throw std::runtime_error("Incomplete NPZ artifact");
        for(const auto& [name,a]:arrays){const auto& b=restored.at(name);
            if(a.shape!=b.shape||a.data.index()!=b.data.index())throw std::runtime_error("NPZ write verification failed");
            std::visit([&](const auto& x){const auto& y=std::get<std::decay_t<decltype(x)>>(b.data);
                using T=typename std::decay_t<decltype(x)>::value_type;
                if(x.size()!=y.size()||(!x.empty()&&std::memcmp(x.data(),y.data(),x.size()*sizeof(T))))throw std::runtime_error("NPZ payload verification failed");},a.data);
        }
        publish(t,p);
    }catch(...){std::filesystem::remove(t);throw;}
}
Arrays read_npz(const std::filesystem::path& p) {
    std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot open NPZ "+p.string());
    std::vector<unsigned char> bytes(std::istreambuf_iterator<char>(f),{});
    if(bytes.size()<22)throw std::runtime_error("Invalid NPZ archive");
    std::size_t end=bytes.size()-22;
    while(u32(bytes,end)!=0x06054b50){if(end==0||bytes.size()-end>65557)throw std::runtime_error("Missing ZIP directory");--end;}
    const auto entries=u16(bytes,end+10);std::size_t cursor=u32(bytes,end+16);
    Arrays result;
    for(unsigned entry=0;entry<entries;++entry){
        if(u32(bytes,cursor)!=0x02014b50)throw std::runtime_error("Invalid ZIP directory");
        auto compressed=u32(bytes,cursor+20),uncompressed=u32(bytes,cursor+24),offset=u32(bytes,cursor+42);
        const auto nl=u16(bytes,cursor+28),el=u16(bytes,cursor+30),cl=u16(bytes,cursor+32);
        if(cursor+46+nl+el+cl>bytes.size()||compressed==0xffffffff||uncompressed==0xffffffff)throw std::runtime_error("Unsupported ZIP64/invalid array");
        std::string name(reinterpret_cast<const char*>(bytes.data()+cursor+46),nl);
        const auto method=u16(bytes,cursor+10);const auto expected_crc=u32(bytes,cursor+16);
        cursor+=46+nl+el+cl;
        if(name.size()<4||name.substr(name.size()-4)!=".npy")continue;
        if(u32(bytes,offset)!=0x04034b50)throw std::runtime_error("Invalid ZIP member");
        auto start=offset+30+u16(bytes,offset+26)+u16(bytes,offset+28);
        if(start+compressed>bytes.size())throw std::runtime_error("Truncated ZIP data");
        std::vector<unsigned char> raw(uncompressed);
        if(method==0){if(compressed!=uncompressed)throw std::runtime_error("Stored ZIP size mismatch");std::copy_n(bytes.begin()+start,uncompressed,raw.begin());}
        else if(method==8){z_stream z{};z.next_in=bytes.data()+start;z.avail_in=compressed;z.next_out=raw.data();z.avail_out=uncompressed;
            if(inflateInit2(&z,-MAX_WBITS)!=Z_OK)throw std::runtime_error("inflate init failed");
            int status=inflate(&z,Z_FINISH);inflateEnd(&z);if(status!=Z_STREAM_END||z.total_out!=uncompressed)throw std::runtime_error("Invalid compressed NPY");}
        else throw std::runtime_error("Unsupported ZIP method");
        if(static_cast<std::uint32_t>(crc32(0,raw.data(),static_cast<uInt>(raw.size())))!=expected_crc)throw std::runtime_error("NPZ CRC mismatch");
        if(raw.size()<10||raw[0]!=0x93||std::string(reinterpret_cast<char*>(raw.data()+1),5)!="NUMPY")throw std::runtime_error("Invalid NPY header");
        const std::size_t pre=raw[6]==1?10:12;
        const auto length=raw[6]==1?u16(raw,8):u32(raw,8);
        if(pre+length>raw.size())throw std::runtime_error("Truncated NPY header");
        std::string header(reinterpret_cast<char*>(raw.data()+pre),length);
        if(header.find("'fortran_order': True")!=std::string::npos)throw std::runtime_error("Fortran NPY arrays unsupported");
        std::smatch match;std::regex dtype("['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]");
        if(!std::regex_search(header,match,dtype))throw std::runtime_error("Missing NPY dtype");
        const std::string kind=match[1];
        std::regex shape_re("['\"]shape['\"]\\s*:\\s*\\(([^)]*)\\)");
        if(!std::regex_search(header,match,shape_re))throw std::runtime_error("Missing NPY shape");
        Shape shape;std::string dims=match[1];std::regex number("[0-9]+");
        for(std::sregex_iterator it(dims.begin(),dims.end(),number),last;it!=last;++it)shape.push_back(std::stoull(it->str()));
        auto n=std::accumulate(shape.begin(),shape.end(),std::size_t{1},std::multiplies<>{});
        const auto* data=raw.data()+pre+length; auto available=raw.size()-pre-length;
        Array a;a.shape=shape;
        if(kind=="<f8"||kind=="=f8"){if(available!=n*8)throw std::runtime_error("NPY payload size");std::vector<double> v(n);std::memcpy(v.data(),data,n*8);a.data=std::move(v);}
        else if(kind=="<i8"||kind=="=i8"){if(available!=n*8)throw std::runtime_error("NPY payload size");std::vector<std::int64_t> v(n);std::memcpy(v.data(),data,n*8);a.data=std::move(v);}
        else if(kind=="|u1"||kind=="<u1"||kind=="=u1"||kind=="|b1"){if(available!=n)throw std::runtime_error("NPY payload size");a.data=std::vector<std::uint8_t>(data,data+n);}
        else if(kind=="|i1"||kind=="<i1"){if(available!=n)throw std::runtime_error("NPY payload size");std::vector<std::int64_t> v(n);for(std::size_t i=0;i<n;++i){std::int8_t x;std::memcpy(&x,data+i,1);v[i]=x;}a.data=std::move(v);}
        else if(kind=="<i4"||kind=="=i4"){if(available!=n*4)throw std::runtime_error("NPY payload size");std::vector<std::int64_t> v(n);for(std::size_t j=0;j<n;++j){std::int32_t x;std::memcpy(&x,data+4*j,4);v[j]=x;}a.data=std::move(v);}
        else throw std::runtime_error("Unsupported nonnumeric NPY dtype: "+kind);
        result.emplace(name.substr(0,name.size()-4),std::move(a));
    }
    return result;
}
Json Parameters::json() const {return {{"capacity",capacity},{"similarity_width",similarity_width},{"baseline_probability",baseline_probability},{"error_gain",error_gain},{"failure_decay",failure_decay},{"global_search",global_search},{"retention",retention},{"execution_inertia",execution_inertia},{"beta_init",beta_init},{"learning_rate",learning_rate}};}
void Parameters::validate() const {
    const auto j=json();for(auto it=j.begin();it!=j.end();++it)if(!std::isfinite(it.value().get<double>()))throw std::invalid_argument("Nonfinite parameter: "+it.key());
    if(!(baseline_probability>0&&baseline_probability<1)||capacity<1||error_gain<0||execution_inertia<0||similarity_width<=0||beta_init<.1||beta_init>25)throw std::invalid_argument("Parameter outside domain");
    for(double v:{failure_decay,global_search,retention,learning_rate})if(v<0||v>1)throw std::invalid_argument("Parameter outside [0,1]");
}
Parameters Parameters::from_json(const Json& values) {
    if(!values.is_object())throw std::invalid_argument("Parameters must be an object");
    Parameters p;const auto defaults=p.json();
    for(auto it=values.begin();it!=values.end();++it)if(!defaults.contains(it.key())||!it.value().is_number()||!std::isfinite(it.value().get<double>()))throw std::invalid_argument("Unknown/nonnumeric/nonfinite parameter: "+it.key());
    if(values.contains("capacity")) {
        const double x=values.at("capacity").get<double>();
        if(x<1 || x>std::numeric_limits<int>::max() || x!=std::floor(x))throw std::invalid_argument("capacity must be an integer >= 1 within int range");
        p.capacity=static_cast<int>(x);
    }
#define GM_PARAM(name) p.name=values.value(#name,p.name)
    GM_PARAM(similarity_width);GM_PARAM(baseline_probability);GM_PARAM(error_gain);GM_PARAM(failure_decay);GM_PARAM(global_search);GM_PARAM(retention);GM_PARAM(execution_inertia);GM_PARAM(beta_init);GM_PARAM(learning_rate);
#undef GM_PARAM
    p.validate();return p;
}
Json provenance(const Catalogue& c){
    return {{"model_version","1031-unified-v2:cpp-v1"},{"source_commit","249144de"},{"source_manifest","src/cpp/resources/data_sources.json"},{"source_data_scope","exp123"},{"implementation_sha256",GM_SOURCE_HASH},{"compiler",GM_COMPILER},{"build_mode",GM_BUILD_MODE},{"floating_point","double-no-fast-math"},{"rng_protocol","mt19937_64-sha256-boxmuller-v1:1031-unified-v2"},{"catalogue_fingerprint",c.fingerprint},{"scientific_fingerprint",sha256(Json::array({"1031-unified-v2:cpp-v1",GM_SOURCE_HASH,c.fingerprint,GM_COMPILER,GM_BUILD_MODE,"mt19937_64-sha256-boxmuller-v1:1031-unified-v2"}).dump())},{"encoding","experience_workspace_boundary"},{"solver","exact_active_set"},{"label_policy","conditional_pairing_category"}};
}
}
