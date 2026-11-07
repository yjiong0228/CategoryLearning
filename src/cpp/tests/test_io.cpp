#include <doctest/doctest.h>
#include "gm/types.hpp"
#include "gm/rng.hpp"
#include <numeric>
#include <chrono>
#include <fstream>
#include <cnpy.h>
#include <sstream>
#include <limits>
using namespace gm;
TEST_CASE("parameter schema and exact hash") {
    CHECK(sha256("abc")=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    auto p=Parameters::from_json(Json::object());
    CHECK(p.json().size()==10);
    CHECK(p.json()==Json{{"capacity",4},{"similarity_width",.25},{"baseline_probability",.15},
        {"error_gain",1.},{"failure_decay",.8},{"global_search",.2},{"retention",.8},
        {"execution_inertia",1.},{"beta_init",5.},{"learning_rate",.1}});
    CHECK(Parameters::from_json(p.json()).json()==p.json());
    CHECK_THROWS(Parameters::from_json({{"unknown",1}}));
    CHECK_THROWS(Parameters::from_json({{"capacity",2.5}}));
    CHECK_THROWS(Parameters::from_json({{"gamma",.8}}));
}
TEST_CASE("1031 parameter domains include capacity one and closed retention bounds") {
    CHECK_NOTHROW(Parameters::from_json({{"capacity",1},{"failure_decay",1.},{"retention",0.}}));
    CHECK_NOTHROW(Parameters::from_json({{"capacity",10000},{"retention",1.},{"global_search",0.},{"learning_rate",1.}}));
    for(const auto& bad:std::vector<Json>{{{"capacity",0}},{{"capacity",1e100}},
        {{"capacity",std::numeric_limits<std::uint64_t>::max()}},{{"capacity",true}},
        {{"similarity_width",0.}},{{"baseline_probability",0.}},{{"baseline_probability",1.}},
        {{"failure_decay",1.01}},{{"retention",-.01}},{{"global_search",1.01}},
        {{"learning_rate",-.01}},{{"error_gain",-.01}},{{"execution_inertia",-.01}},
        {{"beta_init",.09}},{{"beta_init",25.01}}}) CHECK_THROWS(Parameters::from_json(bad));
    auto defaults=Parameters{}.json();
    for(auto it=defaults.begin();it!=defaults.end();++it) {
        CHECK_THROWS(Parameters::from_json(Json{{it.key(),std::numeric_limits<double>::infinity()}}));
        CHECK_THROWS(Parameters::from_json(Json{{it.key(),std::numeric_limits<double>::quiet_NaN()}}));
    }
}
TEST_CASE("1031 provenance distinguishes the adopted scientific model") {
    Catalogue c;c.fingerprint="fixture";
    auto j=provenance(c);
    CHECK(j.at("model_version")=="1031-unified-v2:cpp-v1");
    CHECK(j.at("source_commit")=="249144de");
    CHECK(j.at("label_policy")=="conditional_pairing_category");
    CHECK(j.at("scientific_fingerprint").get<std::string>().size()==64);
}
TEST_CASE("1031 provenance identifies current exp123 source manifest") {
    Catalogue c;c.fingerprint="fixture";
    const auto j=provenance(c);
    REQUIRE(j.contains("source_manifest"));
    CHECK(j.at("source_manifest")=="src/cpp/resources/data_sources.json");
    CHECK(j.at("source_data_scope")=="exp123");
    const auto root=std::filesystem::path(GM_TEST_ROOT);
    const auto repository=root.parent_path().parent_path();
    const auto manifest=read_json(repository/j.at("source_manifest").get<std::string>());
    CHECK(manifest.at("source_commit")==j.at("source_commit"));
    REQUIRE(manifest.at("files").size()==3);
    for(const auto& entry:manifest.at("files")) {
        const auto path=repository/entry.at("path").get<std::string>();
        const auto bytes=std::filesystem::file_size(path);
        const auto digest=file_sha256(path);
        const bool working=bytes==entry.at("bytes").get<std::uintmax_t>() && digest==entry.at("sha256").get<std::string>();
        const bool stored=entry.contains("git_sha256") && bytes==entry.at("git_bytes").get<std::uintmax_t>() && digest==entry.at("git_sha256").get<std::string>();
        CHECK((working || stored));
    }
}
TEST_CASE("frozen catalogue and real subjects load") {
    const auto root=std::filesystem::path(GM_TEST_ROOT);
    for(auto entry:std::vector<std::pair<int,int>>{{122,1},{206,2},{314,3}}){
        auto d=load_subject(root.parent_path().parent_path()/"data","exp123",entry.first,16);
        CHECK(d.condition==entry.second);
        CHECK(std::accumulate(d.valid.begin(),d.valid.end(),0)==16);
        CHECK(std::accumulate(d.score_mask.begin(),d.score_mask.end(),0)==15);
        CHECK(d.fingerprint.size()==64);
        auto c=load_catalogue(root/"resources",entry.second);
        CHECK(c->size()==(entry.second==1?29:152));
        CHECK(c->functional_distance.size()==c->size()*c->size());
    }
    auto meg=load_subject(root.parent_path().parent_path()/"data","meg",335,16);
    CHECK(std::accumulate(meg.valid.begin(),meg.valid.end(),0)==16);
}
TEST_CASE("random protocol reproduces and separates roles") {
    Rng a(12),b(12); for(int i=0;i<100;++i) CHECK(a.uniform()==b.uniform());
    CHECK(derive_seed(1,"a")!=derive_seed(1,"b"));
    Rng c(42); double sum=0,sq=0; for(int i=0;i<20000;++i){double x=c.normal();sum+=x;sq+=x*x;}
    CHECK(std::abs(sum/20000)<.05); CHECK(std::abs(sq/20000-1)<.05);
}
TEST_CASE("NPZ scalar seed and byte masks roundtrip without shape loss") {
    auto p=std::filesystem::path(GM_TEST_ROOT)/".tmp"/("scalar-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".npz");
    Arrays arrays{{"seed",{{},std::vector<std::int64_t>{123}}},{"mask",{{2},std::vector<std::uint8_t>{0,1}}},{"value",{{},std::vector<double>{.5}}},{"empty",{{0},std::vector<double>{}}}};
    write_npz_new(p,arrays);auto restored=read_npz(p);
    CHECK(restored.at("seed").shape.empty());CHECK(restored.at("seed").doubles()[0]==123);
    CHECK(restored.at("mask").doubles()==std::vector<double>{0,1});
    CHECK(restored.at("value").shape.empty());CHECK(restored.at("value").doubles()[0]==.5);
    CHECK(restored.at("empty").shape==Shape{0});CHECK(restored.at("empty").doubles().empty());
    CHECK_THROWS(write_npz_new(p,arrays));std::filesystem::remove(p);
}
TEST_CASE("infinite perception aggregates become zero after averaging") {
    const auto root=std::filesystem::path(GM_TEST_ROOT)/".tmp"/("perception-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    auto dir=root/"exp123/processed";std::filesystem::create_directories(dir);
    {std::ofstream f(dir/"Task2_processed.csv");f<<"iSub,condition,feature1_name,feature2_name,feature3_name,feature4_name,iSession,iBlock,iTrial,feature1,feature2,feature3,feature4,category,choice,presskey,feedback\n122,1,head,neck,leg,tail,1,1,1,.2,.3,.4,.5,1,1,1,1\n122,1,head,neck,leg,tail,1,1,2,.2,.3,.4,.5,1,1,1,1\n";}
    {std::ofstream f(dir/"Task1b_errorsummary_72.csv");f<<"iSub,feature_name,level,threshold_mean_mean\n122,head,1,inf\n122,head,2,.2\n";}
    auto subject=load_subject(root,"exp123",122);CHECK(subject.perception_scale[0]==0);
    std::filesystem::remove_all(root);
}
TEST_CASE("invalid NPZ write never publishes a partial destination") {
    auto path=std::filesystem::path(GM_TEST_ROOT)/".tmp"/("badwrite-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".npz");
    CHECK_THROWS(write_npz_new(path,{{"a",{{2},std::vector<double>{1,2}}},{"b",{{4},std::vector<double>{1}}}}));
    CHECK_FALSE(std::filesystem::exists(path));
}
TEST_CASE("signed byte sentinels are sign extended and corrupted ZIP is rejected") {
    auto path=std::filesystem::path(GM_TEST_ROOT)/".tmp"/("signed-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".npz");
    // cnpy recognizes char as NPY i1, but not the distinct signed-char type.
    std::vector<char> sentinels{static_cast<char>(-1),0,127};cnpy::npz_save(path.string(),"ids",sentinels.data(),Shape{3},"w");
    CHECK(read_npz(path).at("ids").doubles()==std::vector<double>{-1,0,127});
    {std::fstream f(path,std::ios::binary|std::ios::in|std::ios::out);
        unsigned char local[30]{};f.read(reinterpret_cast<char*>(local),30);
        const auto npy=30+local[26]+256*local[27]+local[28]+256*local[29];
        f.seekg(npy+8);unsigned char length[2]{};f.read(reinterpret_cast<char*>(length),2);
        const auto payload=npy+10+length[0]+256*length[1];f.seekp(payload);f.put('!');}
    CHECK_THROWS_WITH_AS(read_npz(path),"NPZ CRC mismatch",std::runtime_error);std::filesystem::remove(path);
}
TEST_CASE("checked NPZ stream rejects write errors without native crash") {
    std::ostringstream broken;broken.setstate(std::ios::badbit);
    CHECK_THROWS_AS(detail::encode_npz(broken,{{"value",{{},std::vector<double>{1}}}}),std::runtime_error);
}
