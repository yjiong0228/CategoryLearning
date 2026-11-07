#include <doctest/doctest.h>
#include "gm/inference.hpp"
#include <cmath>
#include <future>

namespace {
const auto root = std::filesystem::path(GM_TEST_ROOT);
}

TEST_CASE("geometry matches every independent frozen rule and stimulus") {
    for (int condition : {1, 2}) {
        const auto cat = gm::load_catalogue(root / "resources", condition);
        const auto fixture = gm::read_npz(root / "tests/fixtures" /
            ("geometry_c" + std::to_string(condition) + ".npz"));
        const auto x = fixture.at("stimuli").doubles();
        const auto expected = fixture.at("distances").doubles();
        const auto trials = fixture.at("stimuli").shape.at(0);
        REQUIRE(cat->size() == (condition == 1 ? 29 : 152));
        for (std::size_t h = 0; h < cat->size(); ++h) {
            for (std::size_t t = 0; t < trials; ++t) {
                const auto d = gm::category_distances(*cat, static_cast<int>(h),
                    {x[4*t], x[4*t+1], x[4*t+2], x[4*t+3]});
                for (int c = 0; c < cat->n_categories; ++c) {
                    CAPTURE(condition); CAPTURE(h); CAPTURE(t); CAPTURE(c);
                    CHECK(d[c] == doctest::Approx(expected[(h*trials+t)*cat->n_categories+c])
                        .epsilon(1e-6).scale(0.1));
                }
            }
        }
    }
}

TEST_CASE("geometry validates cube inputs and shares immutable compilation across threads") {
    const auto cat = gm::load_catalogue(root / "resources", 2);
    CHECK_THROWS(gm::category_distances(*cat, -1, {0,0,0,0}));
    CHECK_THROWS(gm::category_distances(*cat, 0, {-0.1,0,0,0}));
    CHECK_THROWS(gm::category_distances(*cat, 0, {NAN,0,0,0}));
    auto run = [&] { return gm::category_distances(*cat, 151, {0.041,0.017,0.9,0.8}); };
    auto first = std::async(std::launch::async, run);
    auto second = std::async(std::launch::async, run);
    CHECK(first.get() == second.get());
}
