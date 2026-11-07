# Dependencies

CMakeLists.txt pins each archive and SHA256; downloads.json records the verified archives.
Sources live only in build/vendor or CMake's build/_deps, not a second editable model tree.

| Dependency | Version | License | Role |
|---|---|---|---|
| Eigen | 3.4.0 | MPL-2.0 (see bundled notices) | geometry compilation |
| yaml-cpp | 0.8.0 | MIT | configuration |
| nlohmann/json | 3.11.3 | MIT | metadata and result schema |
| csv-parser | 2.3.0 | MIT | quoted UTF-8 CSV |
| cnpy | 4e8810b | MIT | NPZ writing |
| zlib | 1.3.1 | zlib | ZIP deflate decoding/checksums |
| PicoSHA2 | 161cb3f | MIT | content and seed identity |
| doctest | 2.4.11 | MIT | C++ tests |

The cnpy scalar NPY-header defect is handled by a local tested adapter in io.cpp.
Because upstream npz_save does not check fopen/fwrite/fclose, production uses
checked stream ZIP32 packaging around cnpy's NPY header/type helpers; it never
calls the unsafe writer. Upstream source is not modified. Reading validates CRC, numeric dtype, ZIP sizes,
endianness and C-order; Python pickle/object arrays are never interpreted.
Large ZIP64 arrays and Fortran-order arrays are deliberately unsupported.
Eigen's archive is fetched from eigen-mirror at the pinned 3.4.0 tag.
