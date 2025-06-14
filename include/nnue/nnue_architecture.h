/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// Input features and network structure used in NNUE evaluation function

#ifndef NNUE_ARCHITECTURE_H_INCLUDED
#define NNUE_ARCHITECTURE_H_INCLUDED

#include <cstdint>
#include <cstring>
#include <iosfwd>

// #include "features/half_ka_v2_hm.h"
#include "affine_transform.h"
#include "slow.h"
// #include "layers/affine_transform_sparse_input.h"
#include "clipped_relu.h"
// #include "layers/sqr_clipped_relu.h"
#include "nnue_common.h"

namespace NNUE {

struct NetworkArchitecture {
  static constexpr IndexType ConcatenatedSidesDims = 512;
  static constexpr int FC_0_OUTPUTS = 32;
  static constexpr int FC_1_OUTPUTS = 32;

  Layers::AffineTransform<ConcatenatedSidesDims, FC_0_OUTPUTS> fc_0;
  Layers::ClippedReLU<FC_0_OUTPUTS> ac_0;
  Layers::AffineTransform<FC_0_OUTPUTS, FC_1_OUTPUTS> fc_1;
  Layers::ClippedReLU<FC_1_OUTPUTS> ac_1;
  Layers::AffineTransform<FC_1_OUTPUTS, 1> fc_2;

  // Read network parameters
  bool read_parameters(std::istream &stream) {
    return fc_0.read_parameters(stream) && ac_0.read_parameters(stream) &&
           fc_1.read_parameters(stream) && ac_1.read_parameters(stream) &&
           fc_2.read_parameters(stream);
  }

  // Write network parameters
  bool write_parameters(std::ostream &stream) const {
    return fc_0.write_parameters(stream) && ac_0.write_parameters(stream) &&
           fc_1.write_parameters(stream) && ac_1.write_parameters(stream) &&
           fc_2.write_parameters(stream);
  }

  std::int32_t propagate(const TransformedFeatureType *transformedFeatures) {
    struct alignas(CacheLineSize) Buffer {
      alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
      alignas(CacheLineSize) typename decltype(ac_0)::OutputBuffer ac_0_out;
      alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
      alignas(CacheLineSize) typename decltype(ac_1)::OutputBuffer ac_1_out;
      alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;

      Buffer() { std::memset(this, 0, sizeof(*this)); }
    };

#if defined(__clang__) && (__APPLE__)
    // workaround for a bug reported with xcode 12
    static thread_local auto tlsBuffer = std::make_unique<Buffer>();
    // Access TLS only once, cache result.
    Buffer &buffer = *tlsBuffer;
#else
    alignas(CacheLineSize) static thread_local Buffer buffer;
#endif

    fc_0.propagate(transformedFeatures, buffer.fc_0_out);
    ac_0.propagate(buffer.fc_0_out, buffer.ac_0_out);
    fc_1.propagate(buffer.ac_0_out, buffer.fc_1_out);
    ac_1.propagate(buffer.fc_1_out, buffer.ac_1_out);
    fc_2.propagate(buffer.ac_1_out, buffer.fc_2_out);

    // buffer.fc_0_out[FC_0_OUTPUTS] is such that 1.0 is equal to
    // 127*(1<<WeightScaleBits) in quantized form, but we want 1.0 to be equal
    // to 600*OutputScale
    std::int32_t outputValue = buffer.fc_2_out[0];

    return outputValue;
  }
};

template <int In, int Hidden, int Out> struct WordNet {
  static constexpr IndexType ConcatenatedSidesDims = In;
  static constexpr int FC_0_OUTPUTS = Hidden;
  static constexpr int FC_1_OUTPUTS = Hidden;

  SlowLayer<In, Hidden> fc_0;
  SlowLayer<Hidden, Hidden> fc_1;
  SlowLayer<Hidden, Out> fc_2;

  std::array<std::uint8_t, Out> propagate(const float *transformedFeatures,
                                          bool print = false) {
    struct Buffer {
      alignas(CacheLineSize) typename decltype(fc_0)::OutputBuffer fc_0_out;
      alignas(CacheLineSize) typename decltype(fc_1)::OutputBuffer fc_1_out;
      alignas(CacheLineSize) typename decltype(fc_2)::OutputBuffer fc_2_out;

      Buffer() { std::memset(this, 0, sizeof(*this)); }
    };

    static thread_local Buffer buffer;

    fc_0.propagate(transformedFeatures, buffer.fc_0_out.data());
    fc_1.propagate(buffer.fc_0_out.data(), buffer.fc_1_out.data());
    fc_2.propagate(buffer.fc_1_out.data(), buffer.fc_2_out.data());
    std::array<std::uint8_t, Out> output;
    for (IndexType i = 0; i < Out; ++i) {
      output[i] = static_cast<std::uint8_t>(255 * buffer.fc_2_out[i]);
    }
    const auto arr_print = [](const auto &v) {
      for (const auto x : v) {
        std::cout << x << ' ';
      }
      std::cout << std::endl;
    };

    if (print) {
      std::cout << "input" << std::endl;
      std::cout << "fc0 out" << std::endl;
      arr_print(buffer.fc_0_out);
      std::cout << "fc1 out" << std::endl;
      arr_print(buffer.fc_1_out);
      std::cout << "fc2 out" << std::endl;
      arr_print(buffer.fc_2_out);
    }
    return output;
  }
};

} // namespace NNUE

#endif // #ifndef NNUE_ARCHITECTURE_H_INCLUDED
