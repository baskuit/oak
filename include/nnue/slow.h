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

// Definition of layer AffineTransform of NNUE evaluation function

#pragma once

#include <cstdint>
#include <iostream>

#include "nnue_common.h"
#include "simd.h"

namespace NNUE {

template <size_t InDims, size_t OutDims> class SlowLayer {
public:
  using InputType = float;
  using OutputType = float;

  static constexpr size_t InputDimensions = InDims;
  static constexpr size_t OutputDimensions = OutDims;
  using OutputBuffer = std::array<OutputType, OutDims>;

  bool read_parameters(std::istream &stream_weight, std::istream &stream_bias) {
    read_little_endian_float<BiasType>(stream_bias, biases.data(),
                                       OutputDimensions);
    for (size_t i = 0; i < OutputDimensions * InputDimensions; ++i) {
      weights[i / InputDimensions][i % InputDimensions] =
          read_little_endian_float<WeightType>(stream_weight);
    }
    return !(stream_weight.fail() || stream_bias.fail());
  }

  void propagate(const InputType *input, OutputType *output) const {
    for (size_t i = 0; i < OutputDimensions; ++i) {
      for (size_t j = 0; j < InputDimensions; ++j) {
        output[i] += weights[i][j] * input[j];
      }
      output[i] += biases[i];
      output[i] = std::min<float>(std::max<float>(0.0, output[i]), 1.0);
    }
  }

public:
  using BiasType = OutputType;
  using WeightType = float;

  std::array<std::array<WeightType, InputDimensions>, OutputDimensions> weights;
  std::array<BiasType, OutputDimensions> biases;
};

} // namespace NNUE