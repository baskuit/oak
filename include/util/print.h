#pragma once

#include <iostream>
#include <type_traits>


template <typename Container>
// requires std::is_same_v<T, Container{}[0]>
void print(const Container& container) {
    for (const auto x : container) {
        std::cout << x << ' ';
    }
    std::cout << std::endl;
}