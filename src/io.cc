#include <iostream>
#include <string>
#include <optional>

#include <pinyon.h>

static constexpr std::array<std::string, 2> foo{};

static constexpr int battle_size{384};

std::optional<W::Search> search1{}, search2{};


void read_battle_bytes(std::array<uint8_t, battle_size>& bytes)
{
    std::cout << "#?" << std::endl;
    for (int i = 0; i < 10; ++i)
    {
        std::cin >> bytes[i];
    }
    std::cout << "#." << std::endl;
}

void set_search(std::optional<W::Search> &search) {
    // std::string; 
};


int main()
{

    std::array<uint8_t, battle_size> bytes{};
    read_battle_bytes(bytes);

    for (int i = 0; i < 5; ++i) {
        std::cout << bytes[i] << ' ';
    }
    std::cout << std::endl;

    return 0;
}