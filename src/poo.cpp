#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>
#include <array>
#include <cstdint>
#include <filesystem>

int write() {
    std::filesystem::path path{"./buffer"};
    std::ofstream outFile(path, std::ios::binary);
    
    if (!outFile) {
        std::cerr << "Failed to open file for writing\n";
        return 1;
    } else {
        std::cout << "Successfully opened file." << std::endl;
    }

    std::array<uint8_t, 384> battle{};
    for (int i = 0; i < 384; ++i) {
        battle[i] = uint8_t{i};
    }

    for (int i = 0; i < 100; ++i) {
        outFile.write(reinterpret_cast<const char*>(battle.data()), battle.size());
        float f = i;
        outFile.write(reinterpret_cast<const char*>(&f), sizeof(float));
    }

    // Close the file (optional — happens automatically on destruction)
    outFile.close();

    return 0;
}

int read() {
    std::filesystem::path path{"./buffer"};
    std::ifstream inFile(path, std::ios::binary);

    for (int i = 0; i < 100; ++i) {
        auto index = 384 * i;

        

    }

    return 0;
}

int main () {
    return read();
}
