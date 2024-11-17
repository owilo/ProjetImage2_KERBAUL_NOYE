#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <algorithm>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

std::uint8_t* distortion_RGB(std::uint8_t* image, int width, int height, int x1, int y1, int x2, int y2, int dxR, int dyG, int dxB) {

    std::size_t size_copie = width * height * 3;
    std::uint8_t* copie_image = new std::uint8_t[size_copie];
    std::memcpy(copie_image, image, size_copie);

    for (int i = x1; i <= x2; i++) {
        for (int j = y1; j <= y2; j++) {
            int newX;
            int newY;

            std::size_t new_index = (newY * width + newX) * 3;

            std::uint32_t new_xR = std::clamp(i + dxR, 0, width -1); 
            std::uint32_t new_yG = std::clamp(j + dyG, 0, height -1);
            std::uint32_t new_xB = std::clamp(i + dxB, 0, width -1);

            std::size_t index = (j * width + i) * 3;
            copie_image[index] = image[(j * width + new_xR) * 3];
            copie_image[index + 1] = image[(new_yG * width + i) * 3 + 1];
            copie_image[index + 2] = image[(j * width + new_xB) * 3 + 2];
        }
    }

    return copie_image;
}

int main(int argc, char* argv[]) {
    if (argc != 10) {
        std::cerr << "Utilisation: " << argv[0] << " <entrée> <sortie> <x1> <y1> <x2> <y2> <decalage R> <decalage G> <decalage B>\n";
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];
    int x1 = std::stoi(argv[3]);
    int y1 = std::stoi(argv[4]);
    int x2 = std::stoi(argv[5]);
    int y2 = std::stoi(argv[6]);
    int dxR = std::stoi(argv[7]);
    int dyG = std::stoi(argv[8]);
    int dxB = std::stoi(argv[9]);
    

    int width;
    int height;
    int channels;
    std::uint8_t* image = stbi_load(input, &width, &height, &channels, STBI_rgb);
    if (!image) {
        std::cerr << "Erreur : L'image n'a pas pu être chargée.\n";
        return 1;
    }

    if (x1 < 0 || y1 < 0 || x2 > width || y2 > height) {
        std::cerr << "Erreur : Coordonnées invalides.\n";
        stbi_image_free(image);
        return 1;
    }

    if (x1 > x2) {
        std::swap(x1, x2);
    }
    if (y1 > y2) {
        std::swap(y1, y2);
    }

    std::uint8_t* res = distortion_RGB(image, width, height, x1, y1, x2, y2, dxR, dyG, dxB);
    
    if (!stbi_write_png(output, width, height, channels, res, width * channels)) {
        std::cerr << "Erreur : L'image n'a pas pu être écrite.\n";
        stbi_image_free(image);
        return 1;
    }

    stbi_image_free(image);
    return 0;
}
