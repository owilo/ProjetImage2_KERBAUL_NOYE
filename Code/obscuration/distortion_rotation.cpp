#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

std::uint8_t* distortion_rotation(std::uint8_t* image, std::size_t width, std::size_t height, int x1, int y1, int x2, int y2, int centre_x, int centre_y, float intensite) {

    std::size_t size_copie = width * height * 3;
    std::uint8_t* copie_image = new std::uint8_t[size_copie];
    std::memcpy(copie_image, image, size_copie);

    for (int i = x1; i <= x2; i++) {
        for (int j = y1; j <= y2; j++) {
            float dx = i - centre_x;
            float dy = j - centre_y;
            float r = sqrt(dx*dx + dy*dy);
            float theta = atan2(dy, dx);
            float new_r = tan(r * intensite);
            
            int newX = static_cast<int>(centre_x + new_r * std::cos(theta));
            int newY = static_cast<int>(centre_y + new_r * std::sin(theta));

            if (newX < 0 || newX >= width || newY < 0 || newY >= height) {
                continue;
            }

            std::size_t new_index = (newY * width + newX) * 3;

            std::uint32_t newR = image[new_index];
            std::uint32_t newG = image[new_index+1];
            std::uint32_t newB = image[new_index+2];

            std::size_t index = (j * width + i) * 3;
            copie_image[index] = newR;
            copie_image[index + 1] = newG;
            copie_image[index + 2] = newB;
        }
    }

    return copie_image;
}

int main(int argc, char* argv[]) {
    if (argc != 10) {
        std::cerr << "Utilisation: " << argv[0] << " <entrée> <sortie> <x1> <y1> <x2> <y2> <position x du centre> <position y> <intensité fisheye>\n";
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];
    int x1 = std::stoi(argv[3]);
    int y1 = std::stoi(argv[4]);
    int x2 = std::stoi(argv[5]);
    int y2 = std::stoi(argv[6]);
    int centre_x = std::stoi(argv[7]);
    int centre_y = std::stoi(argv[8]);
    float intensite = std::stof(argv[9]);

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

    std::uint8_t* res = distortion_rotation(image, width, height, x1, y1, x2, y2, centre_x, centre_y, intensite);
    
    if (!stbi_write_png(output, width, height, channels, res, width * channels)) {
        std::cerr << "Erreur : L'image n'a pas pu être écrite.\n";
        stbi_image_free(image);
        return 1;
    }

    stbi_image_free(image);
    return 0;
}
