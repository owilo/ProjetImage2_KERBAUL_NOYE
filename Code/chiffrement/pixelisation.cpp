#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <iostream>
#include <vector>

void pixelise(std::uint8_t* image, std::size_t width, std::size_t height, int x1, int y1, int x2, int y2, int pixelSize) {
    for (int i = x1; i <= x2; i += pixelSize) {
        for (int j = y1; j <= y2; j += pixelSize) {
            int blockWidth = std::min(pixelSize, x2 - i + 1);
            int blockHeight = std::min(pixelSize, y2 - j + 1);

            std::uint32_t rSum = 0;
            std::uint32_t gSum = 0;
            std::uint32_t bSum = 0;

            for (int bi = 0; bi < blockWidth; ++bi) {
                for (int bj = 0; bj < blockHeight; ++bj) {
                    std::size_t index = ((j + bj) * width + i + bi) * 3;
                    rSum += image[index];
                    gSum += image[index + 1];
                    bSum += image[index + 2];
                }
            }

            std::uint32_t pixelCount = blockWidth * blockHeight;
            std::uint8_t r = static_cast<std::uint8_t>(rSum / pixelCount);
            std::uint8_t g = static_cast<std::uint8_t>(gSum / pixelCount);
            std::uint8_t b = static_cast<std::uint8_t>(bSum / pixelCount);

            for (int bi = 0; bi < blockWidth; ++bi) {
                for (int bj = 0; bj < blockHeight; ++bj) {
                    std::size_t index = ((j + bj) * width + i + bi) * 3;
                    image[index] = r;
                    image[index + 1] = g;
                    image[index + 2] = b;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::cerr << "Utilisation: " << argv[0] << " <entrée> <sortie> <x1> <y1> <x2> <y2> <taille de pixel>\n";
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];
    int x1 = std::stoi(argv[3]);
    int y1 = std::stoi(argv[4]);
    int x2 = std::stoi(argv[5]);
    int y2 = std::stoi(argv[6]);
    int pixelSize = std::stoi(argv[7]);

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

    if (pixelSize < 1) {
        std::cerr << "Erreur : Taille de pixels invalides.\n";
        stbi_image_free(image);
        return 1;
    }

    if (x1 > x2) {
        std::swap(x1, x2);
    }
    if (y1 > y2) {
        std::swap(y1, y2);
    }

    pixelise(image, width, height, x1, y1, x2, y2, pixelSize);

    if (!stbi_write_png(output, width, height, channels, image, width * channels)) {
        std::cerr << "Erreur : L'image n'a pas pu être écrite.\n";
        stbi_image_free(image);
        return 1;
    }

    stbi_image_free(image);
    return 0;
}