#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <iostream>
#include <vector>
#include <cstdint>

void masquage(std::uint8_t* image, std::size_t width, std::size_t height, int x1, int y1, int x2, int y2, int replaceR, int replaceG, int replaceB) {
    for (int i = x1; i <= x2; i++) {
        for (int j = y1; j <= y2; j++) {
            std::size_t index = (j * width + i)*3;
            image[index] = replaceR;
            image[index + 1] = replaceG;
            image[index + 2] = replaceB;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 10) {
        std::cerr << "Utilisation: " << argv[0] << " <entrée> <sortie> <x1> <y1> <x2> <y2> <remplacement R> <remplacement G> <remplacement B>\n";
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];
    int x1 = std::stoi(argv[3]);
    int y1 = std::stoi(argv[4]);
    int x2 = std::stoi(argv[5]);
    int y2 = std::stoi(argv[6]);
    // Couleur de remplacement
    int replaceR = std::stoi(argv[7]);
    int replaceG = std::stoi(argv[8]);
    int replaceB = std::stoi(argv[9]);

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

    if (replaceR < 0 || replaceR > 255 || replaceG < 0 || replaceG > 255 || replaceB < 0 || replaceB > 255) {
        std::cerr << "Erreur : Intensités de remplacement invalides.\n";
        stbi_image_free(image);
        return 1;
    }

    if (x1 > x2) {
        std::swap(x1, x2);
    }
    if (y1 > y2) {
        std::swap(y1, y2);
    }

    masquage(image, width, height, x1, y1, x2, y2, replaceR, replaceG, replaceB);

    if (!stbi_write_png(output, width, height, channels, image, width * channels)) {
        std::cerr << "Erreur : L'image n'a pas pu être écrite.\n";
        stbi_image_free(image);
        return 1;
    }

    stbi_image_free(image);
    return 0;
}
