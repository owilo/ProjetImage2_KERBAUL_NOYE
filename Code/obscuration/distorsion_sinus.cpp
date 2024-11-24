#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

std::uint8_t* distortion_sinus(std::uint8_t* image, std::size_t width, std::size_t height, int x1, int y1, int x2, int y2, float amplitude, float frequence, bool sensDistorsion) {

    std::size_t size_copie = width * height * 3;
    std::uint8_t* copie_image = new std::uint8_t[size_copie];
    std::memcpy(copie_image, image, size_copie);

    for (int i = x1; i <= x2; i++) {
        for (int j = y1; j <= y2; j++) {
            int newX;
            int newY;

            if (sensDistorsion){
                newX = i + amplitude * std::sin(2 * M_PI * j * frequence);
                newY = j;
            }else{
                newX = i;
                newY = j + amplitude * std::sin(2 * M_PI * i * frequence);
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
        std::cerr << "Utilisation: " << argv[0] << " <entrée> <sortie> <x1> <y1> <x2> <y2> <amplitude> <frequence> <sens distorsion>\n";
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];
    int x1 = std::stoi(argv[3]);
    int y1 = std::stoi(argv[4]);
    int x2 = std::stoi(argv[5]);
    int y2 = std::stoi(argv[6]);
    float amplitude = std::stof(argv[7]);
    float frequence = std::stof(argv[8]);
    bool sensDistorsion = std::stoi(argv[9]);

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

    if (amplitude <= 0 || frequence <= 0) { // Pas de filtre de taille paire
        std::cerr << "Erreur : Amplitude ou fréquence invalide.\n";
        stbi_image_free(image);
        return 1;
    }

    if (x1 > x2) {
        std::swap(x1, x2);
    }
    if (y1 > y2) {
        std::swap(y1, y2);
    }

    std::uint8_t* res = distortion_sinus(image, width, height, x1, y1, x2, y2, amplitude, frequence, sensDistorsion);
    
    if (!stbi_write_png(output, width, height, channels, res, width * channels)) {
        std::cerr << "Erreur : L'image n'a pas pu être écrite.\n";
        stbi_image_free(image);
        return 1;
    }

    stbi_image_free(image);
    return 0;
}
