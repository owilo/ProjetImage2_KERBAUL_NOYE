#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <iostream>
#include <vector>
#include <cstdint>

// Filtre Moyenneur (filtre de dimension sizeFiltre* sizeFiltre)
void flouMoyenne(std::uint8_t* image, std::size_t width, std::size_t height, int x1, int y1, int x2, int y2, int sizeFiltre) {
    for (int i = x1; i <= x2; i++) {
        for (int j = y1; j <= y2; j++) {
            
            std::uint32_t rSum = 0;
            std::uint32_t gSum = 0;
            std::uint32_t bSum = 0;
            
            int pixelCount = 0;
            for (int bi = (sizeFiltre/2)*-1; bi < sizeFiltre/2+1; ++bi) {
                if (i+bi < 0 || i+bi >= width) continue;
                for (int bj = (sizeFiltre/2)*-1; bj < sizeFiltre/2+1; ++bj) {
                    if (j+bj < 0 || j+bj >= height) continue;
                    ++pixelCount;
                    std::size_t index = ((j + bj) * width + i + bi) * 3;
                    rSum += image[index];
                    gSum += image[index + 1];
                    bSum += image[index + 2];
                }
            }
            
            std::uint8_t r = static_cast<std::uint8_t>(rSum / pixelCount);
            std::uint8_t g = static_cast<std::uint8_t>(gSum / pixelCount);
            std::uint8_t b = static_cast<std::uint8_t>(bSum / pixelCount);
		
	    std::size_t index = (j * width + i) * 3;
            image[index] = r;
            image[index + 1] = g;
            image[index + 2] = b;
        }
    }
}

float filtreGaussien[16] = 
	{ 1, 2, 1,
	  2, 4, 2,
	  1, 2, 1 };

// Filtre Gaussien (filtre de dimension 3*3 uniquement)
void flouGauss(std::uint8_t* image, std::size_t width, std::size_t height, int x1, int y1, int x2, int y2) {
    for (int i = x1; i <= x2; i++) {
        for (int j = y1; j <= y2; j++) {
            
            std::uint32_t rSum = 0;
            std::uint32_t gSum = 0;
            std::uint32_t bSum = 0;
            
            int pixelCount = 0;
            for (int bi = -1; bi < 2; ++bi) {
                if (i+bi < 0 || i+bi >= width) continue;
                for (int bj = -1; bj < 2; ++bj) {
                    if (j+bj < 0 || j+bj >= height) continue;
                    int factor = filtreGaussien[(bj+1) * 3 + (bi+1)];
                    pixelCount += factor;
                    std::size_t index = ((j + bj) * width + i + bi) * 3;
                    rSum += image[index]*factor;
                    gSum += image[index + 1]*factor;
                    bSum += image[index + 2]*factor;
                }
            }
            
            std::uint8_t r = static_cast<std::uint8_t>(rSum / pixelCount);
            std::uint8_t g = static_cast<std::uint8_t>(gSum / pixelCount);
            std::uint8_t b = static_cast<std::uint8_t>(bSum / pixelCount);
		
	    std::size_t index = (j * width + i) * 3;
            image[index] = r;
            image[index + 1] = g;
            image[index + 2] = b;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::cerr << "Utilisation: " << argv[0] << " <entrée> <sortie> <x1> <y1> <x2> <y2> <taille du filtre>\n";
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];
    int x1 = std::stoi(argv[3]);
    int y1 = std::stoi(argv[4]);
    int x2 = std::stoi(argv[5]);
    int y2 = std::stoi(argv[6]);
    int sizeFiltre = std::stoi(argv[7]);

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

    if (sizeFiltre < 0 || !(sizeFiltre%2)) { // Pas de filtre de taille paire
        std::cerr << "Erreur : Taille de filtre invalide.\n";
        stbi_image_free(image);
        return 1;
    }

    if (x1 > x2) {
        std::swap(x1, x2);
    }
    if (y1 > y2) {
        std::swap(y1, y2);
    }

    flouMoyenne(image, width, height, x1, y1, x2, y2, sizeFiltre);
    //flouGauss(image, width, height, x1, y1, x2, y2);
    
    if (!stbi_write_png(output, width, height, channels, image, width * channels)) {
        std::cerr << "Erreur : L'image n'a pas pu être écrite.\n";
        stbi_image_free(image);
        return 1;
    }

    stbi_image_free(image);
    return 0;
}
