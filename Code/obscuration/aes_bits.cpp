#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include "../include/plusaes.hpp"

#include <iostream>
#include <vector>
#include <cstdint>

void encrypt_bits(std::uint8_t* image, std::size_t width, std::size_t height, int x1, int y1, int x2, int y2, const std::uint8_t* key, std::size_t keyLength, std::size_t nbBits) {
    std::vector<std::uint8_t> data(3 * (x2 - x1 + 1) * (y2 - y1 + 1));
    std::size_t pad = x2 - x1;
    for (std::size_t i = 0; i <= x2 - x1; ++i) {
        for (std::size_t j = 0; j <= y2 - y1; ++j) {
            for (int k = 0; k < 3; ++k) {
                data[3 * (j + pad * i) + k] = image[3 * ((y1 + j) * width + x1 + i) + k];
            }
        }
    }

    std::size_t encryptedSize = plusaes::get_padded_encrypted_size(data.size());
    std::vector<std::uint8_t> encryptedData(encryptedSize);

    const std::uint8_t iv[16] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    };

    plusaes::encrypt_cbc(data.data(), data.size(), &key[0], keyLength, &iv, &encryptedData[0], encryptedData.size(), true);

    for (std::size_t i = 0; i <= x2 - x1; ++i) {
        for (std::size_t j = 0; j <= y2 - y1; ++j) {
            for (int k = 0; k < 3; ++k) {
                uint8_t px_intacte = image[3 * ((y1 + j) * width + x1 + i) + k];
                std::bitset<8> bits_intactes(px_intacte);
                uint8_t px_chiffre = encryptedData[3 * (j + pad * i) + k];
                std::bitset<8> bits_chiffres(px_chiffre);

                for (unsigned int n = 0 ; n < 8 ; n++){
                    if (n < nbBits){
                        bits_intactes[n] = bits_chiffres[n];
                    }
                }

                image[3 * ((y1 + j) * width + x1 + i) + k] = static_cast<uint8_t>(bits_intactes.to_ulong());
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 9) {
        std::cerr << "Utilisation: " << argv[0] << " <entrée> <sortie> <x1> <y1> <x2> <y2> <clé> <nbBits>\n";
        return 1;
    }

    const char* input = argv[1];
    const char* output = argv[2];
    int x1 = std::stoi(argv[3]);
    int y1 = std::stoi(argv[4]);
    int x2 = std::stoi(argv[5]);
    int y2 = std::stoi(argv[6]);
    int nbBits = std::stoi(argv[8]);

    const std::uint8_t* key = reinterpret_cast<const std::uint8_t*>(argv[7]);
    std::size_t keyLength = std::strlen(argv[7]);

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

    if (keyLength != 16 && keyLength != 32) {
        std::cerr << "Erreur : La clé doit être sur 128 bits ou 256 bits (16 ou 32 caractères).\n";
        std::cerr << "Exemples de clé : 'EncryptionKey128', 'abcdef0123456789'\n";
        std::cerr << "Taille actuelle : " << keyLength << " caractères\n";
        stbi_image_free(image);
        return 1;
    }

    if (x1 > x2) {
        std::swap(x1, x2);
    }
    if (y1 > y2) {
        std::swap(y1, y2);
    }

    encrypt_bits(image, width, height, x1, y1, x2, y2, key, keyLength, nbBits);

    if (!stbi_write_png(output, width, height, channels, image, width * channels)) {
        std::cerr << "Erreur : L'image n'a pas pu être écrite.\n";
        stbi_image_free(image);
        return 1;
    }

    stbi_image_free(image);
    return 0;
}