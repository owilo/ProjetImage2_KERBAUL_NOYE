#define STB_IMAGE_IMPLEMENTATION
#include "../include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"

#include <iostream>
#include <tuple>
#include <vector>

std::size_t nPixelsChanged(std::uint8_t* image1, std::uint8_t* image2, std::size_t width, std::size_t height) {
    std::size_t nChanges = 0;
    for (std::size_t i = 0; i < width * height; ++i) {
        if (image1[3 * i] != image2[3 * i] || image1[3 * i + 1] != image2[3 * i + 1] || image1[3 * i + 2] != image2[3 * i + 2]) {
            ++nChanges;
        }
    }
    return nChanges;
}

double psnr(std::uint8_t* image1, std::uint8_t* image2, std::size_t width, std::size_t height) {
    double eqm{};
    for (int i{}; i < width * height; ++i) {
        for (int j{}; j < 3; ++j) {
            int diff = static_cast<int>(image1[i * 3 + j]) - image2[i * 3 + j];
            eqm += diff * diff;
        }
    }
    eqm /= (3.0 * width * height);
    return 10.0 * std::log10(65025.0 / eqm);
}

// Fonction utilisé dans le projet image 1
double ssim(unsigned char *ImgIn, unsigned char *ImgIn2, int nH, int nW) {
    double k1 = 0.01;
    double k2 = 0.03;
    int L = 255;
    double c1 = (k1 * L) * (k1 * L);
    double c2 = (k2 * L) * (k2 * L);
    double c3 = c2 / 2;
    double ssim[3] = {0};
    int count = 0;

    for (int i = 0; i < nH - 8; i += 2) {
        for (int j = 0; j < nW - 8; j += 2) {
            count++;
            int pos = i * nW + j;

            double moy1[3] = {0}, moy2[3] = {0};
            double var1[3] = {0}, var2[3] = {0};
            double cov[3] = {0};
            for (int k = 0; k < 8; k++) {
                for (int l = 0; l < 8; l++) {
                    for (int c = 0; c < 3; c++) {
                        moy1[c] += ImgIn[3 * (pos + k + l * nW) + c];
                        moy2[c] += ImgIn2[3 * (pos + k + l * nW) + c];
                    }
                }
            }

            for (int c = 0; c < 3; c++) {
                moy1[c] /= 64;
                moy2[c] /= 64;
            }
            for (int k = 0; k < 8; k++) {
                for (int l = 0; l < 8; l++) {
                    for (int c = 0; c < 3; c++) {
                        var1[c] += pow(ImgIn[3 * (pos + k + l * nW) + c] - moy1[c], 2);
                        var2[c] += pow(ImgIn2[3 * (pos + k + l * nW) + c] - moy2[c], 2);
                        cov[c] += (ImgIn[3 * (pos + k + l * nW) + c] - moy1[c]) * (ImgIn2[3 * (pos + k + l * nW) + c] - moy2[c]);
                    }
                }
            }

            for (int c = 0; c < 3; c++) {
                var1[c] /= 64;
                var2[c] /= 64;
                cov[c] /= 64;
            }
            for (int c = 0; c < 3; c++) {
                double num = (2 * moy1[c] * moy2[c] + c1) * (2 * sqrt(var1[c]) * sqrt(var2[c]) + c2) * (cov[c] + c3);
                double den = (pow(moy1[c], 2) + pow(moy2[c], 2) + c1) * (var1[c] + var2[c] + c2) * (sqrt(var1[c]) * sqrt(var2[c]) + c3);
                ssim[c] += num / den;
            }
        }
    }
    for (int c = 0; c < 3; c++) {
        ssim[c] /= count;
    }
    return (ssim[0] + ssim[1] + ssim[2]) / 3;
}

std::tuple<double, double, double, double> entropy(std::uint8_t* image, std::size_t width, std::size_t height) {
    unsigned rHisto[256]{0};
    unsigned gHisto[256]{0};
    unsigned bHisto[256]{0};
    std::size_t totalPixels = width * height;

    for (std::size_t i = 0; i < totalPixels; ++i) {
        ++rHisto[image[i * 3]];
        ++gHisto[image[i * 3 + 1]];
        ++bHisto[image[i * 3 + 2]];
    }

    double rEntropy = 0.0;
    double gEntropy = 0.0;
    double bEntropy = 0.0;
    for (std::size_t i = 0; i < 256; ++i) {
        double rp = static_cast<double>(rHisto[i]) / totalPixels;
        if (rp > 0.0) {
            rEntropy -= rp * std::log2(rp);
        }

        double gp = static_cast<double>(gHisto[i]) / totalPixels;
        if (gp > 0.0) {
            gEntropy -= gp * std::log2(gp);
        }

        double bp = static_cast<double>(bHisto[i]) / totalPixels;
        if (bp > 0.0) {
            bEntropy -= bp * std::log2(bp);
        }
    }

    return {(rEntropy + gEntropy + bEntropy) / 3.0, rEntropy, gEntropy, bEntropy};
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Utilisation: " << argv[0] << " <originale> <modifiée>\n";
        return 1;
    }

    const char* input1 = argv[1];
    const char* input2 = argv[2];

    int width;
    int height;
    int channels;
    std::uint8_t* image1 = stbi_load(input1, &width, &height, &channels, STBI_rgb);
    std::uint8_t* image2 = stbi_load(input2, &width, &height, &channels, STBI_rgb);
    if (!image1) {
        std::cerr << "Erreur : L'image originale n'a pas pu être chargée.\n";
        return 1;
    }

    if (!image2) {
        std::cerr << "Erreur : L'image modifiée n'a pas pu être chargée.\n";
        return 1;
    }

    std::cout << "PSNR : " << psnr(image1, image2, width, height) << " dB\n";
    std::cout << "SSIM : " << ssim(image1, image2, width, height) << '\n';
    std::size_t pixelsChanged{nPixelsChanged(image1, image2, width, height)};
    std::cout << "Pixels changés : " << pixelsChanged << " (" << 100.0 * pixelsChanged / (width * height) << "%)\n";
    auto [avgE1, rE1, gE1, bE1] = entropy(image1, width, height);
    std::cout << "Entropie image originale : " << avgE1 << "bits/px (par canal : [" << rE1 << ", " << gE1 << ", " << bE1 << "])\n";
    auto [avgE2, rE2, gE2, bE2] = entropy(image2, width, height);
    std::cout << "Entropie image modifiée : " << avgE2 << "bits/px (par canal : [" << rE2 << ", " << gE2 << ", " << bE2 << "])\n";

    stbi_image_free(image1);
    stbi_image_free(image2);
    return 0;
}