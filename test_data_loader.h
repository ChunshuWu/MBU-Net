// ---------------------------------------------------------------------------
// File: test_data_loader.h  
// test data loader for UNet testing with 512x512 images
// ---------------------------------------------------------------------------
// Chunshu Wu, Pacific Northwest National Laboratory (PNNL), U.S.
// Richland, 99352, WA, USA. Feb-03-2026.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <fstream>
#include <algorithm>

void read_test_images(std::string filename, float* images, unsigned* labels, const unsigned batch)
{
    const unsigned width = 512; 
    const unsigned height = 512;
    const unsigned channels = 3;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error: Cannot open test image file: %s\n", filename.c_str());
        exit(1);
    }
    
    printf("Loading test images from: %s\n", filename.c_str());
    printf("Expected format: batch=%d, width=%d, height=%d, channels=%d\n", 
           batch, width, height, channels);
    
    // Temporary buffer for HWC format
    float* temp_image = new float[height * width * channels];
    
    for (unsigned i = 0; i < batch; i++) {
        // Read label (1 byte)
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = static_cast<unsigned>(label);
        
        // Read image data (HWC format)
        file.read(reinterpret_cast<char*>(temp_image), height * width * channels * sizeof(float));
        
        if (!file.good()) {
            fprintf(stderr, "Error: Failed to read image %d from file\n", i);
            delete[] temp_image;
            exit(1);
        }
        
        // Convert from HWC to CHW format
        for (unsigned c = 0; c < channels; c++) {
            for (unsigned h = 0; h < height; h++) {
                for (unsigned w = 0; w < width; w++) {
                    // CHW: images[i][c][h][w] = temp_image[h][w][c]
                    unsigned chw_idx = i * channels * height * width + c * height * width + h * width + w;
                    unsigned hwc_idx = h * width * channels + w * channels + c;
                    images[chw_idx] = temp_image[hwc_idx];
                }
            }
        }
    }
    
    delete[] temp_image;
    file.close();
    
    printf("Successfully loaded %d images of size %dx%dx%d\n", batch, width, height, channels);
    printf("Image value range: [%.3f, %.3f]\n", 
           *std::min_element(images, images + batch * channels * height * width),
           *std::max_element(images, images + batch * channels * height * width));
}