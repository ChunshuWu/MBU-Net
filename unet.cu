// ---------------------------------------------------------------------------
// File: unet.cu
// UNet BNN inference source file. 
// ---------------------------------------------------------------------------
// Chunshu Wu, Pacific Northwest National Laboratory (PNNL), U.S.
// Richland, 99352, WA, USA. Feb-03-2026.
// ---------------------------------------------------------------------------
#include <stdio.h>
#include <string>
#include <cooperative_groups.h>
#include "utility.h"
#include "param.h"
#include "kernel.cuh"
#include "test_data_loader.h"
#include "layer_config.h"
using namespace cooperative_groups;
using namespace std;

// Conditional type definitions
#if USE_TERNARY_INPUT_CONV2
    #define InputConv2LayerParam Conv128LayerTernaryParam
    #define InputConv2LayerFMT Conv128LayerTernaryFMT
    #define InputConv2Layer Conv128LayerTernary
#else
    #define InputConv2LayerParam Conv128LayerParam
    #define InputConv2LayerFMT Conv128LayerFMT
    #define InputConv2Layer Conv128Layer
#endif

#if USE_TERNARY_DOWN1_CONV1
    #define Down1Conv1LayerParam Conv128LayerTernaryParam
    #define Down1Conv1LayerFMT Conv128LayerTernaryFMT
    #define Down1Conv1Layer Conv128LayerTernary
#else
    #define Down1Conv1LayerParam Conv128LayerParam
    #define Down1Conv1LayerFMT Conv128LayerFMT
    #define Down1Conv1Layer Conv128Layer
#endif

#if USE_TERNARY_DOWN1_CONV2
    #define Down1Conv2LayerParam Conv128LayerTernaryParam
    #define Down1Conv2LayerFMT Conv128LayerTernaryFMT
    #define Down1Conv2Layer Conv128LayerTernary
#else
    #define Down1Conv2LayerParam Conv128LayerParam
    #define Down1Conv2LayerFMT Conv128LayerFMT
    #define Down1Conv2Layer Conv128Layer
#endif

#if USE_TERNARY_DOWN2_CONV1
    #define Down2Conv1LayerParam Conv128LayerTernaryParam
    #define Down2Conv1LayerFMT Conv128LayerTernaryFMT
    #define Down2Conv1Layer Conv128LayerTernary
#else
    #define Down2Conv1LayerParam Conv128LayerParam
    #define Down2Conv1LayerFMT Conv128LayerFMT
    #define Down2Conv1Layer Conv128Layer
#endif

#if USE_TERNARY_DOWN2_CONV2
    #define Down2Conv2LayerParam Conv128LayerTernaryParam
    #define Down2Conv2LayerFMT Conv128LayerTernaryFMT
    #define Down2Conv2Layer Conv128LayerTernary
#else
    #define Down2Conv2LayerParam Conv128LayerParam
    #define Down2Conv2LayerFMT Conv128LayerFMT
    #define Down2Conv2Layer Conv128Layer
#endif

#if USE_TERNARY_DOWN3_CONV1
    #define Down3Conv1LayerParam Conv128LayerTernaryParam
    #define Down3Conv1LayerFMT Conv128LayerTernaryFMT
    #define Down3Conv1Layer Conv128LayerTernary
#else
    #define Down3Conv1LayerParam Conv128LayerParam
    #define Down3Conv1LayerFMT Conv128LayerFMT
    #define Down3Conv1Layer Conv128Layer
#endif

#if USE_TERNARY_DOWN3_CONV2
    #define Down3Conv2LayerParam Conv128LayerTernaryParam
    #define Down3Conv2LayerFMT Conv128LayerTernaryFMT
    #define Down3Conv2Layer Conv128LayerTernary
#else
    #define Down3Conv2LayerParam Conv128LayerParam
    #define Down3Conv2LayerFMT Conv128LayerFMT
    #define Down3Conv2Layer Conv128Layer
#endif

#if USE_TERNARY_DOWN4_CONV1
    #define Down4Conv1LayerParam Conv128LayerTernaryParam
    #define Down4Conv1LayerFMT Conv128LayerTernaryFMT
    #define Down4Conv1Layer Conv128LayerTernary
#else
    #define Down4Conv1LayerParam Conv128LayerParam
    #define Down4Conv1LayerFMT Conv128LayerFMT
    #define Down4Conv1Layer Conv128Layer
#endif

#if USE_TERNARY_DOWN4_CONV2
    #define Down4Conv2LayerParam Conv128LayerTernaryParam
    #define Down4Conv2LayerFMT Conv128LayerTernaryFMT
    #define Down4Conv2Layer Conv128LayerTernary
#else
    #define Down4Conv2LayerParam Conv128LayerParam
    #define Down4Conv2LayerFMT Conv128LayerFMT
    #define Down4Conv2Layer Conv128Layer
#endif

#if USE_TERNARY_UP1_TRANSPOSE
    #define Up1TransposeLayerParam ConvTranspose128LayerTernaryParam
    #define Up1TransposeLayerFMT ConvTranspose128LayerTernaryFMT
    #define Up1TransposeLayer ConvTranspose128LayerTernary
#else
    #define Up1TransposeLayerParam ConvTranspose128LayerParam
    #define Up1TransposeLayerFMT ConvTranspose128LayerFMT
    #define Up1TransposeLayer ConvTranspose128Layer
#endif

#if USE_TERNARY_UP1_CONV1
    #define Up1Conv1LayerParam Conv128LayerTernaryParam
    #define Up1Conv1LayerFMT Conv128LayerTernaryFMT
    #define Up1Conv1Layer Conv128LayerTernary
#else
    #define Up1Conv1LayerParam Conv128LayerParam
    #define Up1Conv1LayerFMT Conv128LayerFMT
    #define Up1Conv1Layer Conv128Layer
#endif

#if USE_TERNARY_UP1_CONV2
    #define Up1Conv2LayerParam Conv128LayerTernaryParam
    #define Up1Conv2LayerFMT Conv128LayerTernaryFMT
    #define Up1Conv2Layer Conv128LayerTernary
#else
    #define Up1Conv2LayerParam Conv128LayerParam
    #define Up1Conv2LayerFMT Conv128LayerFMT
    #define Up1Conv2Layer Conv128Layer
#endif

#if USE_TERNARY_UP2_TRANSPOSE
    #define Up2TransposeLayerParam ConvTranspose128LayerTernaryParam
    #define Up2TransposeLayerFMT ConvTranspose128LayerTernaryFMT
    #define Up2TransposeLayer ConvTranspose128LayerTernary
#else
    #define Up2TransposeLayerParam ConvTranspose128LayerParam
    #define Up2TransposeLayerFMT ConvTranspose128LayerFMT
    #define Up2TransposeLayer ConvTranspose128Layer
#endif

#if USE_TERNARY_UP2_CONV1
    #define Up2Conv1LayerParam Conv128LayerTernaryParam
    #define Up2Conv1LayerFMT Conv128LayerTernaryFMT
    #define Up2Conv1Layer Conv128LayerTernary
#else
    #define Up2Conv1LayerParam Conv128LayerParam
    #define Up2Conv1LayerFMT Conv128LayerFMT
    #define Up2Conv1Layer Conv128Layer
#endif

#if USE_TERNARY_UP2_CONV2
    #define Up2Conv2LayerParam Conv128LayerTernaryParam
    #define Up2Conv2LayerFMT Conv128LayerTernaryFMT
    #define Up2Conv2Layer Conv128LayerTernary
#else
    #define Up2Conv2LayerParam Conv128LayerParam
    #define Up2Conv2LayerFMT Conv128LayerFMT
    #define Up2Conv2Layer Conv128Layer
#endif

#if USE_TERNARY_UP3_TRANSPOSE
    #define Up3TransposeLayerParam ConvTranspose128LayerTernaryParam
    #define Up3TransposeLayerFMT ConvTranspose128LayerTernaryFMT
    #define Up3TransposeLayer ConvTranspose128LayerTernary
#else
    #define Up3TransposeLayerParam ConvTranspose128LayerParam
    #define Up3TransposeLayerFMT ConvTranspose128LayerFMT
    #define Up3TransposeLayer ConvTranspose128Layer
#endif

#if USE_TERNARY_UP3_CONV1
    #define Up3Conv1LayerParam Conv128LayerTernaryParam
    #define Up3Conv1LayerFMT Conv128LayerTernaryFMT
    #define Up3Conv1Layer Conv128LayerTernary
#else
    #define Up3Conv1LayerParam Conv128LayerParam
    #define Up3Conv1LayerFMT Conv128LayerFMT
    #define Up3Conv1Layer Conv128Layer
#endif

#if USE_TERNARY_UP3_CONV2
    #define Up3Conv2LayerParam Conv128LayerTernaryParam
    #define Up3Conv2LayerFMT Conv128LayerTernaryFMT
    #define Up3Conv2Layer Conv128LayerTernary
#else
    #define Up3Conv2LayerParam Conv128LayerParam
    #define Up3Conv2LayerFMT Conv128LayerFMT
    #define Up3Conv2Layer Conv128Layer
#endif

#if USE_TERNARY_UP4_TRANSPOSE
    #define Up4TransposeLayerParam ConvTranspose128LayerTernaryParam
    #define Up4TransposeLayerFMT ConvTranspose128LayerTernaryFMT
    #define Up4TransposeLayer ConvTranspose128LayerTernary
#else
    #define Up4TransposeLayerParam ConvTranspose128LayerParam
    #define Up4TransposeLayerFMT ConvTranspose128LayerFMT
    #define Up4TransposeLayer ConvTranspose128Layer
#endif

#if USE_TERNARY_UP4_CONV1
    #define Up4Conv1LayerParam Conv128LayerTernaryParam
    #define Up4Conv1LayerFMT Conv128LayerTernaryFMT
    #define Up4Conv1Layer Conv128LayerTernary
#else
    #define Up4Conv1LayerParam Conv128LayerParam
    #define Up4Conv1LayerFMT Conv128LayerFMT
    #define Up4Conv1Layer Conv128Layer
#endif

#if USE_TERNARY_UP4_CONV2
    #define Up4Conv2LayerParam Conv128LayerTernaryParam
    #define Up4Conv2LayerFMT Conv128LayerTernaryFMT
    #define Up4Conv2Layer Conv128LayerTernary
#else
    #define Up4Conv2LayerParam Conv128LayerParam
    #define Up4Conv2LayerFMT Conv128LayerFMT
    #define Up4Conv2Layer Conv128Layer
#endif

#define OutputConvLayerParam Conv128LayerParam
#define OutputConvLayerFMT Conv128LayerFMT
#define OutputConvLayer Conv128Layer
// =============================================================
// UNet kernel function
__global__ void unet128(
        InConv128LayerParam* input_conv1, 
        InputConv2LayerParam* input_conv2,
        Down1Conv1LayerParam* down1_conv1,
        Down1Conv2LayerParam* down1_conv2,
        Down2Conv1LayerParam* down2_conv1,
        Down2Conv2LayerParam* down2_conv2,
        Down3Conv1LayerParam* down3_conv1,
        Down3Conv2LayerParam* down3_conv2,
        Down4Conv1LayerParam* down4_conv1,
        Down4Conv2LayerParam* down4_conv2,
        Up1TransposeLayerParam* up1_transpose,
        Up1Conv1LayerParam* up1_conv1,
        Up1Conv2LayerParam* up1_conv2,
        Up2TransposeLayerParam* up2_transpose,
        Up2Conv1LayerParam* up2_conv1,
        Up2Conv2LayerParam* up2_conv2,
        Up3TransposeLayerParam* up3_transpose,
        Up3Conv1LayerParam* up3_conv1,
        Up3Conv2LayerParam* up3_conv2,
        Up4TransposeLayerParam* up4_transpose,
        Up4Conv1LayerParam* up4_conv1,
        Up4Conv2LayerParam* up4_conv2,
        OutputConvLayerParam* output_conv)
{
    grid_group grid = this_grid();
    //========= ENCODER (DOWN PATH) =========
    // Input Block
    InConv128LayerFMT(input_conv1);
    grid.sync();
    InputConv2LayerFMT(input_conv2); // Saves skip connection for up4
    grid.sync();
    // Down1
    Down1Conv1LayerFMT(down1_conv1);
    grid.sync();
    Down1Conv2LayerFMT(down1_conv2); // Saves skip connection for up3
    grid.sync();
    
    // Down2
    Down2Conv1LayerFMT(down2_conv1);
    grid.sync();
    Down2Conv2LayerFMT(down2_conv2); // Saves skip connection for up2
    grid.sync();
    
    // Down3
    Down3Conv1LayerFMT(down3_conv1);
    grid.sync();
    Down3Conv2LayerFMT(down3_conv2); // Saves skip connection for up1
    grid.sync();
    // Down4 (Bottleneck)
    Down4Conv1LayerFMT(down4_conv1);
    grid.sync();
    Down4Conv2LayerFMT(down4_conv2);
    grid.sync();
    //========= DECODER (UP PATH) =========
    // Up1
    Up1TransposeLayerFMT(up1_transpose);
    grid.sync();
    ConcatInputSkipBitpack128FMT(up1_conv1); 
    grid.sync();
    Up1Conv1LayerFMT(up1_conv1); // Concatenates skip connection from down3_conv2
    grid.sync();
    Up1Conv2LayerFMT(up1_conv2);
    grid.sync();
    // Up2
    Up2TransposeLayerFMT(up2_transpose);
    grid.sync();
    ConcatInputSkipBitpack128FMT(up2_conv1);
    grid.sync();
    Up2Conv1LayerFMT(up2_conv1); // Concatenates skip connection from down2_conv2
    grid.sync();
    Up2Conv2LayerFMT(up2_conv2);
    grid.sync();
    // Up3
    Up3TransposeLayerFMT(up3_transpose);
    grid.sync();
    ConcatInputSkipBitpack128FMT(up3_conv1);
    grid.sync();
    Up3Conv1LayerFMT(up3_conv1); // Concatenates skip connection from down1_conv2
    grid.sync();
    Up3Conv2LayerFMT(up3_conv2);
    grid.sync();
    // Up4
    Up4TransposeLayerFMT(up4_transpose);
    grid.sync();
    ConcatInputSkipBitpack128FMT(up4_conv1);
    grid.sync();
    Up4Conv1LayerFMT(up4_conv1); // Concatenates skip connection from input_conv2
    grid.sync();
    Up4Conv2LayerFMT(up4_conv2);
    grid.sync();
    //========= OUTPUT =========
    OutputConvLayerFMT(output_conv);
}
int main()
{
    int dev = 0;
    cudaSetDevice(dev);
    const unsigned batch = 128;  // To be adjusted for different GPUs
    const unsigned output_size = 2;  // For segmentation, this would be number of classes
    const unsigned image_width = 512;  // Input image width
    const unsigned image_height = 512; // Input image height  
    const unsigned image_channel = 3; // RGB channels
    const unsigned filter_height = 3; // 3x3 convolution kernels
    const unsigned filter_width = 3;
    //=============== Get Input and Label =================
    float* images = (float*)malloc(batch*image_height*image_width*image_channel*sizeof(float));
    unsigned* image_labels = (unsigned*)malloc(batch*sizeof(unsigned));
    
    // Use test data for UNet testing
    string test_images_path = "./test_images_" + to_string(image_width) + "x" + to_string(image_height) + ".bin";    printf("Loading test images for UNet testing...\n");
    read_test_images(test_images_path, images, image_labels, batch);
    //================ Get Weight =================
    FILE* config_file = fopen("./unet_test_weights.csv","r");
    if (config_file == NULL) {
        fprintf(stderr, "Error: Cannot open weight file: unet_test_weights.csv\n");
        fprintf(stderr, "Make sure to run: python3 create_test_data.py\n");
        exit(1);
    }
    printf("Loading UNet weights from: unet_test_weights.csv\n");

    //========= ENCODER (DOWN PATH) =========
    // Save skip connection for up4
    InConv128LayerParam* input_conv1 = new InConv128LayerParam("InputConv1", image_height, image_width, 
            filter_height, filter_width, image_channel, 64, batch); 
    InConv128LayerParam* input_conv1_gpu = input_conv1->initialize(images, config_file);
    
    InputConv2LayerParam* input_conv2 = new InputConv2LayerParam("InputConv2", input_conv1->output_height, 
            input_conv1->output_width, filter_height, filter_width, 64, 64, batch, 1, 1, true, 2, 2, true, 64, false);
    InputConv2LayerParam* input_conv2_gpu = input_conv2->initialize(config_file, input_conv1->get_output_gpu());
    // Save skip connection for up3
    Down1Conv1LayerParam* down1_conv1 = new Down1Conv1LayerParam("Down1Conv1", input_conv2->output_height, 
            input_conv2->output_width, filter_height, filter_width, 64, 128, batch);
    Down1Conv1LayerParam* down1_conv1_gpu = down1_conv1->initialize(config_file, input_conv2->get_output_gpu());
    
    Down1Conv2LayerParam* down1_conv2 = new Down1Conv2LayerParam("Down1Conv2", down1_conv1->output_height, 
            down1_conv1->output_width, filter_height, filter_width, 128, 128, batch, 1, 1, true, 2, 2, true, 128, false);
    Down1Conv2LayerParam* down1_conv2_gpu = down1_conv2->initialize(config_file, down1_conv1->get_output_gpu());
    // Save skip connection for up2
    Down2Conv1LayerParam* down2_conv1 = new Down2Conv1LayerParam("Down2Conv1", down1_conv2->output_height, 
            down1_conv2->output_width, filter_height, filter_width, 128, 256, batch);
    Down2Conv1LayerParam* down2_conv1_gpu = down2_conv1->initialize(config_file, down1_conv2->get_output_gpu());
    
    Down2Conv2LayerParam* down2_conv2 = new Down2Conv2LayerParam("Down2Conv2", down2_conv1->output_height, 
            down2_conv1->output_width, filter_height, filter_width, 256, 256, batch, 1, 1, true, 2, 2, true, 256, false);
    Down2Conv2LayerParam* down2_conv2_gpu = down2_conv2->initialize(config_file, down2_conv1->get_output_gpu());
    // Save skip connection for up1
    Down3Conv1LayerParam* down3_conv1 = new Down3Conv1LayerParam("Down3Conv1", down2_conv2->output_height, 
            down2_conv2->output_width, filter_height, filter_width, 256, 512, batch);
    Down3Conv1LayerParam* down3_conv1_gpu = down3_conv1->initialize(config_file, down2_conv2->get_output_gpu());
    
    Down3Conv2LayerParam* down3_conv2 = new Down3Conv2LayerParam("Down3Conv2", down3_conv1->output_height, 
            down3_conv1->output_width, filter_height, filter_width, 512, 512, batch, 1, 1, true, 2, 2, true, 512, false);
    Down3Conv2LayerParam* down3_conv2_gpu = down3_conv2->initialize(config_file, down3_conv1->get_output_gpu());
    // Down4 (Bottleneck)
    Down4Conv1LayerParam* down4_conv1 = new Down4Conv1LayerParam("Down4Conv1", down3_conv2->output_height, 
            down3_conv2->output_width, filter_height, filter_width, 512, 512, batch);
    Down4Conv1LayerParam* down4_conv1_gpu = down4_conv1->initialize(config_file, down3_conv2->get_output_gpu());
    
    Down4Conv2LayerParam* down4_conv2 = new Down4Conv2LayerParam("Down4Conv2", down4_conv1->output_height, 
            down4_conv1->output_width, filter_height, filter_width, 512, 512, batch);
    Down4Conv2LayerParam* down4_conv2_gpu = down4_conv2->initialize(config_file, down4_conv1->get_output_gpu());
    //========= DECODER (UP PATH) =========
    // Up1
    Up1TransposeLayerParam* up1_transpose = new Up1TransposeLayerParam("Up1Transpose", 
            down4_conv2->output_height, down4_conv2->output_width, 2, 2, 512, 512, batch);
    Up1TransposeLayerParam* up1_transpose_gpu = up1_transpose->initialize(config_file, down4_conv2->get_output_gpu());
    
    Up1Conv1LayerParam* up1_conv1 = new Up1Conv1LayerParam("Up1Conv1", up1_transpose->output_height, 
            up1_transpose->output_width, filter_height, filter_width, 1024, 256, batch, 1, 1, true, 1, 1, false, 512, true);
    Up1Conv1LayerParam* up1_conv1_gpu = up1_conv1->initialize(config_file, up1_transpose->get_output_gpu(), down3_conv2->get_output_skip_connection_gpu());
    
    Up1Conv2LayerParam* up1_conv2 = new Up1Conv2LayerParam("Up1Conv2", up1_conv1->output_height, 
            up1_conv1->output_width, filter_height, filter_width, 256, 256, batch);
    Up1Conv2LayerParam* up1_conv2_gpu = up1_conv2->initialize(config_file, up1_conv1->get_output_gpu());
    // Up2
    Up2TransposeLayerParam* up2_transpose = new Up2TransposeLayerParam("Up2Transpose", 
            up1_conv2->output_height, up1_conv2->output_width, 2, 2, 256, 256, batch);
    Up2TransposeLayerParam* up2_transpose_gpu = up2_transpose->initialize(config_file, up1_conv2->get_output_gpu());
    
    Up2Conv1LayerParam* up2_conv1 = new Up2Conv1LayerParam("Up2Conv1", up2_transpose->output_height, 
            up2_transpose->output_width, filter_height, filter_width, 512, 128, batch, 1, 1, true, 1, 1, false, 256, true);
    Up2Conv1LayerParam* up2_conv1_gpu = up2_conv1->initialize(config_file, up2_transpose->get_output_gpu(), down2_conv2->get_output_skip_connection_gpu());
    
    Up2Conv2LayerParam* up2_conv2 = new Up2Conv2LayerParam("Up2Conv2", up2_conv1->output_height, 
            up2_conv1->output_width, filter_height, filter_width, 128, 128, batch);
    Up2Conv2LayerParam* up2_conv2_gpu = up2_conv2->initialize(config_file, up2_conv1->get_output_gpu());
    // Up3
    Up3TransposeLayerParam* up3_transpose = new Up3TransposeLayerParam("Up3Transpose", 
            up2_conv2->output_height, up2_conv2->output_width, 2, 2, 128, 128, batch);
    Up3TransposeLayerParam* up3_transpose_gpu = up3_transpose->initialize(config_file, up2_conv2->get_output_gpu());
    
    Up3Conv1LayerParam* up3_conv1 = new Up3Conv1LayerParam("Up3Conv1", up3_transpose->output_height, 
            up3_transpose->output_width, filter_height, filter_width, 256, 64, batch, 1, 1, true, 1, 1, false, 128, true);
    Up3Conv1LayerParam* up3_conv1_gpu = up3_conv1->initialize(config_file, up3_transpose->get_output_gpu(), down1_conv2->get_output_skip_connection_gpu());
    
    Up3Conv2LayerParam* up3_conv2 = new Up3Conv2LayerParam("Up3Conv2", up3_conv1->output_height, 
            up3_conv1->output_width, filter_height, filter_width, 64, 64, batch);
    Up3Conv2LayerParam* up3_conv2_gpu = up3_conv2->initialize(config_file, up3_conv1->get_output_gpu());
    // Up4
    Up4TransposeLayerParam* up4_transpose = new Up4TransposeLayerParam("Up4Transpose", 
            up3_conv2->output_height, up3_conv2->output_width, 2, 2, 64, 64, batch);
    Up4TransposeLayerParam* up4_transpose_gpu = up4_transpose->initialize(config_file, up3_conv2->get_output_gpu());
    
    Up4Conv1LayerParam* up4_conv1 = new Up4Conv1LayerParam("Up4Conv1", up4_transpose->output_height, 
            up4_transpose->output_width, filter_height, filter_width, 128, 64, batch, 1, 1, true, 1, 1, false, 64, true);
    Up4Conv1LayerParam* up4_conv1_gpu = up4_conv1->initialize(config_file, up4_transpose->get_output_gpu(), input_conv2->get_output_skip_connection_gpu());
    
    Up4Conv2LayerParam* up4_conv2 = new Up4Conv2LayerParam("Up4Conv2", up4_conv1->output_height, 
            up4_conv1->output_width, filter_height, filter_width, 64, 64, batch);
    Up4Conv2LayerParam* up4_conv2_gpu = up4_conv2->initialize(config_file, up4_conv1->get_output_gpu());
    //========= OUTPUT =========
    OutputConvLayerParam* output_conv = new OutputConvLayerParam("OutputConv", up4_conv2->output_height, 
            up4_conv2->output_width, 1, 1, 64, output_size, batch);
    OutputConvLayerParam* output_conv_gpu = output_conv->initialize(config_file, up4_conv2->get_output_gpu());

    //================ Setup Kernel =================
    int numThreads = 256;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    int shared_memory = 512*sizeof(int)*32;
    cudaFuncSetAttribute(unet128, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, unet128, numThreads, shared_memory);

    // UNet kernel args array
    void* args[] = {&input_conv1_gpu, &input_conv2_gpu, &down1_conv1_gpu, &down1_conv2_gpu, 
        &down2_conv1_gpu, &down2_conv2_gpu, &down3_conv1_gpu, &down3_conv2_gpu, &down4_conv1_gpu, 
        &down4_conv2_gpu, &up1_transpose_gpu, &up1_conv1_gpu, &up1_conv2_gpu, &up2_transpose_gpu,
        &up2_conv1_gpu, &up2_conv2_gpu, &up3_transpose_gpu, &up3_conv1_gpu, &up3_conv2_gpu,
        &up4_transpose_gpu, &up4_conv1_gpu, &up4_conv2_gpu, &output_conv_gpu};
    
    START_TIMER;
    cudaLaunchCooperativeKernel((void*)unet128, numBlocksPerSm*deviceProp.multiProcessorCount, 
            numThreads, args, shared_memory);
    STOP_TIMER;
    CUDA_CHECK_KERNEL();
    
    //================ Output =================
    float* output = output_conv->download_full_output();
    // Cleanup UNet layers
    delete input_conv1;
    delete input_conv2;
    delete down1_conv1;
    delete down1_conv2;
    delete down2_conv1;
    delete down2_conv2;
    delete down3_conv1;
    delete down3_conv2;
    delete down4_conv1;
    delete down4_conv2;
    delete up1_transpose;
    delete up1_conv1;
    delete up1_conv2;
    delete up2_transpose;
    delete up2_conv1;
    delete up2_conv2;
    delete up3_transpose;
    delete up3_conv1;
    delete up3_conv2;
    delete up4_transpose;
    delete up4_conv1;
    delete up4_conv2;
    delete output_conv;
    return 0;
}