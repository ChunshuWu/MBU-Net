// ---------------------------------------------------------------------------
// File: param.h
// Define basic layer objects.
// ---------------------------------------------------------------------------
// Chunshu Wu, Pacific Northwest National Laboratory (PNNL), U.S.
// Richland, 99352, WA, USA. Feb-03-2026.
// ---------------------------------------------------------------------------

#ifndef PARAM_H
#define PARAM_H

#include "utility.h"

//================================ Convolution ====================================
__global__ void PackFiltersByInChannels128FMT(const float* __restrict__ filter, 
        unsigned* filter_binarized, const int input_channels, const int output_channels, 
        const int filter_width, const int filter_height) 
{
    GET_LANEID;
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over output_channels
    const int ins = 4*STEP128(input_channels);//condense C:in_channel into 32bit-unsigned

    for (int c=0; c<ins; c++) //iter over C:input_channels
    {
        // From shape[filter_height, filter_width, input_channels, output_channels] 
        float f0 = ((c*32+laneid)<input_channels)? filter[bx*input_channels*output_channels 
            + (c*32+laneid)*output_channels + by]:-1.0f;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0));
        if (laneid == 0) //avoid warp conflict
            filter_binarized[bx*PAD32(output_channels)*ins
                + ((by/8)*(ins/4)+c/4)*32+(by%8)*4+(c%4)] = r0;
    }
}

__global__ void PackFiltersByOutChannels32(const float* __restrict__ filter, 
        unsigned* filter_binarized, const int in_channels, const int out_channels, 
        const int filter_width, const int filter_height) 
{
    GET_LANEID;
    const int bx = blockIdx.x;//iter over (filter_width*filter_height)
    const int by = blockIdx.y;//iter over input_channels
    const int ots = STEP32(out_channels);//condense K:output_channel into 32bit-unsigned

    for (int k=0; k<ots; k++) //iter over K:output_channels
    {
        // From shape[filter_height, filter_width, in_channels, out_channels] 
        float f0 = ((k*32+laneid)<out_channels)? filter[bx*in_channels*out_channels 
            + by*out_channels + k*32 + laneid]:-1.0f;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>=0));
        // To shape[filter_height, filter_width, in_channels, out_channels/32]
        filter_binarized[bx*ots*in_channels+ by*ots + k] = r0;
    }
}

__global__ void UnpackConvOutput32FMT(const unsigned* __restrict__ input_binarized, 
        float* input, const int input_height, const int input_width,
        const int input_channels, const int batch) 
{
    GET_LANEID;
    const int bx = blockIdx.x;//input_width
    const int by = blockIdx.y;//input_height
    const int bz = blockIdx.z;//batch
    const int ins = STEP128(input_channels);//condense C:in_channel into 32bit-unsigned
    for (int c=0; c<ins*4; c++) //iter over C:in_channels
    {
        // From shape[input_height, input_width, batch, in_channels/32]
        unsigned r0 = input_binarized[by*input_width*PAD8(batch)*ins*4 + bx*PAD8(batch)*ins*4 
            + ((bz/8)*ins+c/4)*32 + (bz%8)*4+(c%4)];

        // To shape[batch, input_height, input_width, in_channels]
        if (c*32+laneid<input_channels)
        {
            input[bz*input_height*input_width*input_channels + by*input_width*input_channels
                + bx*input_channels + c*32 + laneid] = 2*(float)((r0>>(31-laneid)) & 0x1)-1;
        }
    }
}

class InConv128LayerParam
{
    public:
        InConv128LayerParam(const char* name, int _input_height, int _input_width, 
                int _filter_height, int _filter_width, int _input_channels, 
                int _output_channels, int _batch, int _stride_height=1, 
                int _stride_width=1, bool _padding=true, int _pool_height=1, 
                int _pool_width=1) :
            input_height(_input_height), input_width(_input_width), filter_height(_filter_height),
            filter_width(_filter_width), input_channels(_input_channels),
            output_channels(_output_channels), batch(_batch), stride_height(_stride_height),
            stride_width(_stride_width), pool_height(_pool_height), pool_width(_pool_width),
            padding(_padding), bn(NULL), filter(NULL), output(NULL), output_gpu(NULL), input(NULL), 
            input_gpu(NULL), gpu(NULL)

        {
            strncpy(this->name, name, 8);
            this->pad_h = padding?((( (input_height+stride_height-(input_height%stride_height))
                            /stride_height-1)*stride_height+filter_height-input_height)/2):0;
            this->pad_w = padding?((( (input_width+stride_width-(input_width%stride_width))
                                /stride_width-1)*stride_width+filter_width-input_width)/2):0; 
            int buf_height = padding?(input_height+stride_height-1)/stride_height
                    :((input_height-filter_height)/stride_height+1);
            int buf_width = padding?(input_width+stride_width-1)/stride_width
                    :((input_width-filter_width)/stride_width+1);
            output_height = (buf_height+pool_height-1)/pool_height;//pooling height
            output_width = (buf_width+pool_width-1)/pool_width; //pooling width
        }
        InConv128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(this->gpu, sizeof(InConv128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, 
                        sizeof(InConv128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }
        int input_size() { return input_channels*input_height*input_width*batch;}
        int input_bytes() { return input_size()*sizeof(float);}
        int input_bit_size() { return  input_channels*input_height*input_width*batch;}
        int input_bit_bytes() {return input_bit_size()*sizeof(float);}
        int filter_size() { return output_channels*input_channels*filter_height*filter_width;}
        int filter_bytes() { return filter_size()*sizeof(float);}
        int filter_bit_size() {return PAD128(output_channels)*STEP128(input_channels)
            *filter_height*filter_width;}
        int filter_bit_bytes() { return filter_bit_size() * sizeof(uin128);}
        int output_size() { return output_channels*output_height*output_width*batch;}
        int output_bytes() { return output_size()*sizeof(uin32);}
        int output_bit_size() { return STEP128(output_channels)*output_height
            *output_width*PAD8(batch); }
        int output_bit_bytes() { return output_bit_size() * sizeof(uin128); }
        int bn_size() { return output_channels;}
        int bn_bytes() { return bn_size()*sizeof(float);}

        InConv128LayerParam* initialize(float* input, FILE* config_file)
        {
            //Process input
            CHECK_NULL_POINTER(input);
            this->input = input;
            SAFE_ALOC_GPU(input_gpu, input_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(input_gpu, input, 
                        input_bytes(), cudaMemcpyHostToDevice) );
            //Process weight
            SAFE_ALOC_HOST(filter, filter_bytes());
            launch_array(config_file, this->filter, filter_size());
            SAFE_ALOC_GPU(filter_gpu, filter_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(filter_gpu, 0, filter_bit_bytes()) );
            float* filter_float = NULL;
            SAFE_ALOC_GPU(filter_float, filter_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(filter_float, filter, 
                        filter_bytes(), cudaMemcpyHostToDevice) );
            //Binarize Filter
            PackFiltersByOutChannels32<<<dim3(filter_height*filter_width, input_channels), 32>>>(
                    filter_float, filter_gpu, input_channels, 
                    output_channels, filter_width, filter_height);
            SAFE_FREE_GPU(filter_float);
            //Process bn
            SAFE_ALOC_HOST(bn, bn_bytes());
            launch_array(config_file, bn, bn_size());
            SAFE_ALOC_GPU(bn_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(this->bn_gpu, this->bn, 
                        bn_bytes(), cudaMemcpyHostToDevice) );
            //Allocate output
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            return this->ready();
        }
        uin32* get_output_gpu() { return this->output_gpu; }

        unsigned* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }
        float* download_full_output()
        {
            float* full_output = NULL;
            SAFE_ALOC_HOST(full_output, output_bytes());
            float* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, output_bytes());
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, output_bytes()) );
            UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch);
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, 
                        output_bytes(), cudaMemcpyDeviceToHost) );
            SAFE_FREE_GPU(full_output_gpu);
            return full_output;
        }
        void release()
        {
            SAFE_FREE_HOST(filter);
            SAFE_FREE_HOST(bn);
            SAFE_FREE_HOST(output);
            SAFE_FREE_GPU(input_gpu);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(filter_gpu);
            SAFE_FREE_GPU(bn_gpu);
            SAFE_FREE_GPU(gpu);
        }
        ~InConv128LayerParam() { release(); }
    public:
        float* input;
        float* input_gpu;
        int input_width;
        int input_height;
        int input_channels;
        float* filter;
        uin32* filter_gpu;
        int filter_width;
        int filter_height;
        uin32* output;
        uin32* output_gpu;
        int output_width;
        int output_height;
        int output_channels;
        bool padding;
        float* bn;
        float* bn_gpu;
        int batch;
        int stride_height;
        int stride_width;
        int pad_h;
        int pad_w;
        int pool_width;
        int pool_height;
        InConv128LayerParam* gpu;
        char name[8];
};

class Conv128LayerParam
{
    public:
        Conv128LayerParam(const char* name, int _input_height, int _input_width, 
                int _filter_height, int _filter_width, int _input_channels, 
                int _output_channels, int _batch, int _stride_height=1, 
                int _stride_width=1, bool _padding=true, int _pool_height=1, 
                int _pool_width=1, bool _save_skip_connection=false,
                int _skip_connection_channels=0, bool _concat_skip_connection=false) :
            input_height(_input_height), input_width(_input_width), 
            filter_height(_filter_height), filter_width(_filter_width),
            input_channels(_input_channels), output_channels(_output_channels),
            batch(_batch), stride_height(_stride_height), stride_width(_stride_width),
            pool_height(_pool_height), pool_width(_pool_width), padding(_padding), 
            save_skip_connection(_save_skip_connection), skip_connection_channels(_skip_connection_channels), 
            concat_skip_connection(_concat_skip_connection),
            bn(NULL), bn_gpu(NULL), filter(NULL), filter_gpu(NULL), output(NULL),
            output_gpu(NULL), input(NULL), input_gpu(NULL), gpu(NULL), 
            output_skip_connection_gpu(NULL), input_skip_connection_gpu(NULL),
            runtime_regular_input_gpu(NULL), owns_input_gpu(false)
        {
            strncpy(this->name, name, 8);
            this->pad_h = padding?((( (input_height+stride_height-(input_height%stride_height))
                            /stride_height-1)*stride_height+filter_height-input_height)/2):0;
            this->pad_w = padding?((( (input_width+stride_width-(input_width%stride_width))
                                /stride_width-1)*stride_width+filter_width-input_width)/2):0; 
            int buf_height = padding?(input_height+stride_height-1)/stride_height
                :((input_height-filter_height)/stride_height+1);
            int buf_width = padding?(input_width+stride_width-1)/stride_width
                :((input_width-filter_width)/stride_width+1);
            pre_pool_height = buf_height;
            pre_pool_width = buf_width;
            output_height = (buf_height+pool_height-1)/pool_height;
            output_width  = (buf_width+pool_width-1)/pool_width;
        }

        int input_size() { return input_channels*input_height*input_width*batch;}
        int input_bytes() { return input_size()*sizeof(uin32);}
        int input_bit_size() { return STEP128(input_channels)*input_height * input_width*PAD8(batch); }
        int input_bit_bytes() { return input_bit_size()*sizeof(uin128);}
        int filter_size() { return output_channels*input_channels*filter_height*filter_width;}
        int filter_bytes() { return filter_size()*sizeof(float);}
        int filter_bit_size() { return PAD32(output_channels)*STEP128(input_channels) * filter_height*filter_width; }
        int filter_bit_bytes() { return filter_bit_size()*sizeof(uin128);}
        int output_size() { return output_channels*output_height*output_width*batch;}
        int output_bytes() { return output_size()*sizeof(uin32);}
        int output_bit_size() { return STEP128(output_channels)*output_height * output_width*PAD8(batch); }
        int output_bit_bytes() { return output_bit_size()*sizeof(uin128); }
        int bn_size() { return output_channels;}
        int bn_bytes() { return bn_size()*sizeof(float);}
        // Skip buffer size at pre-pooling resolution (bit-packed)
        int skip_connection_size() { return STEP128(output_channels)*pre_pool_height * pre_pool_width*PAD8(batch); }
        int skip_connection_bytes() { return skip_connection_size()*sizeof(uin128); }

        Conv128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            if (save_skip_connection) CHECK_NULL_POINTER(output_skip_connection_gpu);
            if (concat_skip_connection) {
                CHECK_NULL_POINTER(input_skip_connection_gpu);
                CHECK_NULL_POINTER(runtime_regular_input_gpu);
            }
            SAFE_ALOC_GPU(this->gpu, sizeof(Conv128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(Conv128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }

        void set_input_gpu(uin32* v) { this->input_gpu = v; }
        void set_input_skip_connection_gpu(uin32* v) { this->input_skip_connection_gpu = v; }

        Conv128LayerParam* initialize(FILE* config_file, uin32* prev_layer_gpu,
                uin32* input_skip_connection_gpu = NULL)
        {
            if (concat_skip_connection) {
                // For concatenation: allocate our own input buffer (concatenated), store both sources
                this->runtime_regular_input_gpu = prev_layer_gpu;        // upsampled tensor
                set_input_skip_connection_gpu(input_skip_connection_gpu); // encoder skip tensor
                SAFE_ALOC_GPU(input_gpu, input_bit_bytes());
                CUDA_SAFE_CALL(cudaMemset(input_gpu, 0, input_bit_bytes()));
                owns_input_gpu = true;  // we own and will free this concat buffer
            } else {
                // Non-concat: just point to producer's output (we do NOT own it)
                set_input_gpu(prev_layer_gpu);
                owns_input_gpu = false;
            }

            // Weights
            SAFE_ALOC_HOST(filter, filter_bytes());
            launch_array(config_file, filter, filter_size());
            SAFE_ALOC_GPU(filter_gpu, filter_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(filter_gpu, 0, filter_bit_bytes()) );
            float* filter_float = NULL;
            SAFE_ALOC_GPU(filter_float, filter_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(filter_float, filter, filter_bytes(), cudaMemcpyHostToDevice) );
            PackFiltersByInChannels128FMT<<<dim3(filter_height*filter_width, output_channels), 32>>>(
                filter_float, filter_gpu, input_channels, output_channels, filter_width, filter_height);
            SAFE_FREE_GPU(filter_float);

            // BN
            SAFE_ALOC_HOST(bn, bn_bytes());
            launch_array(config_file, bn, bn_size());
            SAFE_ALOC_GPU(bn_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice) );

            // Output
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            
            // Allocate skip buffer (encoder conv2 layers)
            if (save_skip_connection)
            {
                SAFE_ALOC_GPU(output_skip_connection_gpu, skip_connection_bytes());
                CUDA_SAFE_CALL( cudaMemset(output_skip_connection_gpu, 0, skip_connection_bytes()) );
            }
            
            return this->ready();
        }

        uin32* get_output_gpu() { return this->output_gpu; }
        uin32* get_output_skip_connection_gpu() { return this->output_skip_connection_gpu; }

        unsigned* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }

        float* download_full_output()
        {
            float* full_output = NULL;
            SAFE_ALOC_HOST(full_output, output_bytes());
            float* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, output_bytes());
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, output_bytes()));
            UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch);
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, output_bytes(), cudaMemcpyDeviceToHost) );
            SAFE_FREE_GPU(full_output_gpu);
            return full_output;
        }

        void release()
        {
            SAFE_FREE_HOST(filter);
            SAFE_FREE_HOST(bn);
            SAFE_FREE_HOST(output);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(filter_gpu);
            SAFE_FREE_GPU(bn_gpu);
            SAFE_FREE_GPU(gpu);
            if (save_skip_connection) SAFE_FREE_GPU(output_skip_connection_gpu);
            if (owns_input_gpu && input_gpu) SAFE_FREE_GPU(input_gpu); // free concat buffer we allocated
            // Do not free runtime_regular_input_gpu or input_skip_connection_gpu; not owned here.
        }

        ~Conv128LayerParam() { release(); }

    public:
        // Inputs
        uin32* input;
        uin32* input_gpu;                 // for concat layers: our concatenated buffer
        int input_width;
        int input_height;
        int input_channels;

        // Weights
        float* filter;
        uin32* filter_gpu;
        int filter_width;
        int filter_height;

        // Outputs
        uin32* output;
        uin32* output_gpu;
        int output_width;
        int output_height;
        int output_channels;

        float* bn;
        float* bn_gpu;
        int batch;
        int stride_height;
        int stride_width;
        bool padding;
        int pad_h;
        int pad_w;
        int pool_width;
        int pool_height;

        // Pre-pooling dimensions for skip connections
        int pre_pool_height;
        int pre_pool_width;

        // Skip connection parameters
        bool save_skip_connection;
        uin32* output_skip_connection_gpu;  // encoder-side bit-packed skip buffer
        int  skip_connection_channels;
        bool concat_skip_connection;        // decoder concat flag
        uin32* input_skip_connection_gpu;   // pointer to encoder skip (borrowed)
        uin32* runtime_regular_input_gpu;   // pointer to upsampled input (borrowed)
        bool owns_input_gpu;                // true if we allocated input_gpu (concat buffer)

        Conv128LayerParam* gpu;
        char name[8];
};

// ============================================================================
// Conv128LayerTernaryParam: 2-bit (ternary) weights via two b1 planes
// Activations remain binary; output is bit-packed like binary.
// Supports skip concat by allocating a dedicated input buffer (concatenated).
// ============================================================================
class Conv128LayerTernaryParam
{
public:
    Conv128LayerTernaryParam(const char* name, int _input_height, int _input_width, 
            int _filter_height, int _filter_width, int _input_channels, 
            int _output_channels, int _batch, int _stride_height=1, 
            int _stride_width=1, bool _padding=true, int _pool_height=1, 
            int _pool_width=1, bool _save_skip_connection=false,
            int _skip_connection_channels=0, bool _concat_skip_connection=false) :
        input_height(_input_height), input_width(_input_width), 
        filter_height(_filter_height), filter_width(_filter_width),
        input_channels(_input_channels), output_channels(_output_channels),
        batch(_batch), stride_height(_stride_height), stride_width(_stride_width),
        pool_height(_pool_height), pool_width(_pool_width), padding(_padding), 
        save_skip_connection(_save_skip_connection), skip_connection_channels(_skip_connection_channels), 
        concat_skip_connection(_concat_skip_connection), bn(NULL), bn_gpu(NULL),
        filter_pos_host(NULL), filter_neg_host(NULL),
        filter_pos_gpu(NULL), filter_neg_gpu(NULL),
        output(NULL), output_gpu(NULL), input(NULL), input_gpu(NULL), gpu(NULL), 
        output_skip_connection_gpu(NULL), input_skip_connection_gpu(NULL),
        runtime_regular_input_gpu(NULL), owns_input_gpu(false)
    {
        strncpy(this->name, name, 8);
        this->pad_h = padding?((( (input_height+stride_height-(input_height%stride_height))
                        /stride_height-1)*stride_height+filter_height-input_height)/2):0;
        this->pad_w = padding?((( (input_width+stride_width-(input_width%stride_width))
                        /stride_width-1)*stride_width+filter_width-input_width)/2):0; 
        int buf_height = padding?(input_height+stride_height-1)/stride_height
            :((input_height-filter_height)/stride_height+1);
        int buf_width = padding?(input_width+stride_width-1)/stride_width
            :((input_width-filter_width)/stride_width+1);
        pre_pool_height = buf_height;
        pre_pool_width  = buf_width;
        output_height = (buf_height+pool_height-1)/pool_height;
        output_width  = (buf_width+pool_width-1)/pool_width;
    }

    // Sizes
    int input_size()       { return input_channels*input_height*input_width*batch; }
    int input_bytes()      { return input_size()*sizeof(uin32); }
    int input_bit_size()   { return STEP128(input_channels)*input_height*input_width*PAD8(batch); }
    int input_bit_bytes()  { return input_bit_size()*sizeof(uin128); }

    int filter_size()      { return output_channels*input_channels*filter_height*filter_width; }
    int filter_bytes()     { return filter_size()*sizeof(float); }
    int filter_bit_size()  { return PAD32(output_channels)*STEP128(input_channels) * filter_height*filter_width; }
    int filter_bit_bytes() { return filter_bit_size()*sizeof(uin128); }

    int output_size()      { return output_channels*output_height*output_width*batch; }
    int output_bytes()     { return output_size()*sizeof(uin32); }
    int output_bit_size()  { return STEP128(output_channels)*output_height*output_width*PAD8(batch); }
    int output_bit_bytes() { return output_bit_size()*sizeof(uin128); }

    int bn_size()          { return output_channels; }
    int bn_bytes()         { return bn_size()*sizeof(float); }

    // Skip buffer (pre-pool) size (bit-packed)
    int skip_connection_size()  { return STEP128(output_channels)*pre_pool_height*pre_pool_width*PAD8(batch); }
    int skip_connection_bytes() { return skip_connection_size()*sizeof(uin128); }

    // Pointers setters
    void set_input_gpu(uin32* v)                { this->input_gpu = v; }
    void set_input_skip_connection_gpu(uin32* v){ this->input_skip_connection_gpu = v; }

    // Ready-to-run copy to device
    Conv128LayerTernaryParam* ready()
    {
        CHECK_NULL_POINTER(input_gpu);
        CHECK_NULL_POINTER(output_gpu);
        CHECK_NULL_POINTER(filter_pos_gpu);
        CHECK_NULL_POINTER(filter_neg_gpu);
        if (save_skip_connection) CHECK_NULL_POINTER(output_skip_connection_gpu);
        if (concat_skip_connection) {
            CHECK_NULL_POINTER(input_skip_connection_gpu);
            CHECK_NULL_POINTER(runtime_regular_input_gpu);
        }
        SAFE_ALOC_GPU(this->gpu, sizeof(Conv128LayerTernaryParam));
        CUDA_SAFE_CALL(cudaMemcpy(this->gpu, this, sizeof(Conv128LayerTernaryParam), cudaMemcpyHostToDevice));
        return this->gpu;
    }

    // Initialize: read pos+neg planes from CSV, pack both, allocate buffers
    Conv128LayerTernaryParam* initialize(FILE* config_file, uin32* prev_layer_gpu,
            uin32* input_skip_connection_gpu_ = NULL)
    {
        // Input setup
        if (concat_skip_connection) {
            this->runtime_regular_input_gpu = prev_layer_gpu;
            set_input_skip_connection_gpu(input_skip_connection_gpu_);
            SAFE_ALOC_GPU(input_gpu, input_bit_bytes());
            CUDA_SAFE_CALL(cudaMemset(input_gpu, 0, input_bit_bytes()));
            owns_input_gpu = true;
        } else {
            set_input_gpu(prev_layer_gpu);
            owns_input_gpu = false;
        }

        // 1) Read positive plane
        SAFE_ALOC_HOST(filter_pos_host, filter_bytes());
        launch_array(config_file, filter_pos_host, filter_size());

        // Pack positive plane
        SAFE_ALOC_GPU(filter_pos_gpu, filter_bit_bytes());
        CUDA_SAFE_CALL(cudaMemset(filter_pos_gpu, 0, filter_bit_bytes()));
        {
            float* filter_pos_d = NULL;
            SAFE_ALOC_GPU(filter_pos_d, filter_bytes());
            CUDA_SAFE_CALL(cudaMemcpy(filter_pos_d, filter_pos_host, filter_bytes(), cudaMemcpyHostToDevice));
            PackFiltersByInChannels128FMT<<<dim3(filter_height*filter_width, output_channels), 32>>>(
                filter_pos_d, filter_pos_gpu, input_channels, output_channels, filter_width, filter_height);
            SAFE_FREE_GPU(filter_pos_d);
        }

        // 2) Read negative plane
        SAFE_ALOC_HOST(filter_neg_host, filter_bytes());
        launch_array(config_file, filter_neg_host, filter_size());

        // Pack negative plane
        SAFE_ALOC_GPU(filter_neg_gpu, filter_bit_bytes());
        CUDA_SAFE_CALL(cudaMemset(filter_neg_gpu, 0, filter_bit_bytes()));
        {
            float* filter_neg_d = NULL;
            SAFE_ALOC_GPU(filter_neg_d, filter_bytes());
            CUDA_SAFE_CALL(cudaMemcpy(filter_neg_d, filter_neg_host, filter_bytes(), cudaMemcpyHostToDevice));
            PackFiltersByInChannels128FMT<<<dim3(filter_height*filter_width, output_channels), 32>>>(
                filter_neg_d, filter_neg_gpu, input_channels, output_channels, filter_width, filter_height);
            SAFE_FREE_GPU(filter_neg_d);
        }

        // 3) BN
        SAFE_ALOC_HOST(bn, bn_bytes());
        launch_array(config_file, bn, bn_size());
        SAFE_ALOC_GPU(bn_gpu, bn_bytes());
        CUDA_SAFE_CALL(cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice));

        // Output
        SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
        CUDA_SAFE_CALL(cudaMemset(this->output_gpu, 0, output_bit_bytes()));

        // Encoder-side skip buffer if needed
        if (save_skip_connection) {
            SAFE_ALOC_GPU(output_skip_connection_gpu, skip_connection_bytes());
            CUDA_SAFE_CALL(cudaMemset(output_skip_connection_gpu, 0, skip_connection_bytes()));
        }

        return this->ready();
    }

    // Getters
    uin32* get_output_gpu()                { return this->output_gpu; }
    uin32* get_output_skip_connection_gpu(){ return this->output_skip_connection_gpu; }

    unsigned* download_output()
    {
        if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
        CUDA_SAFE_CALL(cudaMemcpy(output, output_gpu, output_bit_bytes(), cudaMemcpyDeviceToHost));
        return this->output;
    }

    float* download_full_output()
    {
        float* full_output = NULL;
        SAFE_ALOC_HOST(full_output, output_bytes());
        float* full_output_gpu = NULL;
        SAFE_ALOC_GPU(full_output_gpu, output_bytes());
        CUDA_SAFE_CALL(cudaMemset(full_output_gpu, 0, output_bytes()));
        UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(
            output_gpu, full_output_gpu, output_height, output_width, output_channels, batch);
        CUDA_SAFE_CALL(cudaMemcpy(full_output, full_output_gpu, output_bytes(), cudaMemcpyDeviceToHost));
        SAFE_FREE_GPU(full_output_gpu);
        return full_output;
    }

    void release()
    {
        SAFE_FREE_HOST(filter_pos_host);
        SAFE_FREE_HOST(filter_neg_host);
        SAFE_FREE_HOST(bn);
        SAFE_FREE_HOST(output);
        SAFE_FREE_GPU(output_gpu);
        SAFE_FREE_GPU(filter_pos_gpu);
        SAFE_FREE_GPU(filter_neg_gpu);
        SAFE_FREE_GPU(bn_gpu);
        SAFE_FREE_GPU(gpu);
        if (owns_input_gpu && input_gpu) SAFE_FREE_GPU(input_gpu);
    }

    ~Conv128LayerTernaryParam() { release(); }

public:
    // Input
    uin32* input;
    uin32* input_gpu;                 // for concat layers: concatenated buffer (owned)
    int input_width;
    int input_height;
    int input_channels;

    // Weights
    float* filter_pos_host;           // host floats (0 or 1) for pos plane (b1)
    float* filter_neg_host;           // host floats (0 or 1) for neg plane (b2)
    uin32* filter_pos_gpu;            // packed bit-plane: b1
    uin32* filter_neg_gpu;            // packed bit-plane: b2
    int filter_width;
    int filter_height;

    // Output
    uin32* output;
    uin32* output_gpu;
    int output_width;
    int output_height;
    int output_channels;

    float* bn;
    float* bn_gpu;

    int batch;
    int stride_height;
    int stride_width;
    bool padding;
    int pad_h;
    int pad_w;
    int pool_width;
    int pool_height;

    // Pre-pooling dims for skip save
    int pre_pool_height;
    int pre_pool_width;

    // Skip concat support
    bool save_skip_connection;
    uin32* output_skip_connection_gpu;

    int  skip_connection_channels;
    bool concat_skip_connection;
    uin32* input_skip_connection_gpu;    // encoder skip (borrowed)
    uin32* runtime_regular_input_gpu;    // upsampled input (borrowed)
    bool owns_input_gpu;

    Conv128LayerTernaryParam* gpu;
    char name[8];
};


class ConvTranspose128LayerParam
{
    public:
        ConvTranspose128LayerParam(const char* name, int _input_height, int _input_width, 
                int _filter_height, int _filter_width, int _input_channels, 
                int _output_channels, int _batch, int _stride_height=2, 
                int _stride_width=2, int _pad_h=0, int _pad_w=0) :

            input_height(_input_height), input_width(_input_width), 
            filter_height(_filter_height), filter_width(_filter_width),
            input_channels(_input_channels), output_channels(_output_channels),
            batch(_batch), stride_height(_stride_height), stride_width(_stride_width),
            pad_h(_pad_h), pad_w(_pad_w),
            bn(NULL), bn_gpu(NULL), filter(NULL), filter_gpu(NULL), output(NULL),
            output_gpu(NULL), input(NULL), input_gpu(NULL), gpu(NULL)
                
        {
            strncpy(this->name, name, 8);
            output_height = input_height * stride_height;
            output_width = input_width * stride_width;
        }
        int input_size() { return input_channels*input_height*input_width*batch;}
        int input_bytes() { return input_size()*sizeof(uin32);}
        int input_bit_size() { return STEP128(input_channels)*input_height
            *input_width*PAD8(batch); }
        int input_bit_bytes() { return input_bit_size()*sizeof(uin128);}
        int filter_size() { return output_channels*input_channels*filter_height*filter_width;}
        int filter_bytes() { return filter_size()*sizeof(float);}
        int filter_bit_size() { return PAD32(output_channels)*STEP128(input_channels)
            *filter_height*filter_width; }
        int filter_bit_bytes() { return filter_bit_size()*sizeof(uin128);}
        int output_size() { return output_channels*output_height*output_width*batch;}
        int output_bytes() { return output_size()*sizeof(uin32);}
        int output_bit_size() { return STEP128(output_channels)*output_height
            *output_width*PAD8(batch); }
        int output_bit_bytes() { return output_bit_size()*sizeof(uin128); }
        int bn_size() { return output_channels;}
        int bn_bytes() { return bn_size()*sizeof(float);}

        ConvTranspose128LayerParam* ready()
        {
            CHECK_NULL_POINTER(input_gpu);
            CHECK_NULL_POINTER(output_gpu);
            SAFE_ALOC_GPU(this->gpu, sizeof(ConvTranspose128LayerParam));
            CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, 
                        sizeof(ConvTranspose128LayerParam), cudaMemcpyHostToDevice) );
            return this->gpu;
        }
        void set_input_gpu(uin32* input_gpu) { this->input_gpu = input_gpu; }

        ConvTranspose128LayerParam* initialize(FILE* config_file, uin32* prev_layer_gpu)
        {
            //Process weight
            SAFE_ALOC_HOST(filter, filter_bytes());
            launch_array(config_file, filter, filter_size());
            SAFE_ALOC_GPU(filter_gpu, filter_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(filter_gpu, 0, filter_bit_bytes()) );
            float* filter_float = NULL;
            SAFE_ALOC_GPU(filter_float, filter_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(filter_float, filter, 
                        filter_bytes(), cudaMemcpyHostToDevice) );

            PackFiltersByInChannels128FMT<<<dim3(filter_height*filter_width, output_channels),
                32>>>(filter_float, filter_gpu, input_channels, output_channels, 
                    filter_width, filter_height);
            SAFE_FREE_GPU(filter_float);
            //Process bn
            SAFE_ALOC_HOST(bn, bn_bytes());
            launch_array(config_file, bn, bn_size());
            SAFE_ALOC_GPU(bn_gpu, bn_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice) );
            //Allocate output gpu
            SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );
            set_input_gpu(prev_layer_gpu);
            
            return this->ready();
        }
        uin32* get_output_gpu() { return this->output_gpu; }

        unsigned* download_output()
        {
            if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
            CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu, 
                        output_bit_bytes(), cudaMemcpyDeviceToHost) );
            return this->output;
        }
        float* download_full_output()
        {
            float* full_output = NULL;
            SAFE_ALOC_HOST(full_output, output_bytes());
            float* full_output_gpu = NULL;
            SAFE_ALOC_GPU(full_output_gpu, output_bytes());
            CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, output_bytes()));
            UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(output_gpu,
                    full_output_gpu, output_height, output_width, output_channels, batch);
            CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, 
                        output_bytes(), cudaMemcpyDeviceToHost) );
            SAFE_FREE_GPU(full_output_gpu);
            return full_output;
        }
        void release()
        {
            SAFE_FREE_HOST(filter);
            SAFE_FREE_HOST(bn);
            SAFE_FREE_HOST(output);
            SAFE_FREE_GPU(output_gpu);
            SAFE_FREE_GPU(filter_gpu);
            SAFE_FREE_GPU(bn_gpu);
            SAFE_FREE_GPU(gpu);
        }
        ~ConvTranspose128LayerParam() { release(); }

    public:
        //Input
        uin32* input;
        uin32* input_gpu;
        int input_width;
        int input_height;
        int input_channels;
        //Weight
        float* filter;
        uin32* filter_gpu;
        int filter_width;
        int filter_height;
        //Output
        uin32* output;
        uin32* output_gpu;
        int output_width;
        int output_height;
        int output_channels;
        float* bn;
        float* bn_gpu;
        int batch;
        int stride_height;
        int stride_width;
        int pad_h;
        int pad_w;
        ConvTranspose128LayerParam* gpu;
        char name[8];
};

// ============================================================================
// ConvTranspose128LayerTernaryParam: 2-bit (ternary) weights via two b1 planes
// - Activations are binary; output is bit-packed like binary
// ============================================================================
class ConvTranspose128LayerTernaryParam
{
public:
    ConvTranspose128LayerTernaryParam(const char* name, int _input_height, int _input_width, 
            int _filter_height, int _filter_width, int _input_channels, 
            int _output_channels, int _batch, int _stride_height=2, 
            int _stride_width=2, int _pad_h=0, int _pad_w=0) :
        input_height(_input_height), input_width(_input_width), 
        filter_height(_filter_height), filter_width(_filter_width),
        input_channels(_input_channels), output_channels(_output_channels),
        batch(_batch), stride_height(_stride_height), stride_width(_stride_width),
        pad_h(_pad_h), pad_w(_pad_w),
        bn(NULL), bn_gpu(NULL),
        filter_pos_host(NULL), filter_neg_host(NULL),
        filter_pos_gpu(NULL), filter_neg_gpu(NULL),
        output(NULL), output_gpu(NULL), input(NULL), input_gpu(NULL), gpu(NULL)
    {
        strncpy(this->name, name, 8);
        output_height = input_height * stride_height;
        output_width  = input_width  * stride_width;
    }

    // Sizes
    int input_size()       { return input_channels*input_height*input_width*batch; }
    int input_bytes()      { return input_size()*sizeof(uin32); }
    int input_bit_size()   { return STEP128(input_channels)*input_height * input_width*PAD8(batch); }
    int input_bit_bytes()  { return input_bit_size()*sizeof(uin128); }

    int filter_size()      { return output_channels*input_channels*filter_height*filter_width; }
    int filter_bytes()     { return filter_size()*sizeof(float); }
    int filter_bit_size()  { return PAD32(output_channels)*STEP128(input_channels) * filter_height*filter_width; }
    int filter_bit_bytes() { return filter_bit_size()*sizeof(uin128); }

    int output_size()      { return output_channels*output_height*output_width*batch; }
    int output_bytes()     { return output_size()*sizeof(uin32); }
    int output_bit_size()  { return STEP128(output_channels)*output_height * output_width*PAD8(batch); }
    int output_bit_bytes() { return output_bit_size()*sizeof(uin128); }

    int bn_size()          { return output_channels; }
    int bn_bytes()         { return bn_size()*sizeof(float); }

    ConvTranspose128LayerTernaryParam* ready()
    {
        CHECK_NULL_POINTER(input_gpu);
        CHECK_NULL_POINTER(output_gpu);
        CHECK_NULL_POINTER(filter_pos_gpu);
        CHECK_NULL_POINTER(filter_neg_gpu);
        SAFE_ALOC_GPU(this->gpu, sizeof(ConvTranspose128LayerTernaryParam));
        CUDA_SAFE_CALL(cudaMemcpy(this->gpu, this, sizeof(ConvTranspose128LayerTernaryParam), cudaMemcpyHostToDevice));
        return this->gpu;
    }

    void set_input_gpu(uin32* v) { this->input_gpu = v; }

    // CSV is expected to provide: [filter_pos][filter_neg][bn] in this order
    ConvTranspose128LayerTernaryParam* initialize(FILE* config_file, uin32* prev_layer_gpu)
    {
        // 1) Read positive plane
        SAFE_ALOC_HOST(filter_pos_host, filter_bytes());
        launch_array(config_file, filter_pos_host, filter_size());

        // Pack positive plane
        SAFE_ALOC_GPU(filter_pos_gpu, filter_bit_bytes());
        CUDA_SAFE_CALL(cudaMemset(filter_pos_gpu, 0, filter_bit_bytes()));
        {
            float* filter_pos_d = NULL;
            SAFE_ALOC_GPU(filter_pos_d, filter_bytes());
            CUDA_SAFE_CALL(cudaMemcpy(filter_pos_d, filter_pos_host, filter_bytes(), cudaMemcpyHostToDevice));
            PackFiltersByInChannels128FMT<<<dim3(filter_height*filter_width, output_channels), 32>>>(
                filter_pos_d, filter_pos_gpu, input_channels, output_channels, filter_width, filter_height);
            SAFE_FREE_GPU(filter_pos_d);
        }

        // 2) Read negative plane
        SAFE_ALOC_HOST(filter_neg_host, filter_bytes());
        launch_array(config_file, filter_neg_host, filter_size());

        // Pack negative plane
        SAFE_ALOC_GPU(filter_neg_gpu, filter_bit_bytes());
        CUDA_SAFE_CALL(cudaMemset(filter_neg_gpu, 0, filter_bit_bytes()));
        {
            float* filter_neg_d = NULL;
            SAFE_ALOC_GPU(filter_neg_d, filter_bytes());
            CUDA_SAFE_CALL(cudaMemcpy(filter_neg_d, filter_neg_host, filter_bytes(), cudaMemcpyHostToDevice));
            PackFiltersByInChannels128FMT<<<dim3(filter_height*filter_width, output_channels), 32>>>(
                filter_neg_d, filter_neg_gpu, input_channels, output_channels, filter_width, filter_height);
            SAFE_FREE_GPU(filter_neg_d);
        }

        // 3) BN
        SAFE_ALOC_HOST(bn, bn_bytes());
        launch_array(config_file, bn, bn_size());
        SAFE_ALOC_GPU(bn_gpu, bn_bytes());
        CUDA_SAFE_CALL(cudaMemcpy(bn_gpu, bn, bn_bytes(), cudaMemcpyHostToDevice));

        // Output + input
        SAFE_ALOC_GPU(output_gpu, output_bit_bytes());
        CUDA_SAFE_CALL(cudaMemset(this->output_gpu, 0, output_bit_bytes()));
        set_input_gpu(prev_layer_gpu);
        return this->ready();
    }

    uin32* get_output_gpu() { return this->output_gpu; }

    unsigned* download_output()
    {
        if (output == NULL) SAFE_ALOC_HOST(output, output_bit_bytes());
        CUDA_SAFE_CALL(cudaMemcpy(output, output_gpu, output_bit_bytes(), cudaMemcpyDeviceToHost));
        return this->output;
    }

    float* download_full_output()
    {
        float* full_output = NULL;
        SAFE_ALOC_HOST(full_output, output_bytes());
        float* full_output_gpu = NULL;
        SAFE_ALOC_GPU(full_output_gpu, output_bytes());
        CUDA_SAFE_CALL(cudaMemset(full_output_gpu, 0, output_bytes()));
        UnpackConvOutput32FMT<<<dim3(output_width,output_height,batch), 32>>>(
            output_gpu, full_output_gpu, output_height, output_width, output_channels, batch);
        CUDA_SAFE_CALL(cudaMemcpy(full_output, full_output_gpu, output_bytes(), cudaMemcpyDeviceToHost));
        SAFE_FREE_GPU(full_output_gpu);
        return full_output;
    }

    void release()
    {
        SAFE_FREE_HOST(filter_pos_host);
        SAFE_FREE_HOST(filter_neg_host);
        SAFE_FREE_HOST(output);
        SAFE_FREE_HOST(bn);
        SAFE_FREE_GPU(output_gpu);
        SAFE_FREE_GPU(filter_pos_gpu);
        SAFE_FREE_GPU(filter_neg_gpu);
        SAFE_FREE_GPU(bn_gpu);
        SAFE_FREE_GPU(gpu);
    }

    ~ConvTranspose128LayerTernaryParam() { release(); }

public:
    // Input
    uin32* input;
    uin32* input_gpu;
    int input_width;
    int input_height;
    int input_channels;

    // Weights
    float* filter_pos_host;  // host floats for pos plane
    float* filter_neg_host;  // host floats for neg plane
    uin32* filter_pos_gpu;   // packed pos plane
    uin32* filter_neg_gpu;   // packed neg plane
    int filter_width;
    int filter_height;

    // Output
    uin32* output;
    uin32* output_gpu;
    int output_width;
    int output_height;
    int output_channels;

    // BN
    float* bn;
    float* bn_gpu;

    int batch;
    int stride_height;
    int stride_width;
    int pad_h;
    int pad_w;

    ConvTranspose128LayerTernaryParam* gpu;
    char name[8];
};
#endif