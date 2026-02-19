// ---------------------------------------------------------------------------
// File: kernel.cuh
// GPU kernel functions.
// ---------------------------------------------------------------------------
// Chunshu Wu, Pacific Northwest National Laboratory (PNNL), U.S.
// Richland, 99352, WA, USA. Feb-03-2026.
// ---------------------------------------------------------------------------

#ifndef KERNEL_CUH
#define KERNEL_CUH
#include <cuda.h>
#include <mma.h>

#include "param.h"

//======================== Concat helper template ============================
template <typename ParamT>
__device__ __forceinline__ void ConcatInputSkipBitpack128FMT_T(ParamT* p)
{
    if (!p->concat_skip_connection) return;

    const int total_threads = gridDim.x * blockDim.x;
    if (total_threads < 32) return;

    const int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = tid & 31;
    size_t blk_idx    = (size_t)tid >> 5;
    size_t blk_stride = (size_t)total_threads >> 5;

    const int H = p->input_height;
    const int W = p->input_width;
    const int bas = STEP8(p->batch);
    const int insOut = STEP128(p->input_channels);

    const int regularC = p->input_channels - p->skip_connection_channels;
    const int insA = STEP128(regularC);
    const int insB = STEP128(p->skip_connection_channels);

    const size_t total_blocks = (size_t)H * W * bas * insOut;

    uin32*       Out = p->input_gpu;
    const uin32* A   = p->runtime_regular_input_gpu;
    const uin32* B   = p->input_skip_connection_gpu;

    while (blk_idx < total_blocks) {
        size_t s  = blk_idx / (insOut * bas);
        size_t r  = blk_idx % (insOut * bas);
        int bn    = (int)(r / insOut);
        int cblk  = (int)(r % insOut);

        size_t out_base =
            s * (size_t)bas*8 * insOut*4 +
            8 * (size_t)bn * insOut*4 +
            (size_t)cblk * 4*8;

        uin32 val = 0u;
        if (cblk < insA) {
            size_t src_base =
                s * (size_t)bas*8 * insA*4 +
                8 * (size_t)bn * insA*4 +
                (size_t)cblk * 4*8;
            val = A[src_base + lane];
        } else if (cblk < insA + insB) {
            int cb = cblk - insA;
            size_t src_base =
                s * (size_t)bas*8 * insB*4 +
                8 * (size_t)bn * insB*4 +
                (size_t)cb * 4*8;
            val = B[src_base + lane];
        } else {
            val = 0u;
        }

        Out[out_base + lane] = val;
        blk_idx += blk_stride;
    }
}

//================================ Convolution Input ====================================
__device__ __inline__ void InConv128LayerFMT(InConv128LayerParam* p)
{
    GET_LANEID;
    GET_WARPID;
    extern __shared__ int Cs[];
    const int ots = STEP32(p->output_channels); //number of steps in K: output_channels
    const int otm = STEP128(p->output_channels);
    volatile float* Csub = (float*)&Cs[warpid*(p->output_channels)];
    volatile uin32* s_filter = (uin32*)&Cs[32*(p->output_channels)]; 
    const int src_output_height = (p->pool_height)*(p->output_height);
    const int src_output_width = (p->pool_width)*(p->output_width);

    for (int i=threadIdx.x; i<(p->filter_height)*(p->filter_width)
            *(p->input_channels)*ots; i+=32*32) 
        s_filter[i] = p->filter_gpu[i];
    __syncthreads();

    for (int bid = blockIdx.x*32+warpid; bid < src_output_height*src_output_width*(p->batch);
            bid += gridDim.x*32)
    {
        const int bz = bid/(src_output_width*src_output_height); //over N:batch
        const int by = (bid%(src_output_width*src_output_height))
            /(src_output_width);//over P:out_height
        const int bx = (bid%(src_output_width*src_output_height))
            %(src_output_width);//over Q:out_width 
        //coord (ax,ay) in Input from bx,by in Output
        const int ax0 = bx*(p->stride_width)-(p->pad_w);
        const int ay0 = by*(p->stride_height)-(p->pad_h);
        for (int i=laneid; i<(p->output_channels); i+=32) Csub[i] = 0;

        //load a window of data from Input
        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r; //y-coord in Input
            if ((ay>=0) && (ay<(p->input_height))) 
            {
                for (int s=0; s<(p->filter_width); s++)
                {
                    const int ax = ax0 + s; //x-coord in Input
                    //within Input frame
                    if ((ax>=0) && (ax<(p->input_width)) )
                    {
                        float f0 = p->input_gpu[(bz*3+0)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//R
                        float f1 = p->input_gpu[(bz*3+1)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//G
                        float f2 = p->input_gpu[(bz*3+2)*(p->input_height)*(p->input_width)
                            +ay*(p->input_width)+ax];//B

                        for (int k=0; k<ots; k++)
                        {
                            uin32 l0 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 0*ots+k];
                            uin32 l1 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 1*ots+k];
                            uin32 l2 = s_filter[(r*(p->filter_width)+s)
                                *(p->input_channels)*ots + 2*ots+k];

                            Csub[32*k+laneid] += (((l0>>(31-laneid))&0x1)?f0:-f0)
                                + (((l1>>(31-laneid))&0x1)?f1:-f1)
                                + (((l2>>(31-laneid))&0x1)?f2:-f2);
                        }
                    }
                }
            }
        }

        // To shape[batch, input_height, input_width, in_channels/32]
        const int dst_y = by/(p->pool_height);
        const int dst_x = bx/(p->pool_width);

        // To shape[input_height, input_width, batch/8*in_channels/128, batch8*in_channels128/32]
        const int idx = dst_y*(p->output_width)*PAD8(p->batch)*otm*4 //P
                +dst_x*PAD8(p->batch)*otm*4; //Q
        for (int k=0; k<ots; k++)
        {
            // save shape[batch, output_height, output_width, out_channels/64]
            bool bin = (float)(Csub[k*32+laneid])<(p->bn_gpu)[k*32+laneid]?0:1;
            unsigned C = __brev(__ballot_sync(0xFFFFFFFF, bin));
            if (laneid==0) atomicOr(&p->output_gpu[idx
                    +((bz/8)*otm+k/4)*32+((bz%8)*4+k%4)], C); //Q
        }
    }
}

// Device concat: packs regular and skip into p->input_gpu (bit-packed, channel-concat)
__device__ __forceinline__ void ConcatInputSkipBitpack128FMT(Conv128LayerParam* p) {
    ConcatInputSkipBitpack128FMT_T(p);
}

// Convolution: always reads a single input buffer p->input_gpu (which is concatenated when needed)
__device__ __inline__ void Conv128LayerFMT(Conv128LayerParam* p)
{
    // existing BMMA implementation (unchanged)
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;
    GET_LANEID;
    GET_WARPID;

    const int ins = STEP128(p->input_channels);        // total K-blocks (128-ch blocks)
    const int ots = STEP32(p->output_channels);        // output channel blocks (per 32 out ch)
    const int bas = STEP8(p->batch);                   // batch groups

    // Pre-pooling spatial size for this layer (source size before any pooling)
    const int src_output_height = (p->pool_height) * (p->output_height);
    const int src_output_width  = (p->pool_width)  * (p->output_width);

    extern __shared__ int Cs[];

    for (int bid = blockIdx.x*32 + warpid;
         bid < src_output_height*src_output_width*ots*bas;
         bid += gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b0_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b1_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b2_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b3_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c1_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c2_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c3_frag;

        const int by = bid/(src_output_width*ots*bas);
        const int bx = (bid%(src_output_width*ots*bas))/(ots*bas);
        const int bz = (bid%(src_output_width*ots*bas))%(ots*bas);
        const int bn = bz / ots;
        const int bo = bz % ots;

        const int ax0 = bx*(p->stride_width)  - (p->pad_w);
        const int ay0 = by*(p->stride_height) - (p->pad_h);

        int exclude = 0;
        wmma::fill_fragment(c0_frag, 0);
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c2_frag, 0);
        wmma::fill_fragment(c3_frag, 0);

        for (int r=0; r<(p->filter_height); r++)
        {
            const int ay = ay0 + r;
            for (int s=0; s<(p->filter_width); s++)
            {
                const int ax = ax0 + s;
                if ((ay>=0)&&(ay<(p->input_height))&&(ax>=0)&&(ax<(p->input_width)))
                {
                    for (int c=0; c<ins; c++)
                    {
                        // Always load from single buffer (concat already materialized if needed)
                        load_matrix_sync(
                            a_frag,
                            &(p->input_gpu[
                                (ay*(p->input_width)+ax)*bas*8*ins*4
                              + 8*bn*ins*4 + c*4*8]),
                            128);

                        load_matrix_sync(b0_frag,
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+0)*ins*4+c*4*8]), 128);
                        bmma_sync(c0_frag, a_frag, b0_frag, c0_frag);

                        load_matrix_sync(b1_frag,
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+8)*ins*4+c*4*8]), 128);
                        bmma_sync(c1_frag, a_frag, b1_frag, c1_frag);

                        load_matrix_sync(b2_frag,
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+16)*ins*4+c*4*8]), 128);
                        bmma_sync(c2_frag, a_frag, b2_frag, c2_frag);

                        load_matrix_sync(b3_frag,
                            &(p->filter_gpu[(r*(p->filter_width)+s)*ots*32*ins*4
                            +(bo*32+24)*ins*4+c*4*8]), 128);
                        bmma_sync(c3_frag, a_frag, b3_frag, c3_frag);
                    }
                }
                else
                {
                    exclude++;
                }
            }
        }

        store_matrix_sync(&Cs[warpid*256+0],  c0_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+8],  c1_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+16], c2_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+24], c3_frag, 32, wmma::mem_row_major);
        __syncthreads();

        for (int b=0; b<8; b++)
        {
            int res = (int)(p->input_channels)*(p->filter_width)*(p->filter_height)
                    - (int)exclude*(p->input_channels)
                    - (int)(2*Cs[warpid*256+b*32+laneid]);

            unsigned C = __brev(__ballot_sync(0xFFFFFFFF,
                        (float)res<(p->bn_gpu[bo*32+laneid])?0:1));

            if (laneid==0)
                atomicOr(&p->output_gpu[
                        ((by/(p->pool_height))*(p->output_width) * bas*8*STEP128(p->output_channels)*4)
                    + ((bx/(p->pool_width)) * bas*8*STEP128(p->output_channels)*4)
                    + (bn*STEP128(p->output_channels)+(bo/4))*32 + b*4 + (bo%4)], C);

            // Save skip connection at pre-pool resolution in bit-packed form
            if (p->save_skip_connection)
            {
                if (laneid == 0)
                {
                    atomicOr(&p->output_skip_connection_gpu[
                          (by) * (p->pre_pool_width) * bas*8*STEP128(p->output_channels)*4
                        + (bx) * bas*8*STEP128(p->output_channels)*4
                        + (bn*STEP128(p->output_channels)+(bo/4))*32 + b*4 + (bo%4)], C);
                }
            }
        }
    }
}


// ============================================================================
// Ternary concat for Conv128LayerTernaryParam: packs [regular | skip] into p->input_gpu
// ============================================================================
__device__ __forceinline__ void ConcatInputSkipBitpack128FMT(Conv128LayerTernaryParam* p) {
    ConcatInputSkipBitpack128FMT_T(p);
}

// ============================================================================
// Ternary convolution (Conv128LayerTernaryFMT): res = popc(a^b_neg) - popc(a^b_pos)
// Note: requires 512 ints per warp in shared mem (64KB per block with 32 warps)
// ============================================================================
__device__ __inline__ void Conv128LayerTernaryFMT(struct Conv128LayerTernaryParam* p)
{
    // existing BMMA ternary implementation (unchanged)
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;
    GET_LANEID;
    GET_WARPID;

    const int ins = STEP128(p->input_channels);
    const int ots = STEP32(p->output_channels);
    const int bas = STEP8(p->batch);

    const int src_output_height = (p->pool_height) * (p->output_height);
    const int src_output_width  = (p->pool_width)  * (p->output_width);

    extern __shared__ int Cs[];
    int* Cwarp = Cs + warpid * 512;   // pos: 0..255, neg: 256..511

    for (int bid = blockIdx.x*32 + warpid;
         bid < src_output_height*src_output_width*ots*bas;
         bid += gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;

        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b0p, b1p, b2p, b3p;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b0n, b1n, b2n, b3n;

        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0p, c1p, c2p, c3p;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0n, c1n, c2n, c3n;

        wmma::fill_fragment(c0p, 0); wmma::fill_fragment(c1p, 0);
        wmma::fill_fragment(c2p, 0); wmma::fill_fragment(c3p, 0);
        wmma::fill_fragment(c0n, 0); wmma::fill_fragment(c1n, 0);
        wmma::fill_fragment(c2n, 0); wmma::fill_fragment(c3n, 0);

        const int by = bid/(src_output_width*ots*bas);
        const int bx = (bid%(src_output_width*ots*bas))/(ots*bas);
        const int bz = (bid%(src_output_width*ots*bas))%(ots*bas);
        const int bn = bz / ots;
        const int bo = bz % ots;

        const int ax0 = bx*(p->stride_width)  - (p->pad_w);
        const int ay0 = by*(p->stride_height) - (p->pad_h);

        for (int r=0; r<p->filter_height; r++) {
            const int ay = ay0 + r;
            for (int s=0; s<p->filter_width; s++) {
                const int ax = ax0 + s;
                if ((ay>=0)&&(ay<p->input_height)&&(ax>=0)&&(ax<p->input_width)) {
                    for (int c=0; c<ins; c++) {
                        load_matrix_sync(
                            a_frag,
                            &(p->input_gpu[
                                (ay*(p->input_width)+ax)*bas*8*ins*4
                              + 8*bn*ins*4 + c*4*8]),
                            128);

                        // pos plane (b1): accumulates popc(a^b_pos)
                        load_matrix_sync(b0p, &(p->filter_pos_gpu[(r*(p->filter_width)+s)*ots*32*ins*4 +(bo*32+ 0)*ins*4 + c*4*8]), 128);
                        bmma_sync(c0p, a_frag, b0p, c0p);
                        load_matrix_sync(b1p, &(p->filter_pos_gpu[(r*(p->filter_width)+s)*ots*32*ins*4 +(bo*32+ 8)*ins*4 + c*4*8]), 128);
                        bmma_sync(c1p, a_frag, b1p, c1p);
                        load_matrix_sync(b2p, &(p->filter_pos_gpu[(r*(p->filter_width)+s)*ots*32*ins*4 +(bo*32+16)*ins*4 + c*4*8]), 128);
                        bmma_sync(c2p, a_frag, b2p, c2p);
                        load_matrix_sync(b3p, &(p->filter_pos_gpu[(r*(p->filter_width)+s)*ots*32*ins*4 +(bo*32+24)*ins*4 + c*4*8]), 128);
                        bmma_sync(c3p, a_frag, b3p, c3p);

                        // neg plane (b2): accumulates popc(a^b_neg)
                        load_matrix_sync(b0n, &(p->filter_neg_gpu[(r*(p->filter_width)+s)*ots*32*ins*4 +(bo*32+ 0)*ins*4 + c*4*8]), 128);
                        bmma_sync(c0n, a_frag, b0n, c0n);
                        load_matrix_sync(b1n, &(p->filter_neg_gpu[(r*(p->filter_width)+s)*ots*32*ins*4 +(bo*32+ 8)*ins*4 + c*4*8]), 128);
                        bmma_sync(c1n, a_frag, b1n, c1n);
                        load_matrix_sync(b2n, &(p->filter_neg_gpu[(r*(p->filter_width)+s)*ots*32*ins*4 +(bo*32+16)*ins*4 + c*4*8]), 128);
                        bmma_sync(c2n, a_frag, b2n, c2n);
                        load_matrix_sync(b3n, &(p->filter_neg_gpu[(r*(p->filter_width)+s)*ots*32*ins*4 +(bo*32+24)*ins*4 + c*4*8]), 128);
                        bmma_sync(c3n, a_frag, b3n, c3n);
                    }
                }
            }
        }

        // Write pos and neg accumulators to shared memory
        wmma::store_matrix_sync(&Cwarp[  0], c0p, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[  8], c1p, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[ 16], c2p, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[ 24], c3p, 32, wmma::mem_row_major);

        wmma::store_matrix_sync(&Cwarp[256+  0], c0n, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[256+  8], c1n, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[256+ 16], c2n, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[256+ 24], c3n, 32, wmma::mem_row_major);

        for (int b=0; b<8; b++)
        {
            int pos = Cwarp[b*32 + laneid];
            int neg = Cwarp[256 + b*32 + laneid];
            int res = neg - pos;

            unsigned Cbit = __brev(__ballot_sync(0xFFFFFFFF,
                              (float)res < (p->bn_gpu[bo*32+laneid]) ? 0 : 1));

            if (laneid==0) {
                atomicOr(&p->output_gpu[
                        ((by/(p->pool_height))*(p->output_width) * bas*8*STEP128(p->output_channels)*4)
                    + ((bx/(p->pool_width)) * bas*8*STEP128(p->output_channels)*4)
                    + (bn*STEP128(p->output_channels)+(bo/4))*32 + b*4 + (bo%4)], Cbit);
            }

            if (p->save_skip_connection) {
                if (laneid == 0) {
                    atomicOr(&p->output_skip_connection_gpu[
                          (by) * (p->pre_pool_width) * bas*8*STEP128(p->output_channels)*4
                        + (bx) * bas*8*STEP128(p->output_channels)*4
                        + (bn*STEP128(p->output_channels)+(bo/4))*32 + b*4 + (bo%4)], Cbit);
                }
            }
        }
    }
}




//================================ Transposed Convolution ====================================
__device__ __inline__ void ConvTranspose128LayerFMT(ConvTranspose128LayerParam* p)
{
    // existing BMMA transposed conv implementation (unchanged)
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;
    GET_LANEID;
    GET_WARPID;
    const int ins = STEP128(p->input_channels);
    const int ots = STEP32(p->output_channels);
    const int bas = STEP8(p->batch);
    const int src_output_height = (p->output_height);
    const int src_output_width = (p->output_width);
    extern __shared__ int Cs[];

    for (int bid = blockIdx.x*32+warpid; bid < src_output_height*src_output_width*ots*bas;
            bid += gridDim.x*32)
    {
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b0_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b1_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b2_frag;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b3_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c1_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c2_frag;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c3_frag;

        const int by = bid/(src_output_width*ots*bas); //P: output_height
        const int bx = (bid%(src_output_width*ots*bas))/(ots*bas); //Q:output_width
        const int bz = (bid%(src_output_width*ots*bas))%(ots*bas); //output_channel/32*batch/8
        const int bn = bz / ots; //N:batch (8)
        const int bo = bz % ots; //O:out_channel (4*8)

        int exclude = 0;
        wmma::fill_fragment(c0_frag, 0);
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c2_frag, 0);
        wmma::fill_fragment(c3_frag, 0);

        for (int r=0; r<(p->filter_height); r++)
        {
            for (int s=0; s<(p->filter_width); s++)
            {
                int num_y = by - r + p->pad_h;
                int num_x = bx - s + p->pad_w;
                
                bool valid_y = (num_y >= 0) && (num_y % (p->stride_height) == 0);
                bool valid_x = (num_x >= 0) && (num_x % (p->stride_width) == 0);
                
                if (valid_y && valid_x)
                {
                    const int ay = num_y / (p->stride_height);
                    const int ax = num_x / (p->stride_width);
                    
                    if ((ay>=0)&&(ay<(p->input_height))&&(ax>=0)&&(ax<(p->input_width)))
                    {
                        int fr = (p->filter_height) - 1 - r;
                        int fs = (p->filter_width) - 1 - s;
                        
                        for (int c=0; c<ins; c++)
                        {
                            load_matrix_sync(a_frag, 
                                &(p->input_gpu[(ay*(p->input_width)+ax)*bas*8*ins*4
                                +8*bn*ins*4+c*4*8]), 128);
                            load_matrix_sync(b0_frag, 
                                &(p->filter_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4
                                +(bo*32+0)*ins*4+c*4*8]), 128);
                            bmma_sync(c0_frag, a_frag, b0_frag, c0_frag);
                            load_matrix_sync(b1_frag, 
                                &(p->filter_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4
                                +(bo*32+8)*ins*4+c*4*8]), 128);
                            bmma_sync(c1_frag, a_frag, b1_frag, c1_frag);
                            load_matrix_sync(b2_frag, 
                                &(p->filter_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4
                                +(bo*32+16)*ins*4+c*4*8]), 128);
                            bmma_sync(c2_frag, a_frag, b2_frag, c2_frag);
                            load_matrix_sync(b3_frag, 
                                &(p->filter_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4
                                +(bo*32+24)*ins*4+c*4*8]), 128);
                            bmma_sync(c3_frag, a_frag, b3_frag, c3_frag);
                        }
                    }
                    else
                    {
                        // Input coordinates out of bounds
                        exclude++;
                    }
                }
                else
                {
                    // Stride alignment failed
                    exclude++;
                }
            }
        }
        store_matrix_sync(&Cs[warpid*256+0], c0_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+8], c1_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+16], c2_frag, 32, wmma::mem_row_major);
        store_matrix_sync(&Cs[warpid*256+24], c3_frag, 32, wmma::mem_row_major);
        __syncthreads();

        for (int b=0; b<8; b++)
        {
            int res = (int)(p->input_channels)*(p->filter_width)*(p->filter_height) //C*R*S
                - (int)exclude*(p->input_channels) //eliminate invalid filter positions
                - (int)(2*Cs[warpid*256+b*32+laneid]);//n-2acc(a^b) for 0/1 to sim +1/-1

            unsigned C = __brev(__ballot_sync(0xFFFFFFFF,
                        (float)res<(p->bn_gpu[bo*32+laneid])?0:1));

            if (laneid==0)
                atomicOr(&p->output_gpu[(by*(p->output_width)
                            *bas*8*STEP128(p->output_channels)*4)
                        + (bx*bas*8*STEP128(p->output_channels)*4)
                        + (bn*STEP128(p->output_channels)+(bo/4))*32 + b*4+(bo%4)], C);
        }
    }
}

// ============================================================================
// Ternary Transposed Convolution (2-bit weights via two b1 planes)
//   res = popc(a XOR b_neg) - popc(a XOR b_pos)
// - Activations remain binary
// - Thresholding + bit-packing identical to binary
// - Requires 512 ints per warp in shared mem (64KB per block with 32 warps)
// ============================================================================
__device__ __inline__ void ConvTranspose128LayerTernaryFMT(struct ConvTranspose128LayerTernaryParam* p)
{
    // existing BMMA transposed ternary implementation (unchanged)
    using namespace nvcuda;
    using namespace nvcuda::wmma::experimental;
    GET_LANEID;
    GET_WARPID;

    const int ins = STEP128(p->input_channels);
    const int ots = STEP32(p->output_channels);
    const int bas = STEP8(p->batch);

    // Transposed conv produces directly at output HxW
    const int src_output_height = p->output_height;
    const int src_output_width  = p->output_width;

    extern __shared__ int Cs[];            // shared memory
    int* Cwarp = Cs + warpid * 512;        // 512 ints per warp: pos 0..255, neg 256..511

    for (int bid = blockIdx.x*32 + warpid;
         bid < src_output_height*src_output_width*ots*bas;
         bid += gridDim.x*32)
    {
        // A (activations), Bpos/Bneg (weights), Cpos/Cneg accumulators
        wmma::fragment<wmma::matrix_a, 8, 8, 128, precision::b1, wmma::row_major> a_frag;

        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b0p, b1p, b2p, b3p;
        wmma::fragment<wmma::matrix_b, 8, 8, 128, precision::b1, wmma::col_major> b0n, b1n, b2n, b3n;

        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0p, c1p, c2p, c3p;
        wmma::fragment<wmma::accumulator, 8, 8, 128, int> c0n, c1n, c2n, c3n;

        wmma::fill_fragment(c0p, 0); wmma::fill_fragment(c1p, 0);
        wmma::fill_fragment(c2p, 0); wmma::fill_fragment(c3p, 0);
        wmma::fill_fragment(c0n, 0); wmma::fill_fragment(c1n, 0);
        wmma::fill_fragment(c2n, 0); wmma::fill_fragment(c3n, 0);

        const int by = bid/(src_output_width*ots*bas);                  // output y
        const int bx = (bid%(src_output_width*ots*bas))/(ots*bas);      // output x
        const int bz = (bid%(src_output_width*ots*bas))%(ots*bas);
        const int bn = bz / ots;                                        // batch group
        const int bo = bz % ots;                                        // out channel block

        // Accumulate over filter window and input K-blocks
        for (int r=0; r<p->filter_height; r++)
        {
            for (int s=0; s<p->filter_width; s++)
            {
                // Map output (by,bx) back to input pixel if aligned by stride
                int num_y = by - r + p->pad_h;
                int num_x = bx - s + p->pad_w;
                bool valid_y = (num_y >= 0) && (num_y % (p->stride_height) == 0);
                bool valid_x = (num_x >= 0) && (num_x % (p->stride_width)  == 0);
                if (!(valid_y && valid_x)) continue;

                const int ay = num_y / (p->stride_height);
                const int ax = num_x / (p->stride_width);
                if ((ay<0)||(ay>=p->input_height)||(ax<0)||(ax>=p->input_width)) continue;

                // Flip filter indices for transposed convection
                const int fr = (p->filter_height) - 1 - r;
                const int fs = (p->filter_width)  - 1 - s;

                for (int c=0; c<ins; c++)
                {
                    // Load activations
                    wmma::load_matrix_sync(a_frag, 
                        &(p->input_gpu[(ay*(p->input_width)+ax)*bas*8*ins*4
                                     + 8*bn*ins*4 + c*4*8]), 128);

                    // Positive plane (b1): accumulates popc(a^b_pos)
                    wmma::load_matrix_sync(b0p, &(p->filter_pos_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4 +(bo*32+ 0)*ins*4 + c*4*8]), 128);
                    wmma::bmma_sync(c0p, a_frag, b0p, c0p);
                    wmma::load_matrix_sync(b1p, &(p->filter_pos_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4 +(bo*32+ 8)*ins*4 + c*4*8]), 128);
                    wmma::bmma_sync(c1p, a_frag, b1p, c1p);
                    wmma::load_matrix_sync(b2p, &(p->filter_pos_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4 +(bo*32+16)*ins*4 + c*4*8]), 128);
                    wmma::bmma_sync(c2p, a_frag, b2p, c2p);
                    wmma::load_matrix_sync(b3p, &(p->filter_pos_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4 +(bo*32+24)*ins*4 + c*4*8]), 128);
                    wmma::bmma_sync(c3p, a_frag, b3p, c3p);

                    // Negative plane (b2): accumulates popc(a^b_neg)
                    wmma::load_matrix_sync(b0n, &(p->filter_neg_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4 +(bo*32+ 0)*ins*4 + c*4*8]), 128);
                    wmma::bmma_sync(c0n, a_frag, b0n, c0n);
                    wmma::load_matrix_sync(b1n, &(p->filter_neg_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4 +(bo*32+ 8)*ins*4 + c*4*8]), 128);
                    wmma::bmma_sync(c1n, a_frag, b1n, c1n);
                    wmma::load_matrix_sync(b2n, &(p->filter_neg_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4 +(bo*32+16)*ins*4 + c*4*8]), 128);
                    wmma::bmma_sync(c2n, a_frag, b2n, c2n);
                    wmma::load_matrix_sync(b3n, &(p->filter_neg_gpu[(fr*(p->filter_width)+fs)*ots*32*ins*4 +(bo*32+24)*ins*4 + c*4*8]), 128);
                    wmma::bmma_sync(c3n, a_frag, b3n, c3n);
                }
            }
        }

        // Store both accumulators to shared memory (pos then neg), 256 ints each
        wmma::store_matrix_sync(&Cwarp[  0], c0p, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[  8], c1p, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[ 16], c2p, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[ 24], c3p, 32, wmma::mem_row_major);

        wmma::store_matrix_sync(&Cwarp[256+  0], c0n, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[256+  8], c1n, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[256+ 16], c2n, 32, wmma::mem_row_major);
        wmma::store_matrix_sync(&Cwarp[256+ 24], c3n, 32, wmma::mem_row_major);

        __syncthreads();

        // Pack output bits from (neg - pos) vs BN
        for (int b=0; b<8; b++)
        {
            int pos = Cwarp[b*32 + laneid];          // from pos plane (b1)
            int neg = Cwarp[256 + b*32 + laneid];    // from neg plane (b2)
            int res = neg - pos;

            unsigned Cbit = __brev(__ballot_sync(0xFFFFFFFF,
                              (float)res < (p->bn_gpu[bo*32+laneid]) ? 0 : 1));

            if (laneid==0)
                atomicOr(&p->output_gpu[(by*(p->output_width)
                            *bas*8*STEP128(p->output_channels)*4)
                        + (bx*bas*8*STEP128(p->output_channels)*4)
                        + (bn*STEP128(p->output_channels)+(bo/4))*32 + b*4+(bo%4)], Cbit);
        }
    }
}
#endif
