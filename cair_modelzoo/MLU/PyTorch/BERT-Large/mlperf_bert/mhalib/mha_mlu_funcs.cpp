#include <vector>
#include <iostream>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <math.h>
#include "cnnl.h"
#include "cnrt.h"
#include "torch_mlu.h"

#define nstreams 32

namespace F = torch::nn::functional;
namespace ops = torch_mlu::cnnl::ops;

cnrtQueue_t stream[nstreams];
cnnlHandle_t handle;

torch_mlu::Notifier start;
torch_mlu::Notifier end;

//#define PERF
typedef uint16_t half;

//#define BMM

///////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm1Fprop_(torch::Tensor &A,
                         torch::Tensor &B,
                         torch::Tensor &C,
	    	         int batch,
  	  	         torch::Tensor &seq_len,
                         int heads,
		         int embed,
			 bool scale,
			 bool strided,
			 bool enable_stream,
			 bool sync)
{
#ifdef PERF
    start.place();
#endif

#if 0
    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));
    int batch_count = 0;
    int tokens = 0;
    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
        
        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_view = ops::cnnl_view(a_slice, {seq, heads, 3, embed});
        auto a_input = ops::cnnl_squeeze(ops::cnnl_slice(a_view, 2, 1, 2, 1));
        auto b_input = ops::cnnl_squeeze(ops::cnnl_slice(a_view, 2, 0, 1, 1));
        
        auto a_trans = ops::cnnl_permute(a_input, {1, 0, 2});
        auto b_trans = ops::cnnl_permute(b_input, {1, 0, 2});
        a_trans = ops::cnnl_contiguous(a_trans);
        b_trans = ops::cnnl_contiguous(b_trans);

        auto c_slice = ops::cnnl_slice(C, 0, tokens, tokens + seq * seq * 16, 1);
        auto c_view = ops::cnnl_reshape(c_slice, {16, seq, seq});
        ops::cnnl_bmm_internal(c_view, b_trans, ops::cnnl_mul(a_trans, alpha), false, true);

        batch_count += seq;
        tokens += (seq * seq * 16);
    }
#else

    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));
    int batch_count = 0;
    int tokens = 0;
    
    void *ptrC = static_cast<void*>(static_cast<half*>(C.data_ptr()));  

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();

        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_view = ops::cnnl_view(a_slice, {seq, heads, 3, embed});
        auto a_trans = ops::cnnl_permute(a_view, {1, 2, 0, 3}).contiguous();
    
        //auto b_slice = ops::cnnl_slice(B, 0, batch_count, batch_count + seq, 1);
        //auto b_view = ops::cnnl_view(b_slice, {seq, heads, 3, embed});
        //auto b_trans = ops::cnnl_permute(b_view, {1, 2, 3, 0}).contiguous();
        
        auto c_slice = ops::cnnl_slice(C, 0, tokens, tokens + seq * seq * heads, 1);
        auto c_view = ops::cnnl_reshape(c_slice, {16, seq, seq});
    
        void *ptrA = static_cast<void*>(static_cast<half*>(a_trans.data_ptr()) + embed * seq); 
        void *ptrB = static_cast<void*>(static_cast<half*>(a_trans.data_ptr())); 
        void *ptrC = static_cast<void*>(static_cast<half*>(c_view.data_ptr()));  

        ops::cnnl_strided_bmm_internal(false, true, seq, seq, embed, alpha,
                ptrB, 3 * embed * seq,
                ptrA, 3 * embed * seq,
                ptrC, seq * seq,
                A.scalar_type(),
                heads);

        batch_count += seq;
        tokens += (seq * seq * heads);
    }

#endif

#ifdef PERF
    end.place();
    end.synchronize();
    float time = start.hardware_time(end);
    std::cout << __FUNCTION__ << " hardware time: " << time << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm2Fprop_(torch::Tensor &A,
                    torch::Tensor &B,
                    torch::Tensor &C,
                    int batch,
                    torch::Tensor &seq_len,
                    int heads,
                    int embed,
		    bool scale,
		    bool strided,
		    bool enable_stream,
		    bool sync)
{
   
#ifdef PERF
    start.place();
#endif

#if 0
    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));
    int batch_count = 0;
    int tokens = 0;
    int count = 0;

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
        
        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_view = ops::cnnl_view(a_slice, {seq, heads, 3, embed});
        auto a_input = ops::cnnl_squeeze(ops::cnnl_slice(a_view, 2, 2, 3, 1));
        auto a_trans = ops::cnnl_permute(a_input, {1, 2, 0});

        auto b_slice = ops::cnnl_slice(B, 0, count, count + heads * seq * seq, 1);
        auto b_view = ops::cnnl_view(b_slice, {heads, seq, seq});

        auto c_slice = ops::cnnl_slice(C, 0, batch_count, batch_count + seq, 1);
        auto c_view = ops::cnnl_view(c_slice, {seq, heads, embed});
        //auto c_trans = ops::cnnl_permute(c_view, {1, 2, 0});

        auto result = ops::cnnl_bmm(a_trans, b_view, false, true);
        // c_trans.copy_(result);
        auto s = ops::cnnl_permute_internal(result, {2, 0, 1});
        c_view.copy_(s);

        batch_count += seq;
        tokens += (seq * embed * heads);
        count += (heads * seq * seq);
    }

#else
    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));
    int batch_count = 0;
    int tokens = 0;
    int count = 0;

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
        
        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_view = ops::cnnl_view(a_slice, {seq, heads, 3, embed});
        auto a_trans = ops::cnnl_permute(a_view, {1, 2, 3, 0}).contiguous();

        auto b_slice = ops::cnnl_slice(B, 0, count, count + heads * seq * seq, 1);
        auto b_view = ops::cnnl_view(b_slice, {heads, seq, seq});

        auto c_slice = ops::cnnl_slice(C, 0, batch_count, batch_count + seq, 1);
        auto c_view = ops::cnnl_view(c_slice, {seq, heads, embed});
        auto c_trans = ops::cnnl_permute(c_view, {1, 2, 0});
        auto c_temp = at::empty(c_trans.sizes(), c_view.options());

        void *ptrA = static_cast<void*>(static_cast<half*>(a_trans.data_ptr()) + 2 * embed * seq);
        void *ptrB = static_cast<void*>(static_cast<half*>(b_view.data_ptr())); 
        void *ptrC = static_cast<void*>(static_cast<half*>(c_temp.data_ptr()));  

        ops::cnnl_strided_bmm_internal(false, true, embed, seq, seq, one,
                ptrA, 3 * embed * seq,
                ptrB, seq * seq,
                ptrC, embed * seq,
                A.scalar_type(),
                heads);
        ops::cnnl_permute_out_internal(c_view, c_temp, {2, 0, 1});

        batch_count += seq;
        tokens += (seq * embed * heads);
        count += (heads * seq * seq);
    }
#endif

#ifdef PERF
    end.place();
    end.synchronize();
    float time = start.hardware_time(end);
    std::cout << __FUNCTION__ << " hardware time: " << time << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm1Dgrad1_(torch::Tensor &A,
                         torch::Tensor &B,
                         torch::Tensor &C,
                         int batch,
                         torch::Tensor &seq_len,
                         int heads,
                         int embed,
			 bool scale,
			 bool strided,
			 bool enable_stream,
			 bool sync)
{
#ifdef PERF
    start.place();
#endif

#if 0
    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));
    int batch_count = 0;
    int tokens = 0;
    int count = 0;

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
        
        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_view = ops::cnnl_view(a_slice, {seq, heads, 3, embed});
        auto a_input = ops::cnnl_squeeze(ops::cnnl_slice(a_view, 2, 0, 1, 1));
        auto a_trans = ops::cnnl_permute(a_input, {1, 2, 0});

        auto b_slice = ops::cnnl_slice(B, 0, count, count + heads * seq * seq, 1);
        auto b_view = ops::cnnl_view(b_slice, {heads, seq, seq});

        auto c_slice = ops::cnnl_slice(C, 0, batch_count, batch_count + seq, 1);
        auto c_view = ops::cnnl_view(c_slice, {seq, heads, 3, embed});
        auto c_input = ops::cnnl_squeeze(ops::cnnl_slice(c_view, 2, 1, 2, 1));
        //auto c_trans = ops::cnnl_permute(c_input, {1, 2, 0});

        a_trans = ops::cnnl_contiguous(a_trans);
        b_view = ops::cnnl_contiguous(b_view);
        auto result = ops::cnnl_bmm(ops::cnnl_mul(a_trans, alpha), b_view, false, false);
        //c_trans.copy_(result);
        auto s = ops::cnnl_permute_internal(result, {2, 0, 1});
        c_input.copy_(s);
        
        batch_count += seq;
        tokens += (seq * seq * 16);
        count += (heads * seq * seq);
    }

#else

    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));
    int batch_count = 0;
    int tokens = 0;
    int count = 0;

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
       
        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_view = ops::cnnl_view(a_slice, {seq, heads, 3, embed});
        auto a_trans = ops::cnnl_permute(a_view, {1, 2, 3, 0});       
 
        auto b_slice = ops::cnnl_slice(B, 0, count, count + heads * seq * seq, 1);
        auto b_view = ops::cnnl_view(b_slice, {heads, seq, seq}).contiguous();

        auto c_slice = ops::cnnl_slice(C, 0, batch_count, batch_count + seq, 1);
        auto c_view = ops::cnnl_view(c_slice, {seq, heads, 3, embed});
        auto c_trans = ops::cnnl_permute(c_view, {1, 2, 3, 0}).contiguous();

        void *ptrA = static_cast<void*>(static_cast<half*>(a_trans.data_ptr()));
        void *ptrB = static_cast<void*>(static_cast<half*>(b_view.data_ptr())); 
        void *ptrC = static_cast<void*>(static_cast<half*>(c_trans.data_ptr()) + 1 * embed * seq);  


        ops::cnnl_strided_bmm_internal(false, false, embed, seq, seq, alpha,
                ptrA, 3 * embed * seq,
                ptrB, seq * seq,
                ptrC, 3 * embed * seq,
                A.scalar_type(),
                heads);
       
        ops::cnnl_permute_out_internal(c_view, c_trans, {3, 0, 1, 2});


        // auto result = ops::cnnl_bmm(a_trans, b_trans, false, false);
        // //c_trans.copy_(result);
        // auto s = ops::cnnl_permute_internal(result, {2, 0, 1});
        // c_input.copy_(s);

        batch_count += seq;
        tokens += (seq * embed * heads);
        count += (heads * seq * seq);
    }



#endif


#ifdef PERF
    end.place();
    end.synchronize();
    float time = start.hardware_time(end);
    std::cout << __FUNCTION__ << " hardware time: " << time << std::endl;
#endif    
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm2Dgrad1_(torch::Tensor &A,
                     torch::Tensor &B,
                     torch::Tensor &C,
                     int batch,
                     torch::Tensor &seq_len,
                     int heads,
                     int embed,
		     bool scale,
		     bool strided,
		     bool enable_stream,
		     bool sync)
{
   
#ifdef PERF
    start.place();
#endif

#if 0
    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));
    int batch_count = 0;
    int tokens = 0;
    int count = 0;

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
        
        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_view = ops::cnnl_view(a_slice, {seq, heads, 3, embed});
        auto a_input = ops::cnnl_squeeze(ops::cnnl_slice(a_view, 2, 2, 3, 1));
        auto a_trans = ops::cnnl_permute(a_input, {1, 2, 0});
        a_trans = ops::cnnl_contiguous(a_trans);

        auto b_slice = ops::cnnl_slice(B, 0, batch_count, batch_count + seq, 1);
        auto b_view = ops::cnnl_view(b_slice, {seq, heads, embed});
        auto b_trans = ops::cnnl_permute(b_view, {1, 0, 2});
        b_trans = ops::cnnl_contiguous(b_trans);

        auto c_slice = ops::cnnl_slice(C, 0, count, count + heads * seq * seq, 1);
        auto c_view = ops::cnnl_view(c_slice, {heads, seq, seq});

        ops::cnnl_bmm_internal(c_view, b_trans, a_trans, false, false);

        batch_count += seq;
        tokens += (seq * embed * heads);
        count += (heads * seq * seq);
    }
#else
    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));
    int batch_count = 0;
    int tokens = 0;
    int count = 0;

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
        
        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_view = ops::cnnl_view(a_slice, {seq, heads, 3, embed});
        auto a_trans = ops::cnnl_permute(a_view, {1, 2, 3, 0}).contiguous();

        auto b_slice = ops::cnnl_slice(B, 0, batch_count, batch_count + seq, 1);
        auto b_view = ops::cnnl_view(b_slice, {seq, heads, embed});
        auto b_trans = ops::cnnl_permute(b_view, {1, 0, 2}).contiguous();
        //b_trans = ops::cnnl_contiguous(b_trans);

        auto c_slice = ops::cnnl_slice(C, 0, count, count + heads * seq * seq, 1);
        auto c_view = ops::cnnl_view(c_slice, {heads, seq, seq});

        void *ptrA = static_cast<void*>(static_cast<half*>(a_trans.data_ptr()) + 2 * embed * seq);
        void *ptrB = static_cast<void*>(static_cast<half*>(b_trans.data_ptr())); 
        void *ptrC = static_cast<void*>(static_cast<half*>(c_view.data_ptr()));  

        ops::cnnl_strided_bmm_internal(false, false, seq, seq, embed, one,
                ptrB, embed * seq,
                ptrA, 3 * embed * seq,
                ptrC, seq * seq,
                A.scalar_type(),
                heads);

        batch_count += seq;
        tokens += (seq * embed * heads);
        count += (heads * seq * seq);
    }

#endif



#ifdef PERF
    end.place();
    end.synchronize();
    float time = start.hardware_time(end);
    std::cout << __FUNCTION__ << " hardware time: " << time << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm1Dgrad2_(torch::Tensor &A,
                         torch::Tensor &B,
                         torch::Tensor &C,
                         int batch,
                         torch::Tensor &seq_len,
                         int heads,
                         int embed,
			 bool scale,
			 bool strided,
			 bool enable_stream,
			 bool sync)
{

#ifdef PERF
    start.place();
#endif

#if 0
    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));
    int batch_count = 0;
    int tokens = 0;
    int count = 0;

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
        
        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_view = ops::cnnl_view(a_slice, {seq, heads, 3, embed});
        auto a_input = ops::cnnl_squeeze(ops::cnnl_slice(a_view, 2, 1, 2, 1));
        auto a_trans = ops::cnnl_permute(a_input, {1, 2, 0});

        auto b_slice = ops::cnnl_slice(B, 0, count, count + heads * seq * seq, 1);
        auto b_view = ops::cnnl_view(b_slice, {heads, seq, seq});

        auto c_slice = ops::cnnl_slice(C, 0, batch_count, batch_count + seq, 1);
        auto c_view = ops::cnnl_view(c_slice, {seq, heads, 3, embed});
        auto c_input = ops::cnnl_squeeze(ops::cnnl_slice(c_view, 2, 0, 1, 1));
        //auto c_trans = ops::cnnl_permute(c_input, {1, 2, 0});

        a_trans = ops::cnnl_contiguous(a_trans);
        b_view = ops::cnnl_contiguous(b_view);

        auto result = ops::cnnl_bmm(ops::cnnl_mul(a_trans, alpha), b_view, false, true);
        //c_trans.copy_(result);
        auto s = ops::cnnl_permute_internal(result, {2, 0, 1});
        c_input.copy_(s);
        
        batch_count += seq;
        tokens += (seq * seq * 16);
        count += (heads * seq * seq);
    }
#else
    float one = 1.0, zero = 0.0, alpha = 1.0 / sqrt(static_cast<float>(embed));
    int batch_count = 0;
    int tokens = 0;
    int count = 0;

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
       
        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_view = ops::cnnl_view(a_slice, {seq, heads, 3, embed});
        auto a_trans = ops::cnnl_permute(a_view, {1, 2, 3, 0}).contiguous();
        
        auto b_slice = ops::cnnl_slice(B, 0, count, count + heads * seq * seq, 1);
        auto b_view = ops::cnnl_view(b_slice, {heads, seq, seq});

        auto c_slice = ops::cnnl_slice(C, 0, batch_count, batch_count + seq, 1);
        auto c_view = ops::cnnl_view(c_slice, {seq, heads, 3, embed});
        auto c_trans = ops::cnnl_permute(c_view, {1, 2, 3, 0}).contiguous();

        void *ptrA = static_cast<void*>(static_cast<half*>(a_trans.data_ptr()) + embed * seq);
        void *ptrB = static_cast<void*>(static_cast<half*>(b_view.data_ptr())); 
        void *ptrC = static_cast<void*>(static_cast<half*>(c_trans.data_ptr()));  

        ops::cnnl_strided_bmm_internal(false, true, embed, seq, seq, alpha,
                ptrA, 3 * embed * seq,
                ptrB, seq * seq,
                ptrC, 3 * embed * seq,
                A.scalar_type(),
                heads);

        ops::cnnl_permute_out_internal(c_view, c_trans, {3, 0, 1, 2});
        //auto s = ops::cnnl_permute_internal(c_temp, {3, 0, 1, 2});
        //c_view.copy_(s);

        batch_count += seq;
        tokens += (seq * seq * 16);
        count += (heads * seq * seq);
    }

#endif


#ifdef PERF
    end.place();
    end.synchronize();
    float time = start.hardware_time(end);
    std::cout << __FUNCTION__ << " hardware time: " << time << std::endl;
#endif

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastBmm2Dgrad2_(torch::Tensor &A,
                     torch::Tensor &B,
                     torch::Tensor &C,
                     int batch,
                     torch::Tensor &seq_len,
                     int heads,
                     int embed,
		     bool scale,
		     bool strided,
		     bool enable_stream,
		     bool sync)
{
#ifdef PERF
    start.place();
#endif

#if 0
    float one = 1.0, zero = 0.0;
    int batch_count = 0;
    int tokens = 0;
    int count = 0;

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
       
        auto b_slice = ops::cnnl_slice(B, 0, count, count + heads * seq * seq, 1);
        auto b_view = ops::cnnl_view(b_slice, {heads, seq, seq});
        auto b_trans = ops::cnnl_contiguous(b_view);

        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_trans = ops::cnnl_permute(a_slice, {1, 2, 0});
        a_trans = ops::cnnl_contiguous(a_trans);

        auto c_slice = ops::cnnl_slice(C, 0, batch_count, batch_count + seq, 1);
        auto c_view = ops::cnnl_view(c_slice, {seq, heads, 3, embed});
        auto c_input = ops::cnnl_squeeze(ops::cnnl_slice(c_view, 2, 2, 3, 1));
        //auto c_trans = ops::cnnl_permute(c_input, {1, 2, 0});

        auto result = ops::cnnl_bmm(a_trans, b_trans, false, false);
        //c_trans.copy_(result);
        auto s = ops::cnnl_permute_internal(result, {2, 0, 1});
        c_input.copy_(s);

        batch_count += seq;
        tokens += (seq * embed * heads);
        count += (heads * seq * seq);
    }
#else


    float one = 1.0, zero = 0.0;
    int batch_count = 0;
    int tokens = 0;
    int count = 0;

    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
       
        auto a_slice = ops::cnnl_slice(A, 0, batch_count, batch_count + seq, 1);
        auto a_trans = ops::cnnl_permute(a_slice, {1, 2, 0}).contiguous();
        
        auto b_slice = ops::cnnl_slice(B, 0, count, count + heads * seq * seq, 1);
        auto b_view = ops::cnnl_view(b_slice, {heads, seq, seq}).contiguous();

        auto c_slice = ops::cnnl_slice(C, 0, batch_count, batch_count + seq, 1);
        auto c_view = ops::cnnl_view(c_slice, {seq, heads, 3, embed});
        auto c_trans = ops::cnnl_permute(c_view, {1, 2, 3, 0});
        auto c_temp = at::empty(c_trans.sizes(), c_view.options());

        void *ptrA = static_cast<void*>(static_cast<half*>(a_trans.data_ptr()));
        void *ptrB = static_cast<void*>(static_cast<half*>(b_view.data_ptr())); 
        void *ptrC = static_cast<void*>(static_cast<half*>(c_temp.data_ptr()) + 2 * embed * seq);  


        ops::cnnl_strided_bmm_internal(false, false, embed, seq, seq, one,
                ptrA, embed * seq,
                ptrB, seq * seq,
                ptrC, 3 * embed * seq,
                A.scalar_type(),
                heads);
       
        ops::cnnl_permute_out_internal(c_view, c_temp, {3, 0, 1, 2});


        // auto result = ops::cnnl_bmm(a_trans, b_trans, false, false);
        // //c_trans.copy_(result);
        // auto s = ops::cnnl_permute_internal(result, {2, 0, 1});
        // c_input.copy_(s);

        batch_count += seq;
        tokens += (seq * embed * heads);
        count += (heads * seq * seq);
    }

#endif

#ifdef PERF
    end.place();
    end.synchronize();
    float time = start.hardware_time(end);
    std::cout << __FUNCTION__ << " hardware time: " << time << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastSoftmaxFprop_(torch::Tensor &input,
		  int batch,
                  torch::Tensor &seq_len,
		  int heads,
		  bool enable_stream,
		  bool sync)
{
#ifdef PERF
    start.place();
#endif
    int count = 0;
    int mc = 0;
    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
        auto input_slice = ops::cnnl_slice(input, 0, count, count + heads * seq * seq, 1);
        auto input_view = ops::cnnl_view(input_slice, {heads * seq, seq});
        ops::cnnl_softmax_out_internal(input_view, input_view, -1);
        count += (heads * seq * seq);
        mc += seq;
    }
#ifdef PERF
    start.place();
    end.place();
    end.synchronize();
    float time = start.hardware_time(end);
    std::cout << __FUNCTION__ << " hardware time: " << time << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastSoftmaxBprop_(torch::Tensor &input,
		       torch::Tensor &output,
                       int batch,
                       torch::Tensor &seq_len,
                       int heads,
		       bool enable_stream,
		       bool sync)
{
#ifdef PERF
    start.place();
#endif
    int count = 0;
    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
        auto input_slice = ops::cnnl_slice(input, 0, count, count + heads * seq * seq, 1);
        auto input_view = ops::cnnl_view(input_slice, {heads * seq, seq});
        //auto input_float = input_view.to(at::kFloat);

        auto output_slice = ops::cnnl_slice(output, 0, count, count + heads * seq * seq, 1);
        auto output_view = ops::cnnl_view(output_slice, {heads * seq, seq});
        //auto output_float = output_view.to(at::kFloat);

        ops::cnnl_softmax_backward_out_internal(output_view, output_view, input_view, 1, input_view);

        //auto result = ops::cnnl__softmax_backward_data(output_float, input_float, -1, input_float);
        //auto result = ops::cnnl_softmax_backward_internal(output_view, input_view, -1, input_view, CNNL_SOFTMAX_ACCURATE);
        //output_view.copy_(result);

        count += (heads * seq * seq);
    }
#ifdef PERF
    end.place();
    end.synchronize();
    float time = start.hardware_time(end);
    std::cout << __FUNCTION__ << " hardware time: " << time << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastMaskSoftmaxFprop_(torch::Tensor &input,
                           torch::Tensor &mask,
                           int batch,
                           torch::Tensor &seq_len,
                           int heads,
			   bool enable_stream,
			   bool sync)
{
#ifdef PERF
    start.place(); 
#endif
    int count = 0;
    int mc = 0;
    for (int i = 0; i < batch; ++i) {
        auto seq = seq_len[i].item().to<int>();
        auto input_slice = ops::cnnl_slice(input, 0, count, count + heads * seq * seq, 1);
        auto input_view = ops::cnnl_view(input_slice, {heads * seq, seq});
        auto mask_slice = ops::cnnl_slice(mask, 0, mc, mc + seq, 1);
        ops::cnnl_softmax_out_internal(input_view, input_view + mask_slice, -1);
        count += (heads * seq * seq);
        mc += seq;
    }
#ifdef PERF
    end.place();
    end.synchronize();
    float time = start.hardware_time(end);
    std::cout << __FUNCTION__ << " hardware time: " << time << std::endl;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<torch::Tensor> FastMaskSoftmaxDropoutFprop_(torch::Tensor &input,
                                  torch::Tensor &mask,
                                  int batch,
                                  torch::Tensor &seq_len,
                                  int heads,
                                  float dropout_prob,
                                  bool enable_stream,
                                  bool sync,
                                  bool is_training)
{
    return {input, mask};
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void FastMaskSoftmaxDropoutBprop_(torch::Tensor &input,
                              torch::Tensor &output,
                              torch::Tensor &dropout_mask,
                              int batch,
                              torch::Tensor &seq_len,
                              int heads,
                              float dropout_prob,
                              bool enable_stream,
                              bool sync)
{
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void init_mha_cuda_extension()
{
    // CNRT Stream
    for (int i = 0; i < nstreams; ++i) {
        cnrtQueueCreate(&stream[i]);
    }
    // CNNL Handle
    cnnlCreate(&handle);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("InitMHACUDAExtension", &init_mha_cuda_extension, "InitMHACUDAExtension");
  m.def("FastBmm1Fprop", &FastBmm1Fprop_, "FastBmm1Fprop");
  m.def("FastBmm1Dgrad1", &FastBmm1Dgrad1_, "FastBmm1Dgrad1"); 
  m.def("FastBmm1Dgrad2", &FastBmm1Dgrad2_, "FastBmm1Dgrad2"); 
  m.def("FastBmm2Fprop", &FastBmm2Fprop_, "FastBmm2Fprop");
  m.def("FastBmm2Dgrad1", &FastBmm2Dgrad1_, "FastBmm2Dgrad1");
  m.def("FastBmm2Dgrad2", &FastBmm2Dgrad2_, "FastBmm2Dgrad2");
  m.def("FastSoftmaxFprop", &FastSoftmaxFprop_, "FastSoftmaxFprop");
  m.def("FastSoftmaxBprop", &FastSoftmaxBprop_, "FastSoftmaxBprop");
  m.def("FastMaskSoftmaxFprop", &FastMaskSoftmaxFprop_, "FastMaskSoftmaxFprop");
  m.def("FastMaskSoftmaxDropoutFprop", &FastMaskSoftmaxDropoutFprop_, "FastMaskSoftmaxDropoutFprop");  
  m.def("FastMaskSoftmaxDropoutBprop", &FastMaskSoftmaxDropoutBprop_, "FastMaskSoftmaxDropoutBprop");
}
