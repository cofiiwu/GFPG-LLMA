#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thread>
#include <thrust/scan.h>

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include "fpgrowth.h"
#include <assert.h>
#include <sys/time.h>
#include <cub/cub.cuh>

__device__ __host__ unsigned int round_up_pow2(unsigned int v){
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}

#define ENABLE_ASSERT 0
#define HASH_LOAD_FACTOR (1.0f)
#define NULL_NODE (NULL)
#define STOP_NODE (0xFFFFFFFF)
#define SPLIT_NODE (-2)
#define FREQ_LEN	(1)
#define STOP_LEN	(1)
#define PAT_LEN	(1)
#define PATTERN_INFO_LEN (PAT_LEN+STOP_LEN+FREQ_LEN)
#define MAX_DEPTH (300)
//the size of ia size is based on the
#define IA_SIZE_BASE_ITEM	(0)
//the size of ia size is summation of all items
#define IA_SIZE_EACH_ITEMS	(1)
#define HT_IARRAY_LEN_PER_ITEM(n_node) (round_up_pow2((unsigned int)ceil(n_node * (1/HASH_LOAD_FACTOR))))
#define HT_IARRAY_LEN(n_fi, n_node) (n_fi * HT_IARRAY_LEN_PER_ITEM(n_node))
#define HT_IARRAY_SIZE(n_fi, n_node) (unsigned int)(HT_IARRAY_LEN(n_fi, n_node)* sizeof(unsigned int))
#define HT_SIZE(n_fi, n_node)	(unsigned int)(4*sizeof(unsigned int) + 3 * n_fi *sizeof(unsigned int) + 2*HT_IARRAY_SIZE(n_fi, n_node))
//#define PAT_WITH_PATHID

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
void free_gpu_mem(void);
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
int *d_dummy;
__host__ void gpu_dummy_alloc_thread(){
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_dummy, sizeof(int)));
}

int  **h_pattern_ans;

__constant__  unsigned int d_plen_level_c[MAX_DEPTH];//the length of a pattern of a level = plen_item[k]*num_pnode[k]
__constant__  unsigned int d_level_clen[MAX_DEPTH];
__constant__  unsigned long long c_fi_with_baseitem[64*63/2];
__constant__ unsigned int c_msup[1];
__constant__ unsigned int c_num_fi[1];
__constant__ unsigned int c_num_fpnode[1];
__constant__ unsigned int c_num_res_vector[1];
__constant__ unsigned int c_gtree_item_base[MAX_FI];

__host__ void alloc_gpu_cltree(CSTREE *cst){
	int num_fpnode = cst->num_fpnode;
	int num_fi = cst->cnt;
	printf("gtree_buf size=%u\n",cst->gtree_size);
	void *d_gtreebuf;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_gtreebuf,  cst->gtree_size));
	cst->d_gtree_itembase = (unsigned int*)d_gtreebuf;
	cst->d_gtree =  (GTREE_NODE*)(cst->d_gtree_itembase + (((cst->cnt+1)+2)&~1));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&h_pattern_ans,sizeof(int*) * num_fpnode));

}






unsigned int result_size = 0;
void *h_res ;


#define NUM_THREAD_BUILD_TAB (512)
#define UNLOCK (0)
#define LOCK (1)
#define ITEM_LOCK (-1)


unsigned int cal_next_tab_size(const unsigned int *aux_in, const unsigned int num_entry_tab_in, const unsigned int *ia_num, unsigned int *tab_out_offset){
	unsigned int next_tab_size = sizeof(unsigned int)*2;//total #fi + next_table_size
	for(unsigned int i=0; i<num_entry_tab_in;i++){
		tab_out_offset[i] = next_tab_size;
		if(aux_in[i])
			next_tab_size += HT_SIZE(i, ia_num[i]);
		printf("tab_out_offset[%d]=%u\n", i,tab_out_offset[i]);
	}
	return next_tab_size; // the extra unit is for counting the total number of frequent items
}


#if 0
__device__ unsigned int find_bid_base_idx(unsigned int bid, const unsigned int* __restrict__ d_io){
	const unsigned io_cnt = d_io[0];
	const unsigned* __restrict__ io = d_io+1;
	int find = 0;
	int i;
	for(i=1;i<io_cnt;i++){
		if(bid<io[i]){
			find =1;
			break;
		}
	}
	if(!find)
		return io_cnt-1;
	else
		return i-1;

}
#endif


__forceinline__ __device__ unsigned int find_bid_base_idx_bsearch(unsigned int bid, const unsigned int* __restrict__ d_io){
	const unsigned io_cnt = d_io[0];
	const unsigned int* __restrict__ io = d_io+1;
	unsigned int l=0, r = io_cnt-1, find = 0;

	do{
		unsigned int m = (l+r)/2;
//		if(!m)
//			return 0;
//		else if(m == io_cnt-1)
//			return io_cnt-1;
		if(io[m]<=bid && bid<io[m+1]){
			return m;
		}else if(bid<io[m]){
			r = m;
		}else if(bid>=io[m+1]){
			l = m+1;
		}

	}while(r>l);
	return io_cnt-1;
}

#define EMPTY_SLOT	((unsigned)-1)
#define HASH_FULL	((unsigned)-2)
#ifndef DBG_HASH
#define DBG_HASH 0
#endif
__forceinline__ __device__ unsigned int hash_qprob_insert(const unsigned int nidx, const unsigned int num_slot, unsigned int* __restrict__ idx_tab,
		unsigned int* __restrict__ cnt_tab, const unsigned int val, const int item){

	int retry=0;
	for(int i=0;i<num_slot;i++){
#if 0
		unsigned int idx = (unsigned int)((nidx & (num_slot-1)) + 0.5*i + 0.5 *i*i) & (num_slot-1);

		if(idx_tab[idx]==nidx){
			atomicAdd_block(&cnt_tab[idx], val);
			return nidx;
		}else if(atomicCAS_block(&idx_tab[idx], EMPTY_SLOT, nidx) == EMPTY_SLOT){
			//atomicAdd_block(&cnt_tab[idx], val);
			atomicExch_block(&cnt_tab[idx], val); //init
			return EMPTY_SLOT;
		}
#endif
#if 1
		//unsigned int idx = (unsigned int)(nidx % num_slot + 0.5*i + 0.5 *i*i) % num_slot;
		unsigned int idx = (unsigned int)((nidx & (num_slot-1)) + 0.5*i + 0.5 *i*i) & (num_slot-1);
		unsigned int ret = atomicCAS_block(&idx_tab[idx], EMPTY_SLOT, nidx);
		if((ret == EMPTY_SLOT)){
#if DBG_HASH
			printf("<%u, %u> HIT(TRY:%d) #m=%u add item:%d nid:%u at %u ret %u (%p) val:%d\n",blockIdx.x, threadIdx.x, retry, num_slot, item, nidx, idx, ret, &idx_tab[idx],val);
#endif
			atomicExch_block(&cnt_tab[idx], val);
			return ret;
		}else if(ret == nidx){
			atomicAdd_block(&cnt_tab[idx], val);
			return ret;
		}
		retry++;
#if DBG_HASH
		printf("<%u, %u> CONFLICT #m=%u add item:%d nid:%u at %u ret %u (%p)\n",blockIdx.x, threadIdx.x, num_slot, item, nidx, idx, ret, &idx_tab[idx]);
#endif
#endif
	}
	return HASH_FULL;
}


#define QHASH
__forceinline__ __device__ unsigned int hash_node_idx(const unsigned int nidx, const unsigned int num_slot, unsigned int* __restrict__ idx_tab,
		unsigned int* __restrict__ cnt_tab, const unsigned int val, const int item){
#ifdef QHASH
	return  hash_qprob_insert(nidx, num_slot, idx_tab, cnt_tab, val, item);
#endif
}


__device__ void* block_init_array(unsigned int* __restrict__ base, const unsigned int len, const unsigned int val){
	unsigned int tid = threadIdx.x;
	for(int i=tid; i<len; i+=blockDim.x){
		base[i] = val;
	}
}

#ifndef DBG_FPG_ITER
#define DBG_FPG_ITER 0
#endif

#define GLOBAL_TAB_HEADER_SIZE_BYTE (2 * sizeof(unsigned int))

__global__ void kernel_fpg_iter_gtree(const void *d_tab_in, void *d_tab_out,
		const unsigned long long* __restrict__ tab_in_offset, const unsigned long long* __restrict__ tab_out_offset,
		const GTREE_NODE* __restrict__ gtree,
		const unsigned int smin, const unsigned int* __restrict__ relatived_bid_in, const unsigned int* __restrict__ wo_remap_raw,
		unsigned long long* __restrict__ pat_raw, unsigned int* __restrict__ freq_raw, const int max_item, const unsigned int* __restrict__ d_wide_tab_id){
	unsigned int tid = threadIdx.x; //the kernel function should be executed by one block only

	unsigned long long* __restrict__ pat_cnt = pat_raw;
	unsigned long long* __restrict__ pat = pat_cnt+1;
	unsigned int *freq = freq_raw;
	//unsigned int bid_offset_idx = find_bid_base_idx(blockIdx.x, relatived_bid_in); //add head
	//unsigned int bid_offset_idx =  find_bid_base_idx_bsearch(blockIdx.x, relatived_bid_in);
	unsigned int bid_offset_idx = d_wide_tab_id[blockIdx.x];
	unsigned int bid_offset = relatived_bid_in[bid_offset_idx+1];
//	unsigned int res = find_bid_base_idx_bsearch(blockIdx.x, relatived_bid_in);
//	assert(bid_offset_idx==res);

	const int rel_bid = blockIdx.x - bid_offset;

	const unsigned long long* __restrict__ ro = tab_in_offset + 1;

	const unsigned long long* __restrict__ wo = tab_out_offset + 1;
	const unsigned int* __restrict__ wo_remap = wo_remap_raw + 1;

	unsigned int* __restrict__ global_num_fi_out = (unsigned int* )d_tab_out;

	unsigned int* __restrict__ global_next_tab_size_out = global_num_fi_out + 1;
	const unsigned int* __restrict__ global_num_fi_in = (unsigned int* )d_tab_in;
	const unsigned int* __restrict__ global_next_tab_size_in = global_num_fi_in + 1;

    unsigned int* __restrict__ tab_out = (unsigned int*)((uintptr_t)d_tab_out + wo[wo_remap[blockIdx.x]]+GLOBAL_TAB_HEADER_SIZE_BYTE);



	const unsigned long long r_offset = ro[bid_offset_idx];
	const unsigned int* __restrict__ tab_in = (unsigned int*)((uintptr_t)d_tab_in + r_offset +GLOBAL_TAB_HEADER_SIZE_BYTE);
    const unsigned int* __restrict__ n_fi =  tab_in;
    const unsigned int num_fi = *n_fi;
    const unsigned int* __restrict__ ia_type = n_fi+1;
    const unsigned int* __restrict__ ia_size = ia_type+1;
    const unsigned int* __restrict__ basepat_idx = ia_size +1;
    const unsigned int* __restrict__ items = basepat_idx +1;
    const unsigned int* __restrict__ supps = items + num_fi;
    const unsigned int* __restrict__ ia_num = supps + num_fi;
    const unsigned int* __restrict__ ia_arrays; //= ia_num + n_node;
    const unsigned int* __restrict__ node_counts;// = ia_arrays + n_node;
    unsigned int item = items[rel_bid];
#if ENABLE_ASSERT
    assert(item<max_item);
#endif
    //assert(item<5);
    unsigned int supp = supps[rel_bid];
    unsigned int num_path = ia_num[rel_bid];
    unsigned int num_try_path =  HT_IARRAY_LEN_PER_ITEM(*ia_size);//round_up_pow2((unsigned)ceil((float)*ia_size / (float)HASH_LOAD_FACTOR));
    unsigned int chunk_size = (unsigned)ceil((float)num_try_path/blockDim.x);
    unsigned long long pat_idx;

#if 0
 	if(tid==0)
 	    		printf("<%u, %u, %u> item:%u supp:%u\n",blockIdx.x,rel_bid,tid, item,supp);
#endif
    if(supp < smin){

    	return;// all threads of the block return
    }
    else{
    	//fill the pattern
    	if(tid == 0){

    		pat_idx = atomicAdd(pat_cnt, 1);
    		int pat_base = pat_idx * *c_num_res_vector;
    		int sub_idx = item>>6; // DIV 64
    		if(*basepat_idx == (unsigned int)-1){
    			for(int i=0;i<*c_num_res_vector;i++){
    				if(i==sub_idx)
    					pat[pat_base+i] = 1ULL<<(item & 63);
    				else
    					pat[pat_base+i] = 0;
    			}

    			freq[pat_idx] = supp;
#if 0
    			printf("<%u, %u, %u> 1 item:%u pat_idx=%lu pat=0x%016llx freq:%u\n", blockIdx.x,rel_bid,tid, item, pat_idx,pat[pat_idx],freq[pat_idx]);
#endif
#if 0
    			for(int i=0;i<*c_num_res_vector;i++){
    				if(i==sub_idx)
    					pat[pat_base+i] = 1ULL<<(item & 63);
    				else
    					pat[pat_base+i] = 0;
    			}

#endif

    		}
    		else{

    			for(int i=0;i<*c_num_res_vector;i++){
    				if(i==sub_idx)
    					pat[pat_base+i] = pat[*basepat_idx * *c_num_res_vector + i] | 1ULL<<(item & 63);
    				else
    					pat[pat_base+i] = pat[*basepat_idx * *c_num_res_vector + i];//copy
    			}

    			freq[pat_idx] = supp;
#if 0
    			printf("<%u, %u, %u> 2 basepat_idx=%u pat[*basepat_idx]=0x%016llx item:%u pat_idx=%lu pat=0x%016llx freq:%u\n",
    					blockIdx.x,rel_bid,tid, *basepat_idx, pat[*basepat_idx] , item, pat_idx, pat[pat_idx], freq[pat_idx] );
#endif
    		}
    		if(item)
    			atomicAdd(global_num_fi_out, item);
    	}
    }
#if DBG_FPG_ITER
    __syncthreads();
#endif

    //tab_out_offset[0] --> no next run
    if(!item )
    	return;
    if(!num_fi)
    	return;
    if(!tab_out_offset[0])
    	return;




#if 0
   if(tid==0)
    	printf("<%d, %d, %d> bid_offset:%u d_tab_in:%p tab_in:%p d_tab_out:%p tab_out:%p wo_remap=%u wo(remap)=%u(0x%x)\n",blockIdx.x, rel_bid, tid, bid_offset, d_tab_in,tab_in, d_tab_out, tab_out, wo_remap[blockIdx.x],wo[wo_remap[blockIdx.x]] );
#endif
     	ia_arrays = ia_num + *n_fi;
     	node_counts = ia_arrays +  HT_IARRAY_LEN(*n_fi, *ia_size);


    //for new table
   	unsigned int* __restrict__ new_n_fi =  tab_out;
   	*new_n_fi = item; //0~ item-1
   	unsigned int* __restrict__ new_ia_type = new_n_fi+1;
   	*new_ia_type = IA_SIZE_BASE_ITEM;
   	unsigned int* __restrict__ new_ia_size = new_ia_type+1;
   	*new_ia_size = num_path;
   	unsigned int* __restrict__ new_basepat_idx = new_ia_size +1;
   	unsigned int* __restrict__ new_items = new_basepat_idx +1;
   	unsigned int* __restrict__ new_supps = new_items + *new_n_fi;
   	unsigned int* __restrict__ new_item_ia_num = new_supps + *new_n_fi;
   	unsigned int* __restrict__ new_item_ia_arrays = new_item_ia_num + *new_n_fi;
   	unsigned int* __restrict__ new_node_counts = new_item_ia_arrays +  HT_IARRAY_LEN(*new_n_fi, *new_ia_size);

   	unsigned int new_iarray_len_per_item =  HT_IARRAY_LEN_PER_ITEM(*new_ia_size);
   	unsigned int new_iarray_len =  HT_IARRAY_LEN(*new_n_fi, *new_ia_size);

   	//unsigned int strip_size = max((unsigned int)ceilf(((float)new_iarray_len)/blockDim.x),(unsigned int)blockDim.x);
   	//block_init_array(new_item_ia_arrays, new_iarray_len, strip_size, EMPTY_SLOT);

   	block_init_array(new_item_ia_arrays, new_iarray_len, EMPTY_SLOT);
   //	block_init_array(new_node_counts, new_iarray_len, 0);
//   	if(tid==0)
  // 		memset(new_item_ia_arrays, 0xFF, new_iarray_len*sizeof(int));

   	if(tid==0)
   		*new_basepat_idx = pat_idx;


    for(int i= tid; i<item; i+=blockDim.x){
    	new_items[i] = i;
    	new_supps[i] = 0;
    	new_item_ia_num[i] = 0;
    }



    __syncthreads();//necessary id blocksize>32

#if DBG_FPG_ITER
    if(tid==0)
    	printf("P <%u, %u> item:%d num_path:%u\n",blockIdx.x, tid, item, num_path);
#endif
    if(tid<min(blockDim.x, num_try_path)){
#if 0
    	printf("<%u, %u> item:%d supp:%d\n",blockIdx.x, tid, item,supp);
#endif
    	if(supp<smin){
    		 *new_n_fi = 0;

    		 return;
    	}
#if 0
    	printf("<%u, %u> item:%d try path %d ~% d\n",blockIdx.x, tid, item,chunk_size*tid, chunk_size*(tid +1));
#endif
    	//for(unsigned int path_idx=chunk_size*tid ; (path_idx<chunk_size*(tid +1)) && (path_idx<num_try_path); path_idx++){
    	for(unsigned int path_idx=tid ; (path_idx<num_try_path); path_idx+=blockDim.x){
    		unsigned int item_ia_idx;
			//get base index in its index array
			item_ia_idx = num_try_path * item + path_idx;
			unsigned int start_supp = node_counts[item_ia_idx];
			unsigned int start_idx = ia_arrays[item_ia_idx];
#if 0
			if(start_idx != EMPTY_SLOT)
				printf("<b:%u, rb:%u, tid:%u> path_idx:%u(m:%u #p=%u #fp=%u) ia_idx:%u start_idx:%u start_supp:%u item_ia_idx:%u\n",blockIdx.x, rel_bid, tid, path_idx, num_try_path, num_path,*s_num_finished_path,item_ia_idx, start_idx, start_supp, item_ia_idx);
#endif
			if(start_idx == EMPTY_SLOT)
				continue;//next path
#if ENABLE_ASSERT
			assert(start_supp>0);
#endif
            const GPU_TREE_NODE *n;
            n = &gtree[start_idx];
            int pitem = n->pitem;
			unsigned int pidx;
//			printf("1st pitem=%d\n", pitem);
			while(pitem!=ROOT_ITEM){
				pidx = c_gtree_item_base[pitem] + n->index;
				n = &gtree[pidx];
#if 0
				printf("blk:%d(rel_bid:%d) tid:%d idx:%d cur_item=%d pidx=%d\n", blockIdx.x, rel_bid, tid, idx, cur_item,pidx );
#endif

				//search array index
				unsigned int tmp_item_ia_base = new_iarray_len_per_item * pitem ;//for filling new table's IA
//					printf("<%d> base_vec:0x%016llx pathid:%d cur_item:%d tmp_item_ia_base:%d \n", tid, base_vec,path_idx,cur_item,tmp_item_ia_base);
				//assert(cur_item< item);
				atomicAdd_block(&new_supps[pitem], start_supp);

				//hash nodes to assigned item
				unsigned int hash_ret;
				hash_ret = hash_node_idx(pidx, new_iarray_len_per_item, new_item_ia_arrays + tmp_item_ia_base, new_node_counts+tmp_item_ia_base, start_supp,pitem );
				if(hash_ret == EMPTY_SLOT){
#if 1
				//	printf("blk:%d(rel_bid:%d) tid:%d item:%u @%p ++\n",blockIdx.x, rel_bid, tid, cur_item, &new_item_ia_num[cur_item]);
#endif
					atomicAdd_block(&new_item_ia_num[pitem],1);
				}
				else
					assert(hash_ret!=HASH_FULL);
				pitem = n->pitem;

			}

    	}

    }
	//__syncthreads();
	 //global_next_tab_size

#if 0
	if(tid < *n_fi){
		unsigned int subtab_size = gpu_cal_next_tab_size(items, ia_num);
		atomicAdd(global_next_tab_size, subtab_size);
	}

	if(tid==0 && blockIdx.x==0)
		atomicAdd(global_next_tab_size, 2*sizeof(unsigned int)); //total #FI + next_tab_size
#endif


}

/*
 * chess support 30%
 * #define MAX_RO_SIZE (50<<20)
 * #define MAX_WO_SIZE (50<<20)
 * #define MAX_IO_SIZE (50<<20)
 * #define MAX_PAT_SIZE (200<<20)
 * #define MAX_FREQ_SIZE (100<<20)
 * #define MAX_REMAP (5000000-1)
 *
 * */
#define MAX_RO_SIZE (50<<20)
#define MAX_WO_SIZE (50<<20)
#define MAX_IO_SIZE (50<<20)
#define MAX_PAT_SIZE (200<<20)
#define MAX_FREQ_SIZE (100<<20)
#define MAX_REMAP (1000000-1)
#define MAX_REMAP_SIZE ((MAX_REMAP+1)*sizeof(unsigned int))
#ifndef DBG_CAL_OFFSET
#define DBG_CAL_OFFSET	0
#endif
__global__ void kernel_cal_offset(const unsigned int* __restrict__ tab_in,
	unsigned int* __restrict__ pre_idx_offset_raw, unsigned int* __restrict__ new_idx_offset_raw,
	unsigned long long* __restrict__ pre_wo_raw, unsigned long long* __restrict__ new_wo_raw,
	unsigned int* __restrict__ remap_raw, const unsigned int msup, const unsigned max_item, unsigned int* __restrict__ d_wide_tab_id){	//, unsigned int* __restrict__ new_ro_raw){
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x; // a thread = a block
	unsigned int *remap_size = remap_raw;
	unsigned int *remap = remap_size + 1;
	assert(*remap_size<MAX_REMAP);
	unsigned int* __restrict__ idx_offset_size = pre_idx_offset_raw;
	const unsigned int* __restrict__ idx_offset= idx_offset_size + 1;
	const unsigned long long* __restrict__ pre_wo_offset= pre_wo_raw + 1;
	const unsigned int* __restrict__ total_num_fi = tab_in;
	const unsigned int* __restrict__ next_tab_size = total_num_fi+1;
	unsigned long long* __restrict__ new_wo_size = new_wo_raw;
	unsigned long long* __restrict__ new_wo = new_wo_size+1;
	unsigned int* __restrict__ new_io_size = new_idx_offset_raw;
	unsigned int* __restrict__ new_io = new_io_size+1;

	const unsigned int* __restrict__ tab;
	if(tid < *total_num_fi){
		int tab_idx = find_bid_base_idx_bsearch(tid, pre_idx_offset_raw);
		d_wide_tab_id[tid] = tab_idx;

#if DBG_CAL_OFFSET
		printf("<%u> tab_idx=%u wo=%llu\n", tid,tab_idx,pre_wo_offset[tab_idx]);
#endif
		//new_ro[tid] = pre_wo_offset[tab_idx];
		tab = (unsigned int*)((uintptr_t)tab_in + pre_wo_offset[tab_idx] + GLOBAL_TAB_HEADER_SIZE_BYTE);
		const unsigned int* __restrict__ n_fi =  tab;
		const unsigned int num_fi = *n_fi;
		const unsigned int* __restrict__ ia_type = n_fi+1;
		const unsigned int* __restrict__ ia_size = ia_type+1;
		const unsigned int* __restrict__ basepat_idx = ia_size +1;
		const unsigned int* __restrict__ items = basepat_idx +1;
		const unsigned int* __restrict__ supps = items + num_fi;
		const unsigned int* __restrict__ ia_num = supps + num_fi;
//		const unsigned int* __restrict__ ia_arrays; //= ia_num + n_node;
//		const unsigned int* __restrict__ node_counts;// = ia_arrays + n_node;
		unsigned int rel_tid = tid -idx_offset[tab_idx];
		unsigned int item = items[rel_tid];
		unsigned int supp = supps[rel_tid];
#if DBG_CAL_OFFSET
		printf("<%u, rel_tid:%u> item:%u supp:%d\n",tid,rel_tid, item, supp);
#endif
		assert(item<max_item);
		if(item && (supp >= msup)){
			//unsigned int num_path = ia_num[rel_tid];
			//unsigned int num_path = *ia_size;
			unsigned int num_path;
			if(*ia_type==IA_SIZE_EACH_ITEMS)
				num_path = ia_num[rel_tid];
			else
				num_path = *ia_size;
			remap[tid] =  atomicAdd(remap_size, 1);
//#if DBG_CAL_OFFSET
//			printf("<%u, rel_tid:%u> item:%u remap=%u\n",tid,rel_tid, item,remap[tid]);
//#endif
			//*new_wo_size = *total_num_fi;

			new_wo[remap[tid]] = HT_SIZE(item, num_path);
#if DBG_CAL_OFFSET
			printf("<%u, rel_tid:%u> item:%u num_path=%u new_wo[%u]=%llu(0x%X) /%llu MB \n",tid,rel_tid, item,num_path,remap[tid],new_wo[remap[tid]],new_wo[remap[tid]],new_wo[remap[tid]]>>20);
#endif
			//*new_io_size = *total_num_fi;
			new_io[remap[tid]] = item;
		//	*new_ro_size = *total_num_fi;
			atomicAdd(new_wo_size, 1);
			atomicAdd(new_io_size, 1);
		}
	}
}
//int max_node_item = 0;

#define DEBUG_FINAL_GET_TAB_SIZE 0
__host__ void get_num_next_tabs(unsigned long long *d_tmp_wo_raw, unsigned num, unsigned long long *res){
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	//unsigned long long *d_total;
	//cudaMalloc(&d_total, sizeof(unsigned long long));
#if DEBUG_FINAL_GET_TAB_SIZE
	unsigned long long *wobuf_tmp = (unsigned long long*) malloc(MAX_WO_SIZE);
	CUDA_CHECK_RETURN(cudaMemcpy(wobuf_tmp, d_tmp_wo_raw, MAX_WO_SIZE, cudaMemcpyDeviceToHost));//copy the context with counter
#endif
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_tmp_wo_raw+1, d_tmp_wo_raw, num);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run sum-reduction
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_tmp_wo_raw+1, d_tmp_wo_raw, num);

#if DEBUG_FINAL_GET_TAB_SIZE
	CUDA_CHECK_RETURN(cudaMemcpy(wobuf_tmp, d_tmp_wo_raw, MAX_WO_SIZE, cudaMemcpyDeviceToHost));//copy the context with counter
#endif

	// the uint on d_tmp_wo_raw is borrowed for saving the sum or the new tab
	CUDA_CHECK_RETURN(cudaMemcpy(res, d_tmp_wo_raw, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

}
#define DEBUG_FINAL_WO 0
__host__ void final_next_write_offset(unsigned long long *d_dst, unsigned long long *d_src, int num)
{
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
#if DEBUG_FINAL_WO
	unsigned long long *wobuf_tmp = (unsigned long long*) malloc(MAX_WO_SIZE);
	CUDA_CHECK_RETURN(cudaMemcpy(wobuf_tmp, d_dst-1, MAX_WO_SIZE, cudaMemcpyDeviceToHost));//copy the context with counter
#endif
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_src, d_dst, num);
	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	// Run exclusive prefix sum
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_src, d_dst, num);
#if DEBUG_FINAL_WO
	CUDA_CHECK_RETURN(cudaMemcpy(wobuf_tmp, d_dst-1, MAX_WO_SIZE, cudaMemcpyDeviceToHost));//copy the context with counter
#endif
}
__host__ void fianl_next_index_offset(unsigned int *d_dst, unsigned int *d_src, int num)
{
	//prefix sum for IO
	void     *d_temp_storage = NULL;
	size_t   temp_storage_bytes = 0;
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_src, d_dst, num);
		// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
		// Run exclusive prefix sum
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_src, d_dst, num);
#if 0
	unsigned int *iobuf_tmp = (unsigned int*) malloc(MAX_IO_SIZE);
	CUDA_CHECK_RETURN(cudaMemcpy(iobuf_tmp, d_dst-1, MAX_IO_SIZE, cudaMemcpyDeviceToHost));//copy the context with counter

#endif
}

__host__ long  cuda_main(CSTREE *cst, SUPP smin)
{

	int num_fpnode = cst->num_fpnode;
	int num_fi = cst->cnt;
	cst->real_max_depth += 1;
	printf("worst max_depth =%d real max_dekpth=%d\n",cst->max_depth,cst->real_max_depth);
	alloc_gpu_cltree(cst);

	CUDA_CHECK_RETURN(cudaMemcpy(cst->d_gtree_itembase, cst->h_gtree_itembase,cst->gtree_size, cudaMemcpyHostToDevice));

	assert(MAX_DEPTH > cst->real_max_depth);
	//CUDA_CHECK_RETURN(cudaMemcpyToSymbol (d_level_clen, cst->level_clen, (cst->real_max_depth) *sizeof(unsigned int )));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol (c_msup, &smin, sizeof(unsigned int)));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol (c_num_fi, &cst->cnt, sizeof(unsigned int )));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol (c_num_fpnode, &cst->num_fpnode, sizeof(unsigned int )));
	unsigned int num_res_vector = ((cst->cnt + 63) & ~63) >> 6;
	printf("cst->cnt:%d num_res_vector=%u\n", cst->cnt, num_res_vector);
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol (c_num_res_vector, &num_res_vector, sizeof(unsigned int )));
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol (c_gtree_item_base, cst->h_gtree_itembase, cst->cnt*sizeof(unsigned int )));

	void *global_htab_buf;
	void *d_global_htab_buf;
	int max_num_node = 0;
	for(int i=0; i< cst->cnt;i++){
	  if(cst->heads[i].cnt > max_num_node)
	    max_num_node = cst->heads[i].cnt;
	}
	unsigned int tab_size = HT_SIZE(cst->cnt, max_num_node) + 2 * sizeof(unsigned int);
	CUDA_CHECK_RETURN(cudaMallocHost((void**) &global_htab_buf, tab_size));
	memset(global_htab_buf, 0, tab_size);
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_global_htab_buf, tab_size));
	CUDA_CHECK_RETURN(cudaMemset(d_global_htab_buf, 0, tab_size));
	printf("global htb %p ~ %p\n", global_htab_buf, global_htab_buf+ tab_size);
	printf("d_global htb %p ~ %p\n", d_global_htab_buf, (unsigned long long)d_global_htab_buf+ tab_size);
	unsigned int *n_global_fi = (unsigned int*) global_htab_buf;
	*n_global_fi = cst->cnt;
	unsigned int *next_table_size = n_global_fi + 1;
	unsigned int *n_fi =  next_table_size+1;
	unsigned int *type_ia_size = n_fi+1;
	unsigned int *ia_size = type_ia_size+1;
	unsigned int *basepat_idx = ia_size+1;
	*basepat_idx= -1; //means NULL
	unsigned int *items = basepat_idx+1;
	unsigned int *supps = items + cst->cnt;
	unsigned int *ia_num = supps + cst->cnt;
	unsigned int *ia_arrays = ia_num + cst->cnt;
	memset(ia_arrays, -1, HT_IARRAY_LEN(cst->cnt, max_num_node)*sizeof(unsigned));
	unsigned int *node_counts = ia_arrays + HT_IARRAY_LEN(cst->cnt, max_num_node);
	*type_ia_size = IA_SIZE_EACH_ITEMS;
	*n_fi =  cst->cnt;
	*ia_size = max_num_node;

	//fill 1st htb
	for(int i=0;i< cst->cnt;i++){
	    static unsigned int pre_node = 0;
	    items[i]=i;
	    supps[i]=cst->heads[i].supp;
	    ia_num[i] = cst->heads[i].cnt;
	    for(int j=0;j<cst->heads[i].cnt; j++){
	    	ia_arrays[pre_node+j] = cst->h_gtree_itembase[i]+j; //index in the gtree
	    	node_counts[pre_node+j] = cst->h_gtree[cst->h_gtree_itembase[i]+j].freq;

	    }
	    pre_node += HT_IARRAY_LEN_PER_ITEM(max_num_node);
	}

	CUDA_CHECK_RETURN(cudaMemcpy(d_global_htab_buf, global_htab_buf, tab_size, cudaMemcpyHostToDevice));

	unsigned int *relatived_id_in, *d_relatived_id_in;
	unsigned long long next_tab_size;
	void *d_tab_in, *d_tab_out;
	unsigned long long *d_tab_in_offset, *d_tab_out_offset;
	unsigned int *tab_out;

	d_tab_in = d_global_htab_buf;
	unsigned int num_result_entry = 0;
	unsigned int *global_num_fi;
	CUDA_CHECK_RETURN(cudaMallocHost((void**) &global_num_fi, sizeof(unsigned int)));

	void *buf_pool;

	unsigned int total_buf_size = (MAX_WO_SIZE+MAX_IO_SIZE)*2 + MAX_RO_SIZE + MAX_REMAP_SIZE;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &buf_pool, total_buf_size));
	unsigned int *d_idx_offset_raw, *d_tmp_idx_offset_raw, *d_remap_raw, *d_ro_raw;
	unsigned long long *d_wo_raw, *d_tmp_wo_raw;
	d_idx_offset_raw = (unsigned int*)buf_pool;
	d_tmp_idx_offset_raw = d_idx_offset_raw + MAX_IO_SIZE/sizeof(unsigned int);

	d_wo_raw = (unsigned long long*)(d_tmp_idx_offset_raw + MAX_IO_SIZE/sizeof(unsigned long long));
	d_tmp_wo_raw = 	(unsigned long long*)(d_wo_raw + MAX_WO_SIZE/sizeof(unsigned long long));

	d_remap_raw = (unsigned int*)(d_tmp_wo_raw + MAX_WO_SIZE/sizeof(unsigned long long));
	d_ro_raw = d_remap_raw + MAX_REMAP_SIZE/sizeof(unsigned long long);

//initial idx_offset
	unsigned int *init_idx_offset = (unsigned int*) malloc(MAX_IO_SIZE);
	init_idx_offset[0] = 1;
	init_idx_offset[1] = 0;
	unsigned long long *init_wo_raw = (unsigned long long*) malloc(MAX_IO_SIZE);
	init_wo_raw[0]=1;
	init_wo_raw[1] = 0; //kernel will consider the global header
	CUDA_CHECK_RETURN(cudaMemcpy(d_idx_offset_raw, init_idx_offset, 2* sizeof(unsigned int), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(d_wo_raw, init_wo_raw, 2 * sizeof(unsigned long long), cudaMemcpyHostToDevice));
	d_tab_in_offset =  d_wo_raw;
	d_tab_out_offset = d_tmp_wo_raw;
	unsigned int *d_bid_offset_raw =  d_idx_offset_raw;
	unsigned int *d_bid_offset_next_raw =  d_tmp_idx_offset_raw;
	unsigned int *d_write_offset_remap =  d_remap_raw;
	*global_num_fi = cst->cnt;
	void *d_res;
	unsigned long long *d_pat;
	unsigned int *d_freq;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_res, MAX_PAT_SIZE+ MAX_FREQ_SIZE));
	d_pat = (unsigned long long*)d_res;
	d_freq = (unsigned int*)((uintptr_t)d_pat + MAX_PAT_SIZE);
	CUDA_CHECK_RETURN(cudaMemset(d_pat, 0, sizeof(unsigned long long)));

	int num_wo;
	int k=1;

	size_t old_global_num_fi = 0, new_global_num_fi = *global_num_fi;
	size_t old_tab_in_size = 0;
	void *d_old_tab_in=0;
	do{
		printf("== %d-item set==\n",k++);
		printf("kernel_cal_offset\n");
		unsigned int *d_wide_tab_id;
		if(new_global_num_fi > old_global_num_fi){
			CUDA_CHECK_RETURN(cudaMalloc((void**) &d_wide_tab_id, new_global_num_fi * sizeof(unsigned int)));
			//printf("new d_wide_tab_id:%llu\n",new_global_num_fi);
			old_global_num_fi = new_global_num_fi;
		}else{
			//printf("reuse d_wide_tab_id:%llu\n",old_global_num_fi);
		}

		CUDA_CHECK_RETURN(cudaMemset(d_write_offset_remap, 0, sizeof(unsigned long long)));
		CUDA_CHECK_RETURN(cudaMemset(d_bid_offset_next_raw, 0, sizeof(unsigned long long)));
		CUDA_CHECK_RETURN(cudaMemset(d_tab_out_offset, 0, sizeof(unsigned int)));

		kernel_cal_offset<<<ceil(new_global_num_fi/128.0),128>>>((unsigned int*)d_tab_in,
				d_bid_offset_raw, d_bid_offset_next_raw,
				d_tab_in_offset, d_tab_out_offset,
				d_write_offset_remap, smin, cst->cnt, d_wide_tab_id);



		CUDA_CHECK_RETURN(cudaMemcpy(&num_wo, d_tab_out_offset, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		//printf("#tab in next run : %u(size:%uKB)\n",num_wo, (num_wo*sizeof(unsigned long long))>>10);
		get_num_next_tabs(d_tab_out_offset, num_wo, &next_tab_size);
		printf("next_tab_size in next run : %lluMB\n",next_tab_size>>20);
		if(num_wo){
			//final_next_write_offset(d_wo_raw+1, d_tmp_wo_raw+1, num_wo);
			final_next_write_offset(d_tab_out_offset+1, d_tab_out_offset+1, num_wo);

			//get_num_next_tabs(d_tmp_wo_raw, num_wo, &next_tab_size);
			next_tab_size += sizeof(unsigned int)*2;
			// count size of the next table and fill the tab_out_offset
			//CUDA_CHECK_RETURN(cudaMallocHost((void**) &tab_out, next_tab_size));
			if(next_tab_size>old_tab_in_size){
				if(d_old_tab_in){
					//printf("free d_old_tab_in:%p\n",d_old_tab_in);
					CUDA_CHECK_RETURN(cudaFree(d_old_tab_in));
				}
				CUDA_CHECK_RETURN(cudaMalloc((void**) &d_tab_out, next_tab_size));
			//	printf("d_tab_in:0x%p new d_tab_out:%p(%lluMB)\n",d_tab_in, d_tab_out, next_tab_size>>20);

			}else{

				d_tab_out = d_old_tab_in;
			//	printf("d_tab_in:0x%p reuse d_tab_out = d_old_tab_in:%p(%lluMB)\n",d_tab_in, d_tab_out, old_tab_in_size>>20);
			}

	//		printf("num_wo=%u next_tab_size=%u(%p~%p)\n",num_wo, next_tab_size, d_tab_out, (uintptr_t)d_tab_out + next_tab_size);
			//CUDA_CHECK_RETURN(cudaMemset(d_tab_out, 0, next_tab_size));

			CUDA_CHECK_RETURN(cudaMemset(d_tab_out, 0, 8));// clear global counters for all blocks, it is for support initializing tables by each block's self

		}
		printf("kernel_fpg_iter\n");

		kernel_fpg_iter_gtree<<<new_global_num_fi,512>>>(d_tab_in,d_tab_out,d_tab_in_offset,d_tab_out_offset,
						cst->d_gtree, smin, d_bid_offset_raw, d_write_offset_remap, d_pat, d_freq, cst->cnt, d_wide_tab_id);
		//printf("%s\n",cudaGetErrorString(cudaGetLastError()));
		if(!num_wo)
			break;
//		CUDA_CHECK_RETURN(cudaMemcpy(tab_out, d_tab_out, next_tab_size, cudaMemcpyDeviceToHost));//for debug

		CUDA_CHECK_RETURN(cudaMemcpy(global_num_fi, d_tab_out, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		printf("global_num_fi=%u\n",*global_num_fi);
		new_global_num_fi = *global_num_fi;
		void *d_ptmp;
		//swap input and output tab
//		CUDA_CHECK_RETURN(cudaFree(d_tab_in));
		if(new_global_num_fi> old_global_num_fi)
			CUDA_CHECK_RETURN(cudaFree(d_wide_tab_id));
		d_old_tab_in = d_tab_in;
		old_tab_in_size = tab_size;

		d_tab_in = d_tab_out;
		tab_size = next_tab_size;

		fianl_next_index_offset(d_bid_offset_next_raw+1,d_bid_offset_next_raw+1,num_wo);
		d_ptmp =  d_bid_offset_raw;
		d_bid_offset_raw = d_bid_offset_next_raw;
		d_bid_offset_next_raw = (unsigned int*)d_ptmp;
		//swap WO buf
		d_ptmp =  d_tab_in_offset;
		d_tab_in_offset = d_tab_out_offset;
		d_tab_out_offset = (unsigned long long*)d_ptmp;

	}while(num_wo);
	/*
	unsigned int *tab_out;
	CUDA_CHECK_RETURN(cudaMallocHost((void**) &tab_out, next_tab_size));
	CUDA_CHECK_RETURN(cudaMemcpy(tab_out, d_tab_out, next_tab_size, cudaMemcpyDeviceToHost));
	 */
	//cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(cudaMallocHost((void**) &h_res,  sizeof(unsigned long long)));
	CUDA_CHECK_RETURN(cudaMemcpy(h_res, d_res, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	unsigned long long *h_pat = (unsigned long long*)h_res;
	printf("CUDA #pat = %llu\n",h_pat[0]);
	return (long) h_res;
}


void free_gpu_mem()
{

}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
