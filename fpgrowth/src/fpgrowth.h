/*----------------------------------------------------------------------
  File    : fpgrowth.h
  Contents: fpgrowth algorithm for finding frequent item sets
  Author  : Christian Borgelt
  History : 2011.08.22 file created
            2011.09.21 available variants and modes reorganized
            2014.08.07 association rule generation/evaluation added
            2014.08.19 adapted to modified item set reporter interface
            2014.08.21 parameter 'body' added to function fpgrowth()
            2014.08.28 functions fpg_data() and fpg_repo() added
----------------------------------------------------------------------*/
#ifndef __FPGROWTH__
#define __FPGROWTH__
#include "tract.h"
#ifndef ISR_CLOMAX
#define ISR_CLOMAX
#endif
#include "report.h"
#include "ruleval.h"
#include "istree.h"

/*----------------------------------------------------------------------
  Preprocessor Definitions
----------------------------------------------------------------------*/
/* --- evaluation measures --- */
/* most definitions in ruleval.h */
#define FPG_LDRATIO RE_FNCNT    /* binary log. of support quotient */
#define FPG_INVBXS  IST_INVBXS  /* inval. eval. below exp. supp. */

/* --- aggregation modes --- */
#define FPG_NONE    IST_NONE    /* no aggregation (use first value) */
#define FPG_FIRST   IST_FIRST   /* no aggregation (use first value) */
#define FPG_MIN     IST_MIN     /* minimum of measure values */
#define FPG_MAX     IST_MAX     /* maximum of measure values */
#define FPG_AVG     IST_AVG     /* average of measure values */

/* --- algorithm variants --- */
#define FPG_SIMPLE  0           /* simple  nodes (parent/link) */
#define FPG_COMPLEX 1           /* complex nodes (children/sibling) */
#define FPG_SINGLE  2           /* top-down processing on single tree */
#define FPG_TOPDOWN 3           /* top-down processing of the tree */

/* --- operation modes --- */
#define FPG_FIM16   0x001f      /* use 16 items machine (bit rep.) */
#define FPG_PERFECT 0x0020      /* perfect extension pruning */
#define FPG_REORDER 0x0040      /* reorder items in cond. databases */
#define FPG_TAIL    0x0080      /* head union tail pruning */
#define FPG_DEFAULT (FPG_PERFECT|FPG_REORDER|FPG_TAIL|FPG_FIM16)
#ifdef NDEBUG
#define FPG_NOCLEAN 0x8000      /* do not clean up memory */
#else                           /* in function fpgrowth() */
#define FPG_NOCLEAN 0           /* in debug version */
#endif                          /* always clean up memory */
#define FPG_VERBOSE INT_MIN     /* verbose message output */
/*
#define GPUTREE_ITEM_BITS	7
#define GPUTREE_INDEX_BITS	12
#define GPUTREE_FREQ_BITS	13
struct GPU_TREE_NODE{
	unsigned int pitem:GPUTREE_ITEM_BITS;
	unsigned int index:GPUTREE_INDEX_BITS;
	unsigned int freq:GPUTREE_FREQ_BITS;

};
*/
//64bit

#define GPUTREE_ITEM_BITS	13
#define GPUTREE_INDEX_BITS	19
#define GPUTREE_FREQ_BITS	32
struct GPU_TREE_NODE{
	unsigned long long pitem:GPUTREE_ITEM_BITS;
	unsigned long long index:GPUTREE_INDEX_BITS;
	unsigned long long freq:GPUTREE_FREQ_BITS;

};



//#define ROOT_ITEM (-1)
#define ROOT_ITEM (1<<GPUTREE_ITEM_BITS -1)
#define MAX_FI (2200)
#define MAX_NUM_PAT_PER_ITEM 200000
#define MAX_NUM_PAT_PER_PATH	(1500000)
#define GPU_MAX_PATTERN_SIZE_PER_ITEM (200<<20)
#define  MAX_NUM_PATTERN_PER_ITEM GPU_MAX_PATTERN_SIZE_PER_ITEM

typedef struct csnode {         /* --- children/sibling tree node --- */
  ITEM          id;             /* item/head identifier */
  SUPP          supp;           /* support (weight of transactions) */
  unsigned int gnidx;
  struct csnode *children;      /* list of child nodes */
  struct csnode *sibling;       /* successor node in sibling list */
  struct csnode *parent;        /* parent node (preceding item) */
  struct csnode *succ;          /* successor node with same item */
  struct csnode *lv_link;
  int lv;
  struct csnode *right_leaf;		/*chain all leaf*/
  struct csnode *left_leaf;
} CSNODE;                       /* (children/sibling tree node) */

typedef struct {                /* --- ch./sibling tree node list --- */
	int cnt;
	int max_depth;
  ITEM     item;                /* associated item (item base code) */
  SUPP     supp;                /* support (weight of transactions) */
  CSNODE   *list;               /* list of nodes with this item */
} CSHEAD;                       /* (children/sibling tree head) */


typedef struct {                /* --- ch./sibling tree node list --- */
	int cnt;
	CSNODE   *list;               /* list of nodes with this item */
} CSLVHEAD;                       /* (children/sibling tree head) */

typedef struct _pattern{
    unsigned long long *bmap;
    int len;
    int freq;
    int oneitem;// > 0: It's 1-item FIS and it is followed by oneitem FIS, =0: it's no an 1-item FIS
}pattern;



typedef struct GPU_TREE_NODE GTREE_NODE;
//designe for GPU , SoA
typedef struct{
	int *frequency;
	ITEM *item;
	int *lv;
	int *parent;
} CLTREE;

typedef CLTREE CLNODE;

typedef struct {                /* --- children/sibling tree --- */
  CSLVHEAD *lvhead;
  int *cltree_index_pos;
  int *cltree_item_index;
  int cltree_idx_struct_size;
  int *level_clen;
  CLTREE cl_tree;
  CSNODE **cl_tree_aux;
  //gputree_itembase and gputree are allocated together
  unsigned int *d_gtree_itembase, *h_gtree_itembase;
  GTREE_NODE *d_gtree, *h_gtree;

  unsigned int gtree_size;
  int *fparray;
  int cnt_cfpb_leaf;
  ITEM     cnt;                 /* number of items / heads */
  unsigned int num_fpnode;
  unsigned int max_depth;
  unsigned int real_max_depth;
  pattern *item_bmap;
  unsigned long long* item_gpu_bmap;
  MEMSYS   *mem;                /* memory system for the nodes */
  CSNODE   root;                /* root node connecting trees */
  CSHEAD   heads[1];            /* header table (item lists) */
} CSTREE;                       /* (children/sibling tree) */





typedef struct pt_node {         /* --- children/sibling tree node --- */
  unsigned long long int v;             /* a vector of a pattern*/
  SUPP          supp;           /* support (weight of transactions) */
  struct pt_node *children;      /* list of child nodes */
  struct pt_node *sibling;       /* successor node in sibling list */
  struct pt_node *parent;        /* parent node (preceding item) */
  struct pt_node *succ;          /* successor node with same item */
  struct pt_node *right_leaf;		/*chain all leaf*/
  struct pt_node *left_leaf;
} PT_NODE;                       /* (children/sibling tree node) */


typedef struct {                /* --- children/sibling tree --- */
  int *cltree_index_pos;
  int *cltree_item_index;
  int cltree_idx_struct_size;
  int *level_clen;
  unsigned int num_node;
  unsigned int max_depth;
  unsigned int real_max_depth;
  pattern *item_bmap;
  MEMSYS   *mem;                /* memory system for the nodes */
  PT_NODE   root;                /* root node connecting trees */
} PT_TREE;                       /* (children/sibling tree) */

typedef struct _entey_head{
	unsigned int item;
	unsigned int supp;
	unsigned int num_node;
} entry_head;

//pattern *gen_pattern(CSTREE* , ITEM , const int *, const pattern *,  const int, int*);
PT_TREE* gen_pattern(CSTREE* , ITEM , const int *, const pattern *,  const int, int*);
CSTREE * prefix_merge_by_item(int *, CSTREE *, ITEM , int**, int*);
//PT_TREE* prefix_pt_merge_by_item(ITEM, const pattern *, int );
void prefix_pt_merge_by_item(PT_TREE*, ITEM, const pattern *, int );

unsigned long long* gpu_gen_bmap(int);
void free_gpu_mem_starge1();
/*----------------------------------------------------------------------
  Functions
----------------------------------------------------------------------*/
extern int fpg_data (TABAG *tabag, int target, SUPP smin, ITEM zmin,
                     int eval, int algo, int mode, int sort);
extern int fpg_repo (ISREPORT *report, int target,
                     int eval, double thresh, int algo, int mode);
extern int fpgrowth (TABAG *tabag, int target, SUPP smin, SUPP body,
                     double conf, int eval, int agg, double thresh,
                     ITEM prune, int algo, int mode,
                     int order, ISREPORT *report);
#endif
