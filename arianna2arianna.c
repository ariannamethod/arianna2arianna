/* arianna2arianna
 *
 * One self-contained C organism: GGUF parser + byte-level BPE + Llama/Qwen forward
 * + sampler, ALL inlined. No external -lnotorch, no Metal dependency. Vendored
 * faithfully from notorch (gguf.{c,h}, examples/infer_gguf_metal.c, examples/bpe.{c,h}).
 * CPU base; packed-Q4_K / Metal is an optional optimization for a later #ifdef.
 *
 *   theta = epsilon + gamma + alpha*delta
 *   epsilon = one shared nanoArianna body (this forward over a GGUF, weights shared read-only)
 *   gamma   = Arianna's voice (SFT, baked into the weights)
 *   delta   = the field of ephemeral transformer-cells (NEXT layer — scaffold at bottom)
 *
 * MVP-0 (this build): nanoArianna speaks, standalone.
 *   cc -O2 arianna2arianna.c -lm -o arianna2arianna
 *   ./arianna2arianna <model.gguf> [prompt] [max_tokens] [temp]

 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define Q_NEON 1
#endif

/* ===================== GGUF parser (vendored: notorch/gguf.{c,h}) ===================== */

#define GGUF_MAGIC       0x46554747
#define GGUF_TYPE_F32    0
#define GGUF_TYPE_F16    1
#define GGUF_TYPE_Q4_0   2
#define GGUF_TYPE_Q5_0   6
#define GGUF_TYPE_Q8_0   8
#define GGUF_TYPE_Q4_K   12
#define GGUF_TYPE_Q6_K   14
#define GGUF_MAX_TENSORS 2048
#define GGUF_MAX_NAME    128
#define GGUF_MAX_KV      128

typedef struct { char name[GGUF_MAX_NAME]; uint32_t ndim; uint64_t shape[4]; uint32_t dtype; uint64_t offset; uint64_t n_elements; } gguf_tensor_info;
typedef struct { char key[GGUF_MAX_NAME]; uint32_t type; union { uint32_t u32; int32_t i32; float f32; uint8_t b; char str[256]; uint64_t u64; } val; } gguf_kv;
typedef struct {
    uint32_t version; uint64_t n_tensors, n_kv;
    gguf_kv kv[GGUF_MAX_KV]; int n_kv_parsed;
    gguf_tensor_info tensors[GGUF_MAX_TENSORS];
    uint8_t *data; uint64_t data_offset, data_size;
    int n_layers, n_heads, n_kv_heads, embed_dim, ffn_dim, vocab_size, ctx_len;
    float rope_freq_base, rms_eps; char arch[64];
} gguf_file;

static int read_u32(FILE* f, uint32_t* v) { return fread(v, 4, 1, f) == 1; }
static int read_u64(FILE* f, uint64_t* v) { return fread(v, 8, 1, f) == 1; }
static int read_f32(FILE* f, float* v)    { return fread(v, 4, 1, f) == 1; }

