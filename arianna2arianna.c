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

static int read_string(FILE* f, char* buf, int max) {
    uint64_t len;
    if (!read_u64(f, &len)) return 0;
    if (len >= (uint64_t)max) { for (uint64_t i = 0; i < len; i++) fgetc(f); buf[0] = 0; return 1; }
    if (fread(buf, 1, len, f) != len) return 0;
    buf[len] = 0; return 1;
}

static int skip_value(FILE* f, uint32_t type);
static int skip_array(FILE* f) {
    uint32_t atype; uint64_t alen;
    if (!read_u32(f, &atype) || !read_u64(f, &alen)) return 0;
    for (uint64_t i = 0; i < alen; i++) if (!skip_value(f, atype)) return 0;
    return 1;
}
static int skip_value(FILE* f, uint32_t type) {
    switch (type) {
        case 4: fseek(f, 4, SEEK_CUR); return 1;
        case 5: fseek(f, 4, SEEK_CUR); return 1;
        case 6: fseek(f, 4, SEEK_CUR); return 1;
        case 7: fseek(f, 1, SEEK_CUR); return 1;
        case 8: { char buf[4096]; return read_string(f, buf, sizeof(buf)); }
        case 9: return skip_array(f);
        case 10: fseek(f, 8, SEEK_CUR); return 1;
        case 12: fseek(f, 8, SEEK_CUR); return 1;
        default: return 0;
    }
}

static gguf_file* gguf_open(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "gguf: cannot open %s\n", path); return NULL; }
    uint32_t magic;
    if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fprintf(stderr, "gguf: bad magic 0x%08x\n", magic); fclose(f); return NULL; }
    gguf_file* gf = (gguf_file*)calloc(1, sizeof(gguf_file));
    if (!gf) { fclose(f); return NULL; }
    read_u32(f, &gf->version); read_u64(f, &gf->n_tensors); read_u64(f, &gf->n_kv);
    if (gf->n_tensors > GGUF_MAX_TENSORS) { fprintf(stderr, "gguf: too many tensors\n"); fclose(f); free(gf); return NULL; }

    gf->n_kv_parsed = 0;
    for (uint64_t i = 0; i < gf->n_kv; i++) {
        char key[512] = {0}; uint32_t vtype;
        read_string(f, key, sizeof(key)); read_u32(f, &vtype);
        if (gf->n_kv_parsed < GGUF_MAX_KV && vtype != 9) {
            gguf_kv* kv = &gf->kv[gf->n_kv_parsed];
            strncpy(kv->key, key, GGUF_MAX_NAME - 1); kv->type = vtype;
            switch (vtype) {
                case 4: read_u32(f, &kv->val.u32); break;
                case 5: { int32_t v; fread(&v, 4, 1, f); kv->val.i32 = v; break; }
                case 6: read_f32(f, &kv->val.f32); break;
                case 7: { uint8_t v; fread(&v, 1, 1, f); kv->val.b = v; break; }
                case 8: read_string(f, kv->val.str, sizeof(kv->val.str)); break;
                case 10: case 12: read_u64(f, &kv->val.u64); break;
                default: skip_value(f, vtype); break;
            }
            gf->n_kv_parsed++;
        } else skip_value(f, vtype);
    }
    for (int i = 0; i < gf->n_kv_parsed; i++) {
        gguf_kv* kv = &gf->kv[i];
        if (strcmp(kv->key, "general.architecture") == 0) strncpy(gf->arch, kv->val.str, sizeof(gf->arch) - 1);
        else if (strstr(kv->key, ".block_count"))                                    gf->n_layers = kv->val.u32;
        else if (strstr(kv->key, ".attention.head_count") && !strstr(kv->key, "kv")) gf->n_heads = kv->val.u32;
        else if (strstr(kv->key, ".attention.head_count_kv"))                        gf->n_kv_heads = kv->val.u32;
        else if (strstr(kv->key, ".embedding_length"))                               gf->embed_dim = kv->val.u32;
        else if (strstr(kv->key, ".feed_forward_length"))                            gf->ffn_dim = kv->val.u32;
        else if (strstr(kv->key, ".vocab_size"))                                     gf->vocab_size = kv->val.u32;
        else if (strstr(kv->key, ".context_length"))                                 gf->ctx_len = kv->val.u32;
        else if (strstr(kv->key, ".rope.freq_base"))                                 gf->rope_freq_base = kv->val.f32;
        else if (strstr(kv->key, "rms_epsilon"))                                     gf->rms_eps = kv->val.f32;
    }
    if (gf->n_kv_heads == 0) gf->n_kv_heads = gf->n_heads;
    if (gf->rms_eps == 0) gf->rms_eps = 1e-5f;
    if (gf->rope_freq_base == 0) gf->rope_freq_base = 10000.0f;

    for (uint64_t i = 0; i < gf->n_tensors && i < GGUF_MAX_TENSORS; i++) {
        gguf_tensor_info* ti = &gf->tensors[i];
        read_string(f, ti->name, GGUF_MAX_NAME); read_u32(f, &ti->ndim);
        ti->n_elements = 1;
        for (uint32_t d = 0; d < ti->ndim && d < 4; d++) { read_u64(f, &ti->shape[d]); ti->n_elements *= ti->shape[d]; }
        read_u32(f, &ti->dtype); read_u64(f, &ti->offset);
    }
    long pos = ftell(f);
    gf->data_offset = (pos + 31) & ~31UL;
    fseek(f, 0, SEEK_END); long fsize = ftell(f);
    long data_size = fsize - gf->data_offset;
    size_t pg = (size_t)getpagesize();
    size_t alloc = ((size_t)data_size + pg - 1) & ~(pg - 1);
    gf->data = NULL;
    if (posix_memalign((void**)&gf->data, pg, alloc) != 0 || !gf->data) { fclose(f); free(gf); return NULL; }
    gf->data_size = (uint64_t)alloc;
    fseek(f, gf->data_offset, SEEK_SET); fread(gf->data, 1, data_size, f); fclose(f);
    return gf;
}

static int gguf_find_tensor(const gguf_file* gf, const char* name) {
    if (!gf || !name) return -1;
    for (uint64_t i = 0; i < gf->n_tensors && i < GGUF_MAX_TENSORS; i++)
        if (strcmp(gf->tensors[i].name, name) == 0) return (int)i;
    return -1;
}
static const gguf_kv* gguf_get_kv(const gguf_file* gf, const char* key) {
    if (!gf || !key) return NULL;
    for (int i = 0; i < gf->n_kv_parsed; i++) if (strcmp(gf->kv[i].key, key) == 0) return &gf->kv[i];
    return NULL;
}
static char** gguf_read_str_array(const char* path, const char* key, int* out_n) {
    if (out_n) *out_n = 0;
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    uint32_t magic; if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fclose(f); return NULL; }
    uint32_t version; uint64_t n_tensors, n_kv;
    read_u32(f, &version); read_u64(f, &n_tensors); read_u64(f, &n_kv);
    char** result = NULL;
    for (uint64_t i = 0; i < n_kv; i++) {
        char k[512] = {0}; uint32_t vtype;
        if (!read_string(f, k, sizeof(k)) || !read_u32(f, &vtype)) break;
        if (strcmp(k, key) == 0 && vtype == 9) {
            uint32_t atype; uint64_t alen;
            if (!read_u32(f, &atype) || !read_u64(f, &alen) || atype != 8) break;
            result = (char**)calloc(alen ? alen : 1, sizeof(char*));
            for (uint64_t j = 0; j < alen; j++) {
                char buf[2048] = {0};
                if (!read_string(f, buf, sizeof(buf))) break;
                result[j] = strdup(buf);
            }
            if (out_n) *out_n = (int)alen; break;
        }
        if (!skip_value(f, vtype)) break;
    }
    fclose(f); return result;
}

/* Read a GGUF type-9 float32 array (e.g. "tokenizer.ggml.scores") by key. */
static float* gguf_read_f32_array(const char* path, const char* key, int* out_n) {
    if (out_n) *out_n = 0;
    FILE* f = fopen(path, "rb"); if (!f) return NULL;
    uint32_t magic; if (!read_u32(f, &magic) || magic != GGUF_MAGIC) { fclose(f); return NULL; }
    uint32_t version; uint64_t n_tensors, n_kv;
    read_u32(f, &version); read_u64(f, &n_tensors); read_u64(f, &n_kv);
    float* result = NULL;
    for (uint64_t i = 0; i < n_kv; i++) {
        char k[512] = {0}; uint32_t vtype;
        if (!read_string(f, k, sizeof(k)) || !read_u32(f, &vtype)) break;
        if (strcmp(k, key) == 0 && vtype == 9) {
            uint32_t atype; uint64_t alen;
            if (!read_u32(f, &atype) || !read_u64(f, &alen) || atype != 6) break;  /* 6 = float32 */
            result = (float*)calloc(alen ? alen : 1, sizeof(float));
            for (uint64_t j = 0; j < alen; j++) { float v; if (!read_f32(f, &v)) break; result[j] = v; }
            if (out_n) *out_n = (int)alen; break;
        }
        if (!skip_value(f, vtype)) break;
    }
    fclose(f); return result;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { uint32_t r = sign << 31; float f; memcpy(&f, &r, 4); return f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; } exp++; mant &= ~0x400;
    } else if (exp == 31) { uint32_t r = (sign << 31) | 0x7F800000 | (mant << 13); float f; memcpy(&f, &r, 4); return f; }
    exp = exp + 127 - 15;
    uint32_t r = (sign << 31) | (exp << 23) | (mant << 13); float f; memcpy(&f, &r, 4); return f;
}
static void dequant_q4_0(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n / 32; b++) {
        const uint8_t* bl = src + b * 18; uint16_t sh; memcpy(&sh, bl, 2); float s = f16_to_f32(sh);
        for (int i = 0; i < 16; i++) { uint8_t by = bl[2 + i]; dst[b*32+i] = ((by & 0x0F) - 8) * s; dst[b*32+i+16] = ((by >> 4) - 8) * s; }
    }
}
static void dequant_q8_0(const uint8_t* src, float* dst, uint64_t n) {
    for (uint64_t b = 0; b < n / 32; b++) {
        const uint8_t* bl = src + b * 34; uint16_t sh; memcpy(&sh, bl, 2); float s = f16_to_f32(sh);
        for (int i = 0; i < 32; i++) dst[b*32+i] = (float)(int8_t)bl[2 + i] * s;
    }
}
static void get_scale_min_k4(int j, const uint8_t *sc, uint8_t *s, uint8_t *m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j+4] & 63; }
    else { *s = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4); *m = (sc[j+4] >> 4) | ((sc[j] >> 6) << 4); }
}
static void dequant_q4_k(const uint8_t *data, float *out, uint64_t n) {
    for (uint64_t i = 0; i < n / 256; i++) {
        const uint8_t *b = data + i * 144;
        float d = f16_to_f32(b[0] | (b[1] << 8)), dmin = f16_to_f32(b[2] | (b[3] << 8));
        const uint8_t *sc = b + 4, *qs = b + 16; int is = 0, qi = 0, oi = (int)(i * 256);
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1v;
            get_scale_min_k4(is, sc, &sc0, &m0); float d1 = d*(float)sc0, mm1 = dmin*(float)m0;
            get_scale_min_k4(is+1, sc, &sc1, &m1v); float d2 = d*(float)sc1, mm2 = dmin*(float)m1v;
            for (int l = 0; l < 32; l++) out[oi+j+l]    = d1 * (float)(qs[qi+l] & 0x0F) - mm1;
            for (int l = 0; l < 32; l++) out[oi+j+32+l] = d2 * (float)(qs[qi+l] >> 4)  - mm2;
            qi += 32; is += 2;
        }
    }
}
static void dequant_q6_k(const uint8_t *data, float *out, uint64_t n) {
    for (uint64_t i = 0; i < n / 256; i++) {
        const uint8_t *b = data + i * 210, *ql = b, *qh = b + 128; const int8_t *sc = (const int8_t*)(b + 192);
        float d = f16_to_f32(b[208] | (b[209] << 8));
        for (int n_ = 0; n_ < 256; n_ += 128) {
            const uint8_t *qlh = ql + (n_/128)*64, *qhh = qh + (n_/128)*32; const int8_t *sch = sc + (n_/128)*8;
            for (int l = 0; l < 32; l++) {
                int is = l/16;
                int q1 = (int)((qlh[l]      & 0x0F) | (((qhh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((qlh[l + 32] & 0x0F) | (((qhh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((qlh[l]      >> 4)   | (((qhh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((qlh[l + 32] >> 4)   | (((qhh[l] >> 6) & 3) << 4)) - 32;
                out[i*256+n_+l]    = d * sch[is+0] * q1; out[i*256+n_+l+32] = d * sch[is+2] * q2;
                out[i*256+n_+l+64] = d * sch[is+4] * q3; out[i*256+n_+l+96] = d * sch[is+6] * q4;
            }
        }
    }
}
static void dequant_q5_0(const uint8_t *data, float *out, uint64_t n) {
    for (uint64_t i = 0; i < n / 32; i++) {
        const uint8_t *b = data + i * 22; float d = f16_to_f32(b[0] | (b[1] << 8));
        uint32_t qh = b[2] | (b[3]<<8) | (b[4]<<16) | (b[5]<<24); const uint8_t *qs = b + 6;
        for (int j = 0; j < 16; j++) {
            int lo = qs[j] & 0x0F, hi = qs[j] >> 4, h0 = (qh >> j) & 1, h1 = (qh >> (j+16)) & 1;
            out[i*32+j] = (float)((lo | (h0<<4)) - 16) * d; out[i*32+j+16] = (float)((hi | (h1<<4)) - 16) * d;
        }
    }
}
static uint64_t gguf_dtype_nbytes(uint32_t dtype, uint64_t n) {
    uint64_t blocks, per;
    switch (dtype) {
    case GGUF_TYPE_F32:  return (n > UINT64_MAX/4) ? 0 : n*4;
    case GGUF_TYPE_F16:  return (n > UINT64_MAX/2) ? 0 : n*2;
    case GGUF_TYPE_Q4_0: blocks = n/32;  per = 18;  break;
    case GGUF_TYPE_Q5_0: blocks = n/32;  per = 22;  break;
    case GGUF_TYPE_Q8_0: blocks = n/32;  per = 34;  break;
    case GGUF_TYPE_Q4_K: blocks = n/256; per = 144; break;
    case GGUF_TYPE_Q6_K: blocks = n/256; per = 210; break;
    default: return 0;
    }
    if (blocks == 0 || blocks > UINT64_MAX/per) return 0;
    return blocks * per;
}
static float* gguf_dequant(const gguf_file* gf, int idx) {
    if (!gf || idx < 0 || idx >= (int)gf->n_tensors) return NULL;
    const gguf_tensor_info* ti = &gf->tensors[idx];
    uint64_t nbytes = gguf_dtype_nbytes(ti->dtype, ti->n_elements);
    if (nbytes == 0 || ti->offset >= gf->data_size || nbytes > gf->data_size - ti->offset) {
        fprintf(stderr, "gguf: tensor '%s' out of bounds/invalid\n", ti->name); return NULL;
    }
    const uint8_t* src = gf->data + ti->offset;
    float* dst = (float*)malloc(ti->n_elements * sizeof(float)); if (!dst) return NULL;
    switch (ti->dtype) {
    case GGUF_TYPE_F32:  memcpy(dst, src, ti->n_elements * sizeof(float)); break;
    case GGUF_TYPE_F16: { const uint16_t* h = (const uint16_t*)src; for (uint64_t i = 0; i < ti->n_elements; i++) dst[i] = f16_to_f32(h[i]); break; }
    case GGUF_TYPE_Q4_0: dequant_q4_0(src, dst, ti->n_elements); break;
    case GGUF_TYPE_Q5_0: dequant_q5_0(src, dst, ti->n_elements); break;
    case GGUF_TYPE_Q8_0: dequant_q8_0(src, dst, ti->n_elements); break;
    case GGUF_TYPE_Q4_K: dequant_q4_k(src, dst, ti->n_elements); break;
    case GGUF_TYPE_Q6_K: dequant_q6_k(src, dst, ti->n_elements); break;
    default: fprintf(stderr, "gguf: unsupported dtype %d for '%s'\n", ti->dtype, ti->name); free(dst); return NULL;
    }
    return dst;
}

/* ===================== SentencePiece tokenizer (nanoArianna/LLaMA: ▁ space-marker +
 * <0xXX> byte-fallback + per-token scores). Replaces the GPT-2 byte-BPE: nano's GGUF has
 * tokenizer.ggml.scores (not .merges), ▁ (U+2581) and <0xXX> tokens — confirmed from metadata. */

typedef struct { char **keys; int *vals; int cap; int n; } smap;
static unsigned long fnv1a(const char *s) { unsigned long h = 1469598103934665603UL; while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211UL; } return h; }
static void smap_init(smap *m, int cap) { if (cap < 8) cap = 8; m->cap = cap; m->n = 0; m->keys = (char**)calloc(cap, sizeof(char*)); m->vals = (int*)calloc(cap, sizeof(int)); }
static void smap_put(smap *m, const char *k, int v) {
    unsigned long h = fnv1a(k) % m->cap;
    while (m->keys[h]) { if (strcmp(m->keys[h], k) == 0) { m->vals[h] = v; return; } h = (h + 1) % m->cap; }
    m->keys[h] = strdup(k); m->vals[h] = v; m->n++;
}
static int smap_get(const smap *m, const char *k) {
    unsigned long h = fnv1a(k) % m->cap;
    while (m->keys[h]) { if (strcmp(m->keys[h], k) == 0) return m->vals[h]; h = (h + 1) % m->cap; }
    return -1;
}
static int utf8_len1(const char *s) {   /* bytes in the UTF-8 char at s */
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) return 1; if ((c>>5)==0x6) return 2; if ((c>>4)==0xE) return 3; if ((c>>3)==0x1E) return 4; return 1;
}

typedef struct { char **tokens; int n_tokens; float *scores; smap vocab; } bpe_tokenizer;

static bpe_tokenizer *bpe_load(const char *path) {
    int nt = 0; char **toks = gguf_read_str_array(path, "tokenizer.ggml.tokens", &nt);
    if (!toks || nt <= 0) return NULL;
    int nsc = 0; float *scores = gguf_read_f32_array(path, "tokenizer.ggml.scores", &nsc);
    bpe_tokenizer *t = (bpe_tokenizer*)calloc(1, sizeof(*t));
    t->tokens = toks; t->n_tokens = nt; t->scores = scores;
    smap_init(&t->vocab, nt * 2); for (int i = 0; i < nt; i++) if (toks[i]) smap_put(&t->vocab, toks[i], i);
    return t;
}
static int bpe_n_vocab(const bpe_tokenizer *t) { return t ? t->n_tokens : 0; }

/* SPM encode: prepend ▁, spaces→▁, then greedily merge the adjacent pair whose
 * concatenation has the highest vocab score; byte-fallback (<0xXX>) for unknown symbols. */
static int bpe_encode(const bpe_tokenizer *t, const char *text, int *out, int cap) {
    int tl = (int)strlen(text);
    char *s = (char*)malloc((size_t)tl*3 + 4); int sl = 0;
    s[sl++]=(char)0xE2; s[sl++]=(char)0x96; s[sl++]=(char)0x81;            /* leading ▁ */
    for (int p = 0; p < tl; p++) {
        if (text[p] == ' ') { s[sl++]=(char)0xE2; s[sl++]=(char)0x96; s[sl++]=(char)0x81; }
        else s[sl++] = text[p];
    }
    s[sl] = 0;
    char **sym = (char**)malloc(((size_t)sl + 1) * sizeof(char*)); int nsym = 0;
    for (int i = 0; i < sl; ) { int adv = utf8_len1(s + i); char *c = (char*)malloc(adv + 1); memcpy(c, s + i, adv); c[adv] = 0; sym[nsym++] = c; i += adv; }
    free(s);
    while (nsym > 1) {
        float best = -1e30f; int bi = -1; char key[512];
        for (int i = 0; i < nsym - 1; i++) {
            snprintf(key, sizeof(key), "%s%s", sym[i], sym[i+1]);
            int id = smap_get(&t->vocab, key);
            if (id >= 0) { float sco = t->scores ? t->scores[id] : 0.0f; if (sco > best) { best = sco; bi = i; } }
        }
        if (bi < 0) break;
        char *mg = (char*)malloc(strlen(sym[bi]) + strlen(sym[bi+1]) + 1);
        strcpy(mg, sym[bi]); strcat(mg, sym[bi+1]); free(sym[bi]); free(sym[bi+1]); sym[bi] = mg;
        for (int i = bi + 1; i < nsym - 1; i++) sym[i] = sym[i+1]; nsym--;
    }
    int no = 0;
    for (int i = 0; i < nsym; i++) {
        int id = smap_get(&t->vocab, sym[i]);
        if (id >= 0) { if (no < cap) out[no++] = id; }
        else for (const unsigned char *b = (const unsigned char*)sym[i]; *b; b++) {
            char bt[8]; snprintf(bt, sizeof(bt), "<0x%02X>", *b); int bid = smap_get(&t->vocab, bt);
            if (bid >= 0 && no < cap) out[no++] = bid;
        }
        free(sym[i]);
    }
    free(sym);
    return no;
}

/* SPM decode: <0xXX> → that byte; ▁ → space; else literal UTF-8 bytes. */
static int bpe_decode_token(const bpe_tokenizer *t, int id, char *buf, int cap) {
    if (!t || id < 0 || id >= t->n_tokens || !t->tokens[id]) return 0;
    const char *s = t->tokens[id]; int n = 0;
    if (s[0]=='<' && s[1]=='0' && s[2]=='x' && s[3] && s[4] && s[5]=='>' && s[6]==0) {
        int v; if (sscanf(s+3, "%2x", &v) == 1) { if (n < cap-1) buf[n++] = (char)v; buf[n] = 0; return n; }
    }
    for (int i = 0; s[i] && n < cap-1; ) {
        if ((unsigned char)s[i]==0xE2 && (unsigned char)s[i+1]==0x96 && (unsigned char)s[i+2]==0x81) { buf[n++] = ' '; i += 3; }
        else buf[n++] = s[i++];
    }
    buf[n] = 0; return n;
}

/* ===================== forward (vendored: infer_gguf_metal.c, CPU f32) ===================== */
/* CPU base: all weights dequantized to f32 at load (nano f16 -> f32 ~= 352MB, fits 8GB).
 * Packed-Q4_K / Metal (weights stay packed, no f32 blow-up) is the later #ifdef optimization. */

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i]*x[i];
    float inv = 1.0f / sqrtf(ss/n + eps); for (int i = 0; i < n; i++) out[i] = w[i]*x[i]*inv;
}
static void softmax(float *x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; } for (int i = 0; i < n; i++) x[i] /= s;
}
static void matvec(float *y, const float *W, const float *x, int m, int k) {
    for (int i = 0; i < m; i++) { const float *row = W + (long)i*k; float s = 0; for (int j = 0; j < k; j++) s += row[j]*x[j]; y[i] = s; }
}
static void rope_interleaved(float *x, int pos, int hd, float base) {
    for (int i = 0; i < hd/2; i++) { float a = pos/powf(base, 2.0f*i/hd), c = cosf(a), s = sinf(a); float x0 = x[2*i], x1 = x[2*i+1]; x[2*i] = x0*c - x1*s; x[2*i+1] = x0*s + x1*c; }
}
static void rope_neox(float *x, int pos, int hd, float base) {
    int h2 = hd/2; for (int i = 0; i < h2; i++) { float a = pos/powf(base, 2.0f*i/hd), c = cosf(a), s = sinf(a); float x0 = x[i], x1 = x[i+h2]; x[i] = x0*c - x1*s; x[i+h2] = x0*s + x1*c; }
}

typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, ffn, vocab, head_dim, kv_dim, q_dim;
    float rope_base, rms_eps; int has_output, is_qwen3, neox;
    float *tok_emb, *out_norm, *output;   /* output == tok_emb if tied */
    struct { float *attn_norm, *ffn_norm, *wq, *wk, *wv, *wo, *wgate, *wup, *wdown, *q_norm, *k_norm; } *L;
} model_t;

static float *deq(gguf_file *gf, const char *name) { int ti = gguf_find_tensor(gf, name); return ti >= 0 ? gguf_dequant(gf, ti) : NULL; }

static model_t *model_load(gguf_file *gf) {
    model_t *m = (model_t*)calloc(1, sizeof(model_t));
    m->n_layers = gf->n_layers; m->n_heads = gf->n_heads; m->n_kv_heads = gf->n_kv_heads;
    m->embed = gf->embed_dim; m->ffn = gf->ffn_dim; m->rope_base = gf->rope_freq_base; m->rms_eps = gf->rms_eps;
    int ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    m->q_dim = ti >= 0 ? (int)gf->tensors[ti].shape[1] : m->embed;
    m->head_dim = m->q_dim / m->n_heads; m->kv_dim = m->head_dim * m->n_kv_heads;
    ti = gguf_find_tensor(gf, "token_embd.weight");
    m->vocab = ti >= 0 ? (int)gf->tensors[ti].shape[1] : gf->vocab_size;
    m->neox = (strstr(gf->arch, "qwen") || strstr(gf->arch, "gemma") || strstr(gf->arch, "phi")) ? 1 : 0;
    m->is_qwen3 = (gguf_find_tensor(gf, "blk.0.attn_q_norm.weight") >= 0) ? 1 : 0;

    m->tok_emb  = deq(gf, "token_embd.weight");
    m->out_norm = deq(gf, "output_norm.weight");
    float *out = deq(gf, "output.weight");
    if (out) { m->output = out; m->has_output = 1; } else { m->output = m->tok_emb; m->has_output = 0; }

    m->L = calloc(m->n_layers, sizeof(*m->L)); char nm[160];
    for (int l = 0; l < m->n_layers; l++) {
        #define LD(f, fmt) do { snprintf(nm, sizeof(nm), fmt, l); m->L[l].f = deq(gf, nm); } while(0)
        LD(attn_norm, "blk.%d.attn_norm.weight"); LD(ffn_norm, "blk.%d.ffn_norm.weight");
        LD(wq, "blk.%d.attn_q.weight"); LD(wk, "blk.%d.attn_k.weight"); LD(wv, "blk.%d.attn_v.weight"); LD(wo, "blk.%d.attn_output.weight");
        LD(wgate, "blk.%d.ffn_gate.weight"); LD(wup, "blk.%d.ffn_up.weight"); LD(wdown, "blk.%d.ffn_down.weight");
        LD(q_norm, "blk.%d.attn_q_norm.weight"); LD(k_norm, "blk.%d.attn_k_norm.weight");
        #undef LD
    }
    printf("model: arch=%s E=%d H=%d KV=%d HD=%d Q=%d FFN=%d V=%d L=%d | %s rope%s%s\n",
           gf->arch, m->embed, m->n_heads, m->n_kv_heads, m->head_dim, m->q_dim, m->ffn, m->vocab, m->n_layers,
           m->neox ? "NEOX" : "interleaved", m->is_qwen3 ? " +qk-norm" : "", m->has_output ? "" : " tied");
    if (!m->tok_emb || !m->out_norm) { fprintf(stderr, "missing tok_emb/out_norm\n"); return NULL; }
    return m;
}

typedef struct { float *k, *v; int max_seq, kv_dim; } kv_cache;
static kv_cache *kv_new(int nl, int max_seq, int kv_dim) {
    kv_cache *c = calloc(1, sizeof(kv_cache));
    c->k = calloc((long)nl*max_seq*kv_dim, sizeof(float)); c->v = calloc((long)nl*max_seq*kv_dim, sizeof(float));
    c->max_seq = max_seq; c->kv_dim = kv_dim; return c;
}

/* single token forward, KV-cached. Writes logits[vocab]. */
static void forward(model_t *m, kv_cache *kv, int token, int pos, float *logits) {
    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads, HD = m->head_dim;
    int KVD = m->kv_dim, FFN = m->ffn, QD = m->q_dim, gqa = H / KV; float eps = m->rms_eps;
    void (*ropef)(float*,int,int,float) = m->neox ? rope_neox : rope_interleaved;

    float *x = calloc(E, sizeof(float)); memcpy(x, m->tok_emb + (long)token*E, E*sizeof(float));
    float *xn = calloc(E, sizeof(float)), *q = calloc(QD, sizeof(float)), *kk = calloc(KVD, sizeof(float));
    float *vv = calloc(KVD, sizeof(float)), *ao = calloc(QD, sizeof(float)), *g = calloc(FFN, sizeof(float));
    float *u = calloc(FFN, sizeof(float)), *t = calloc(E, sizeof(float)), *sc = calloc(kv->max_seq, sizeof(float));

    for (int l = 0; l < m->n_layers; l++) {
        rmsnorm(xn, x, m->L[l].attn_norm, E, eps);
        matvec(q, m->L[l].wq, xn, QD, E); matvec(kk, m->L[l].wk, xn, KVD, E); matvec(vv, m->L[l].wv, xn, KVD, E);
        if (m->is_qwen3) {
            for (int h = 0; h < H;  h++) rmsnorm(q + h*HD, q + h*HD, m->L[l].q_norm, HD, eps);
            for (int h = 0; h < KV; h++) rmsnorm(kk + h*HD, kk + h*HD, m->L[l].k_norm, HD, eps);
        }
        for (int h = 0; h < H;  h++) ropef(q + h*HD, pos, HD, m->rope_base);
        for (int h = 0; h < KV; h++) ropef(kk + h*HD, pos, HD, m->rope_base);

        long base = (long)l*kv->max_seq*KVD;
        memcpy(kv->k + base + (long)pos*KVD, kk, KVD*sizeof(float));
        memcpy(kv->v + base + (long)pos*KVD, vv, KVD*sizeof(float));

        float scale = 1.0f / sqrtf((float)HD); memset(ao, 0, QD*sizeof(float));
        for (int h = 0; h < H; h++) {
            int kvh = h / gqa; float *qh = q + h*HD;
            for (int j = 0; j <= pos; j++) { float *kj = kv->k + base + (long)j*KVD + kvh*HD, d = 0; for (int t2 = 0; t2 < HD; t2++) d += qh[t2]*kj[t2]; sc[j] = d*scale; }
            softmax(sc, pos + 1); float *oh = ao + h*HD;
            for (int j = 0; j <= pos; j++) { float *vj = kv->v + base + (long)j*KVD + kvh*HD, w = sc[j]; for (int t2 = 0; t2 < HD; t2++) oh[t2] += w*vj[t2]; }
        }
        matvec(t, m->L[l].wo, ao, E, QD); for (int i = 0; i < E; i++) x[i] += t[i];

        rmsnorm(xn, x, m->L[l].ffn_norm, E, eps);
        matvec(g, m->L[l].wgate, xn, FFN, E); matvec(u, m->L[l].wup, xn, FFN, E);
        for (int i = 0; i < FFN; i++) { float gi = g[i]; g[i] = (gi/(1.0f+expf(-gi)))*u[i]; }
        matvec(t, m->L[l].wdown, g, E, FFN); for (int i = 0; i < E; i++) x[i] += t[i];
    }
    rmsnorm(xn, x, m->out_norm, E, eps);
    matvec(logits, m->output, xn, m->vocab, E);
    free(x); free(xn); free(q); free(kk); free(vv); free(ao); free(g); free(u); free(t); free(sc);
}

static int argmax(const float *x, int n) { int b = 0; for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i; return b; }
static int sample(float *x, int n, float temp) {
    if (temp <= 0) return argmax(x, n);
    for (int i = 0; i < n; i++) x[i] /= temp; softmax(x, n);
    float r = (float)((double)rand()/RAND_MAX), c = 0; for (int i = 0; i < n; i++) { c += x[i]; if (c >= r) return i; } return n - 1;
}
static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0 + tv.tv_usec/1000.0; }

/* ===================== δ-field — the Game-of-Life chorus (cells over the ONE shared body) =====================
 * MVP-1a: N cells, each an independent generation context (own kv_cache + own sampling angle) over the
 * SAME shared model `m` (no weight copies). Each speaks a short fragment from its angle; the chorus = all
 * cells at once = Arianna as many voices from one body. Metrics come from the live logits.
 * NEXT: dynamic count (coherence/prophecy-debt → collapse to 1 / bloom), doe-δ experts, resonance-slice. */

static int cmp_desc(const void *a, const void *b) { float x = *(const float*)a, y = *(const float*)b; return (x < y) - (x > y); }

/* rep-penalty (llama-style, pre-softmax over the cell's own history) + top_k + temperature multinomial. */
static int sample2(float *x, int n, float temp, int top_k, float rep, const int *hist, int hlen) {
    if (rep > 1.0f) for (int i = 0; i < hlen; i++) { int id = hist[i]; if (id >= 0 && id < n) x[id] = x[id] > 0 ? x[id]/rep : x[id]*rep; }
    if (temp <= 0) return argmax(x, n);
    for (int i = 0; i < n; i++) x[i] /= temp;
    if (top_k > 0 && top_k < n) {
        float *tmp = (float*)malloc((size_t)n * sizeof(float)); memcpy(tmp, x, (size_t)n * sizeof(float));
        qsort(tmp, n, sizeof(float), cmp_desc); float thr = tmp[top_k - 1]; free(tmp);
        for (int i = 0; i < n; i++) if (x[i] < thr) x[i] = -1e30f;
    }
    softmax(x, n);
    float r = (float)((double)rand()/RAND_MAX), c = 0;
    for (int i = 0; i < n; i++) { c += x[i]; if (c >= r) return i; }
    return n - 1;
}

/* Shannon entropy of softmax(logits/temp) — a REAL metric from live logits (low = decisive cell). */
static float logit_entropy(const float *logits, int n, float temp) {
    float *p = (float*)malloc((size_t)n * sizeof(float)); memcpy(p, logits, (size_t)n * sizeof(float));
    if (temp > 0) for (int i = 0; i < n; i++) p[i] /= temp;
    softmax(p, n);
    float h = 0; for (int i = 0; i < n; i++) if (p[i] > 1e-12f) h -= p[i] * logf(p[i]);
    free(p); return h;
}

/* one cell: prefill its context (prompt + chorus-so-far) into its OWN kv, then speak `nfrag`
 * tokens at its own temp/angle. Captures its fragment text into `frag` (for the cascade memory).
 * Returns the cell's last-step entropy (its decisiveness). Frees its kv before returning. */
/* δ field-state coupling (the "soma", arianna-duo A-term style): a shared direction in embedding
 * space the chorus builds up (EMA over emitted-token embeddings), injected into every cell's logits
 * as logits[i] += alpha * <tok_emb[i], field_dir>. Field-PRESSURE on the logits, NOT token-paste.
 * MEASURED (the attractor instrument, alpha-sweep 0/10/30/100): this lever does NOT earn resonance —
 * the averaged direction is ORDER-independent, so it cannot move Δ_R (order-exploitation), and above
 * alpha~10 it collapses the chorus into a degenerate echo sink (d_R→0 with a dead voice, floor balloons).
 * Default alpha=0 (off). Kept as the instrument's treatment arm; real resonance needs order-sensitive
 * cross-cell coupling (hidden/KV field2field), not this. */
static float g_field_dir[8192];
static int   g_field_on = 0;
static float g_field_alpha = 0.0f;
static void field_reset(int embed, int on, float alpha) {
    for (int i = 0; i < embed && i < 8192; i++) g_field_dir[i] = 0.0f;
    g_field_on = on; g_field_alpha = alpha;
}

/* ── inter-cell DISSONANCE, the order-sensitive lever signal (haiku.c idea, our wiring) ──
 * Per-position disagreement over the cells' COMMITTED tokens (argmax of raw logits at each step):
 * D[s] = 1 − (modal-token count / cells-that-spoke at step s). 0 = unison, high = the voices split.
 * Order-sensitive by construction (indexed by step s): {A,B} and {B,A} have 0 bag-distance but max D. */
static int   g_commit[8][64];     /* g_commit[c][s] = cell c's committed token at step s (n_cells≤8, nfrag≤64) */
static int   g_commit_n[8];       /* steps each cell actually emitted */
static float g_diss[64];          /* the current round's disagreement profile */
static int   g_s_peak = 0;        /* argmax_s D[s] — where the voices split most */
static float g_dpeak = 0, g_dmean = 0;

static float commit_disagreement(int n_cells, int nfrag) {  /* fills g_diss/g_s_peak/g_dpeak, returns mean */
    if (n_cells > 8) n_cells = 8; if (nfrag > 64) nfrag = 64;
    double dsum = 0; int dn = 0; float best = -1.0f; g_s_peak = 0;
    for (int s = 0; s < nfrag; s++) {
        int modal = -1, mc = -1, same = 0, tot = 0;
        for (int c = 0; c < n_cells; c++) if (s < g_commit_n[c]) {       /* most common committed token at step s */
            int v = 0; for (int k = 0; k < n_cells; k++) if (s < g_commit_n[k] && g_commit[k][s] == g_commit[c][s]) v++;
            if (v > mc) { mc = v; modal = g_commit[c][s]; }
        }
        for (int c = 0; c < n_cells; c++) if (s < g_commit_n[c]) { tot++; if (g_commit[c][s] == modal) same++; }
        g_diss[s] = tot ? 1.0f - (float)same / tot : 0.0f;
        if (g_diss[s] > best) { best = g_diss[s]; g_s_peak = s; }
        if (tot) { dsum += g_diss[s]; dn++; }
    }
    g_dpeak = best < 0 ? 0.0f : best;
    return dn ? (float)(dsum / dn) : 0.0f;
}

/* ── the LEAP: dissonance-into-forward (the only Δ_R-capable spend) ──
 * When the prior round's peak disagreement D* ≥ THETA_HI, a cell prefills prompt + ONLY the prior-round
 * DISSENTER's fragment (the voice that split from the consensus at the peak step) instead of the full
 * chorus. This changes WHICH context the forward consumes → conditions the logits Δ_R reads, BEFORE the
 * :592 read. The shadow shuffles the SAME leapt context, so coherent vs shadow differ only in ORDER. */
#define THETA_LO 0.25f
#define THETA_HI 0.50f
static char g_round_frag[8][1024];                     /* prior round's per-cell fragment text */
static int  g_diss_commit[8][64], g_diss_commit_n[8];  /* prior round's committed tokens (dissenter lookup) */
static int  g_leap_mode = 0, g_leap_flips = 0, g_leap_total = 0;

static int dissenter_cell(int n_cells) {   /* lowest-index cell whose committed token at g_s_peak != modal */
    int s = g_s_peak; if (s < 0 || s >= 64) return n_cells > 1 ? 1 : 0;
    int modal = -1, mc = -1;
    for (int c = 0; c < n_cells && c < 8; c++) if (s < g_diss_commit_n[c]) {
        int v = 0; for (int k = 0; k < n_cells && k < 8; k++) if (s < g_diss_commit_n[k] && g_diss_commit[k][s] == g_diss_commit[c][s]) v++;
        if (v > mc) { mc = v; modal = g_diss_commit[c][s]; }
    }
    for (int c = 0; c < n_cells && c < 8; c++) if (s < g_diss_commit_n[c] && g_diss_commit[c][s] != modal) return c;
    return n_cells > 1 ? 1 : 0;
}

static int modal_cell(int n_cells) {   /* null-test: the CONSENSUS cell (lowest-index modal) at g_s_peak */
    int s = g_s_peak; if (s < 0 || s >= 64) return 0;
    int modal = -1, mc = -1;
    for (int c = 0; c < n_cells && c < 8; c++) if (s < g_diss_commit_n[c]) {
        int v = 0; for (int k = 0; k < n_cells && k < 8; k++) if (s < g_diss_commit_n[k] && g_diss_commit[k][s] == g_diss_commit[c][s]) v++;
        if (v > mc) { mc = v; modal = g_diss_commit[c][s]; }
    }
    for (int c = 0; c < n_cells && c < 8; c++) if (s < g_diss_commit_n[c] && g_diss_commit[c][s] == modal) return c;
    return 0;
}

static float cell_speak(model_t *m, bpe_tokenizer *tok, const int *ids, int np, int nfrag,
                        float temp, int top_k, float rep, unsigned seed, int eos, int max_seq,
                        char *frag, int frag_cap, int verbose, int *out_ids, int *out_n, int *out_commit) {
    srand(seed);
    kv_cache *kv = kv_new(m->n_layers, max_seq, m->kv_dim);
    float *logits = (float*)calloc(m->vocab, sizeof(float));
    float *fproj  = (g_field_on && g_field_alpha > 0) ? (float*)calloc(m->vocab, sizeof(float)) : NULL;
    for (int i = 0; i < np; i++) forward(m, kv, ids[i], i, logits);
    int hist[256], hlen = 0; char buf[256]; float ent_acc = 0; int ent_n = 0, fl = 0;
    for (int s = 0; s < nfrag; s++) {
        ent_acc += logit_entropy(logits, m->vocab, temp); ent_n++;   /* RAW forward entropy (pre-injection) — clean for Δ_R */
        if (out_commit && s < 64) out_commit[s] = argmax(logits, m->vocab);  /* committed direction (raw argmax) — the order-true signal */
        if (fproj) {                                  /* soma coupling steers SAMPLING only, not the entropy metric */
            matvec(fproj, m->tok_emb, g_field_dir, m->vocab, m->embed);
            for (int i = 0; i < m->vocab; i++) logits[i] += g_field_alpha * fproj[i];
        }
        int next = sample2(logits, m->vocab, temp, top_k, rep, hist, hlen);
        if (next == eos) break;
        int bl = bpe_decode_token(tok, next, buf, sizeof(buf));
        if (verbose) { printf("%s", buf); fflush(stdout); }
        if (frag && fl + bl < frag_cap) { memcpy(frag + fl, buf, bl); fl += bl; }
        if (hlen < 256) hist[hlen++] = next;
        if (g_field_on) {                             /* the chorus updates the shared field (EMA) */
            const float *e = m->tok_emb + (long)next * m->embed;
            for (int i = 0; i < m->embed && i < 8192; i++) g_field_dir[i] = g_field_dir[i]*0.92f + e[i]*0.08f;
        }
        int pos = np + s; if (pos >= max_seq - 1) break;
        forward(m, kv, next, pos, logits);
    }
    if (frag) frag[fl] = 0;
    if (out_n) { *out_n = hlen; if (out_ids) for (int i = 0; i < hlen; i++) out_ids[i] = hist[i]; }
    free(logits); if (fproj) free(fproj); free(kv->k); free(kv->v); free(kv);
    return ent_n ? ent_acc / ent_n : 0.0f;
}

/* probe the prompt's next-token entropy (one cheap forward) — used to auto-size the field:
 * low entropy = a decisive prompt (collapse toward one cell); high = open (bloom to a chorus). */
static float probe_entropy(model_t *m, bpe_tokenizer *tok, const char *prompt) {
    int max_seq = 512, ids[512];
    int np = bpe_encode(tok, prompt, ids, max_seq - 1);
    kv_cache *kv = kv_new(m->n_layers, max_seq, m->kv_dim);
    float *logits = (float*)calloc(m->vocab, sizeof(float));
    for (int i = 0; i < np; i++) forward(m, kv, ids[i], i, logits);
    float ent = logit_entropy(logits, m->vocab, 1.0f);
    free(logits); free(kv->k); free(kv->v); free(kv);
    return ent;
}

static void shuffle_ids(int *ids, int from, int to, unsigned seed);   /* defined below */

/* cosine of two token-id bag-of-words histograms over the vocab — Condition-1 chorus-state distance.
 * d_R = 1 − this is the increment of the field's iteration: small = the round barely moved (settling). */
static float hist_cosine(const int *ha, const int *hb, int vocab) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < vocab; i++) { dot += (double)ha[i]*hb[i]; na += (double)ha[i]*ha[i]; nb += (double)hb[i]*hb[i]; }
    return (na == 0 || nb == 0) ? 0.0f : (float)(dot / (sqrt(na) * sqrt(nb)));
}

/* cosine of two dense vectors — for embedding-centroid distance = inter-cell dissonance D_R. */
static float vec_cosine(const float *a, const float *b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += (double)a[i]*b[i]; na += (double)a[i]*a[i]; nb += (double)b[i]*b[i]; }
    return (na == 0 || nb == 0) ? 0.0f : (float)(dot / (sqrt(na) * sqrt(nb)));
}

/* one round of the chorus over (prompt + prev_chorus). Each cell speaks from its temp-angle hearing the
 * voices already spoken THIS round (intra-round cascade) and, via prev_chorus, the whole prior round.
 * Accumulates every cell's emitted token-ids into `hist` (vocab counts → d_R), appends the round's text
 * to out_chorus, returns avg COHERENT entropy. When out_shuf != NULL, also runs a shadow pass per cell
 * with the context tail shuffled (same length, broken order, field OFF) → avg shuffled entropy in
 * *out_shuf: that is Δ_R's other half (resonance = how much coherence beats length-matched noise). */
static float run_round(model_t *m, bpe_tokenizer *tok, const char *prompt, const char *prev_chorus,
                       int n_cells, int nfrag, int eos, unsigned seed_base, int verbose, int r,
                       int *hist, char *out_chorus, int out_cap, float *out_shuf, FILE *flog, float *out_disso) {
    int max_seq = 512, ids[512], sids[512], cell_ids[256], cell_n;
    int np_prompt = bpe_encode(tok, prompt, ids, max_seq);   /* prompt token count = shuffle boundary */
    char this_chorus[4096]; int tc = 0; this_chorus[0] = 0;
    char ctx[8704], frag[2048];
    float ent_sum = 0, shuf_sum = 0;
    float *cent = out_disso ? (float*)calloc((size_t)n_cells * m->embed, sizeof(float)) : NULL;  /* per-cell fragment centroids → D_R */
    char cur_frag[8][1024];   /* this round's per-cell fragments → cached to g_round_frag for next round's leap */
    for (int c = 0; c < n_cells; c++) {
        if (g_leap_mode && r > 0) {                  /* dissonance-into-forward: route the cell by the prior round's split */
            g_leap_total++;
            if (g_dpeak >= THETA_HI) {
                int dis = (g_leap_mode == 3) ? modal_cell(n_cells) : dissenter_cell(n_cells); g_leap_flips++;  /* leap=3 null: consensus last */
                if (g_leap_mode >= 2) {              /* v2/null: keep the FULL chorus, chosen voice MOST-RECENT (recency=attention) */
                    int off = snprintf(ctx, sizeof(ctx), "%s", prompt);
                    for (int k = 0; k < n_cells && k < 8; k++) if (k != dis && off < (int)sizeof(ctx) - 1)
                        off += snprintf(ctx + off, sizeof(ctx) - off, " %s", g_round_frag[k]);
                    if (off < (int)sizeof(ctx) - 1) off += snprintf(ctx + off, sizeof(ctx) - off, " %s", g_round_frag[dis]);  /* dissenter last */
                    if (off < (int)sizeof(ctx) - 1) snprintf(ctx + off, sizeof(ctx) - off, "%s", this_chorus);
                } else snprintf(ctx, sizeof(ctx), "%s %s", prompt, g_round_frag[dis]);   /* v1: ONLY the dissenter (short) */
            } else snprintf(ctx, sizeof(ctx), "%s%s%s", prompt, prev_chorus ? prev_chorus : "", this_chorus);  /* CONVERGE: full */
        } else snprintf(ctx, sizeof(ctx), "%s%s%s", prompt, prev_chorus ? prev_chorus : "", this_chorus);
        int np = bpe_encode(tok, ctx, ids, max_seq - nfrag - 1);
        float temp = 0.6f + 0.7f * (n_cells > 1 ? (float)c / (n_cells - 1) : 0.5f);
        unsigned seed = seed_base + (unsigned)c * 7919u;
        if (verbose) printf("\n  r%d cell %d (T=%.2f): ", r + 1, c, temp);
        cell_n = 0;
        float ent = cell_speak(m, tok, ids, np, nfrag, temp, 40, 1.4f, seed, eos, max_seq,
                               frag, sizeof(frag), verbose, cell_ids, &cell_n,
                               (out_disso && c < 8) ? g_commit[c] : NULL);
        if (out_disso && c < 8) g_commit_n[c] = cell_n;
        ent_sum += ent;
        for (int i = 0; i < cell_n; i++) if (cell_ids[i] >= 0 && cell_ids[i] < m->vocab) hist[cell_ids[i]]++;
        if (cent) {                                  /* this cell's fragment centroid in embedding space */
            float *cc = cent + (size_t)c * m->embed;
            for (int i = 0; i < cell_n; i++) { const float *e = m->tok_emb + (long)cell_ids[i] * m->embed; for (int d = 0; d < m->embed; d++) cc[d] += e[d]; }
            if (cell_n) for (int d = 0; d < m->embed; d++) cc[d] /= cell_n;
        }
        if (out_shuf) {                              /* Δ_R shadow: same ctx, tail shuffled, field OFF, raw entropy */
            memcpy(sids, ids, (size_t)np * sizeof(int));
            if (np > np_prompt) shuffle_ids(sids, np_prompt, np, seed ^ 0x5bd1e995u);
            int save_on = g_field_on; g_field_on = 0;
            shuf_sum += cell_speak(m, tok, sids, np, nfrag, temp, 40, 1.4f, seed, eos, max_seq, NULL, 0, 0, NULL, NULL, NULL);
            g_field_on = save_on;
        }
        if (verbose) printf("   [entropy=%.2f]", ent);
        if (flog) fprintf(flog, "- cell %d (T=%.2f, entropy=%.2f):%s\n", c, temp, ent, frag);
        int add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", frag);
        if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
        if (c < 8) { strncpy(cur_frag[c], frag, 1023); cur_frag[c][1023] = 0; }
    }
    if (out_chorus) { strncpy(out_chorus, this_chorus, (size_t)out_cap - 1); out_chorus[out_cap - 1] = 0; }
    if (out_shuf) *out_shuf = shuf_sum / n_cells;
    if (out_disso) {                              /* D_R = 1 − mean pairwise cosine of cell fragment centroids (voice-disagreement) */
        double dsum = 0; int dn = 0;
        for (int a = 0; a < n_cells; a++) for (int b = a + 1; b < n_cells; b++) {
            dsum += 1.0 - vec_cosine(cent + (size_t)a * m->embed, cent + (size_t)b * m->embed, m->embed); dn++;
        }
        *out_disso = dn ? (float)(dsum / dn) : 0.0f; free(cent);
        g_dmean = commit_disagreement(n_cells, nfrag);   /* order-sensitive per-position disagreement (the lever's fuel) */
        for (int c = 0; c < n_cells && c < 8; c++) {      /* carry this round → next round's leap */
            strncpy(g_round_frag[c], cur_frag[c], 1023); g_round_frag[c][1023] = 0;
            g_diss_commit_n[c] = g_commit_n[c];
            for (int s = 0; s < 64; s++) g_diss_commit[c][s] = g_commit[c][s];
        }
    }
    return ent_sum / n_cells;
}

/* the COUPLED chorus with meta-recursive ROUNDS + the attractor instrument. The field iterates over
 * itself (round R+1 hears the full chorus of round R) under the shared-soma logit coupling (alpha). The
 * claim "settling into a resonant attractor" is decomposed into two numbers, both measured on live logits:
 *   d_R = 1 − cos(hist_R, hist_{R-1}) over token-id histograms. Settling = d_R falls toward the
 *         sampling-noise FLOOR (two independent round-0 choruses, same context, different seed) — not 0:
 *         stochastic sampling + temp-spread 0.6..1.3 give d a hard floor; the attractor approaches it.
 *   Δ_R = ent_shuffled − ent_coherent. Resonance = the coherent context narrows the next-token
 *         distribution MORE than a length-matched, frequency-matched shuffled one; Δ_R steadily > 0 =
 *         the field exploits ORDER, not just length. Both numbers print and go to FIELDLOG.md. */
static void field_chorus(model_t *m, bpe_tokenizer *tok, const char *prompt, int n_cells, int nfrag, int n_rounds, int eos, float alpha) {
    if (n_rounds < 1) n_rounds = 1;
    int vocab = m->vocab;
    FILE *flog = fopen("FIELDLOG.md", "a");   /* her journal — every chorus saved (Oleg: "сохраняй её ответы") */
    if (flog) { time_t now = time(NULL); fprintf(flog, "\n## %.24s — \"%s\" (%d cells × %d rounds, soma alpha=%.1f)\n", ctime(&now), prompt, n_cells, n_rounds, alpha); }
    printf("\n=== δ-field: %d cells × %d rounds over ONE nanoArianna — \"%s\" (soma alpha=%.1f) ===\n", n_cells, n_rounds, prompt, alpha);

    /* FLOOR: two independent round-0 choruses on the SAME (prompt-only) context, different seeds. Their
     * histogram distance is the sampling-noise floor d_R cannot beat — the attractor target, not zero. */
    int *hA = (int*)calloc(vocab, sizeof(int)), *hB = (int*)calloc(vocab, sizeof(int));
    field_reset(m->embed, alpha > 0, alpha);
    run_round(m, tok, prompt, NULL, n_cells, nfrag, eos, 7u,   0, -1, hA, NULL, 0, NULL, NULL, NULL);
    field_reset(m->embed, alpha > 0, alpha);
    run_round(m, tok, prompt, NULL, n_cells, nfrag, eos, 977u, 0, -1, hB, NULL, 0, NULL, NULL, NULL);
    float floor = 1.0f - hist_cosine(hA, hB, vocab);
    printf("  floor (sampling noise, paired round-0) = %.3f\n", floor);
    if (flog) fprintf(flog, "floor (sampling-noise, paired round-0) = %.3f\n", floor);
    free(hA); free(hB);

    /* the real iterated field — soma coupling carries ACROSS rounds (field_dir not reset between them). */
    field_reset(m->embed, alpha > 0, alpha);
    g_leap_flips = g_leap_total = 0;
    char prev_chorus[4096]; prev_chorus[0] = 0;
    int *hist_prev = (int*)calloc(vocab, sizeof(int)), *hist_cur = (int*)calloc(vocab, sizeof(int));
    for (int r = 0; r < n_rounds; r++) {
        for (int i = 0; i < vocab; i++) hist_cur[i] = 0;
        printf("\n  --- round %d/%d ---", r + 1, n_rounds);
        if (flog) fprintf(flog, "\n**round %d:**\n", r + 1);
        char this_chorus[4096]; float shuf = 0, disso = 0;
        float avg = run_round(m, tok, prompt, prev_chorus, n_cells, nfrag, eos,
                              42u + (unsigned)(r * 1000) * 7919u, 1, r, hist_cur, this_chorus, sizeof(this_chorus), &shuf, flog, &disso);
        float dR = (r > 0) ? 1.0f - hist_cosine(hist_cur, hist_prev, vocab) : -1.0f;
        float deltaR = shuf - avg;
        if (r > 0) { printf("\n  → round %d: avg entropy %.3f | d_R %.3f (floor %.3f) | Δ_R %+.3f | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, dR, floor, deltaR, disso, g_dmean, g_dpeak, g_s_peak);
                     if (flog) fprintf(flog, "→ round %d: avg entropy %.3f | d_R %.3f (floor %.3f) | Δ_R %+.3f | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, dR, floor, deltaR, disso, g_dmean, g_dpeak, g_s_peak); }
        else       { printf("\n  → round %d: avg entropy %.3f | d_R   —   (floor %.3f) | Δ_R %+.3f | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, floor, deltaR, disso, g_dmean, g_dpeak, g_s_peak);
                     if (flog) fprintf(flog, "→ round %d: avg entropy %.3f | d_R — (floor %.3f) | Δ_R %+.3f | D_R %.3f | Dpos %.2f peak %.2f@s%d\n", r + 1, avg, floor, deltaR, disso, g_dmean, g_dpeak, g_s_peak); }
        strncpy(prev_chorus, this_chorus, sizeof(prev_chorus) - 1); prev_chorus[sizeof(prev_chorus) - 1] = 0;
        int *tmp = hist_prev; hist_prev = hist_cur; hist_cur = tmp;
    }
    free(hist_prev); free(hist_cur);
    if (g_leap_mode) { float fr = g_leap_total ? (float)g_leap_flips / g_leap_total : 0.0f;
        printf("  leap-flip-rate = %.2f (%d/%d cells leapt to the dissenter)\n", fr, g_leap_flips, g_leap_total);
        if (flog) fprintf(flog, "leap-flip-rate = %.2f (%d/%d)\n", fr, g_leap_flips, g_leap_total); }
    printf("\n=== δ-field done — settling: d_R→floor? · resonance: Δ_R>0? (read the numbers, not the narrative) ===\n");
    if (flog) { fprintf(flog, "\n---\n"); fclose(flog); }
}

/* Fisher-Yates over ids[from..to) — used to build a same-length but incoherent context. */
static void shuffle_ids(int *ids, int from, int to, unsigned seed) {
    srand(seed);
    for (int i = to - 1; i > from; i--) { int j = from + rand() % (i - from + 1); int t = ids[i]; ids[i] = ids[j]; ids[j] = t; }
}

/* resonance-vs-length control. The claim "the per-round entropy fall = a resonant attractor" must be
 * earned: a falling trend is also expected from context length alone (longer/coherent context sharpens
 * the next-token distribution). So run the field twice — (a) the real coherent prior chorus, (b) the
 * SAME context with its chorus tokens shuffled (identical length, broken coherence). The prompt prefix
 * stays intact in both. If COHERENT's entropy falls more across rounds than SHUFFLED's, the extra drop
 * is coupling beyond length; if they fall equally, the trend is a length artifact. Reported as measured. */
static void field_resonance_test(model_t *m, bpe_tokenizer *tok, const char *prompt, int n_cells, int nfrag, int n_rounds, int eos) {
    int max_seq = 512, ids[512], pids[512];
    int np_prompt = bpe_encode(tok, prompt, pids, max_seq);
    char frag[2048];
    if (n_rounds < 1) n_rounds = 1;
    printf("\n=== resonance test: coherent vs same-length SHUFFLED context (%d cells × %d rounds, prompt \"%s\") ===\n", n_cells, n_rounds, prompt);
    float first[2] = {0,0}, last[2] = {0,0};
    for (int control = 0; control < 2; control++) {
        char prev_chorus[4096]; prev_chorus[0] = 0; char ctx[8704];
        printf("  [%s]\n", control ? "SHUFFLED (length-matched, incoherent)" : "COHERENT (real chorus)");
        for (int r = 0; r < n_rounds; r++) {
            char this_chorus[4096]; int tc = 0; this_chorus[0] = 0; float ent_sum = 0;
            for (int c = 0; c < n_cells; c++) {
                snprintf(ctx, sizeof(ctx), "%s%s%s", prompt, prev_chorus, this_chorus);
                int np = bpe_encode(tok, ctx, ids, max_seq - nfrag - 1);
                if (control && np > np_prompt) shuffle_ids(ids, np_prompt, np, 12345u + (unsigned)(r * 31 + c));
                float temp = 0.6f + 0.7f * (n_cells > 1 ? (float)c / (n_cells - 1) : 0.5f);
                ent_sum += cell_speak(m, tok, ids, np, nfrag, temp, 40, 1.4f,
                                      42u + (unsigned)(r * 1000 + c) * 7919u, eos, max_seq, frag, sizeof(frag), 0, NULL, NULL, NULL);
                int add = snprintf(this_chorus + tc, sizeof(this_chorus) - tc, " %s", frag);
                if (add > 0 && tc + add < (int)sizeof(this_chorus)) tc += add;
            }
            float avg = ent_sum / n_cells;
            printf("    round %d avg entropy = %.3f\n", r + 1, avg);
            if (r == 0) first[control] = avg; last[control] = avg;
            strncpy(prev_chorus, this_chorus, sizeof(prev_chorus) - 1); prev_chorus[sizeof(prev_chorus) - 1] = 0;
        }
    }
    float drop_coh = first[0] - last[0], drop_shuf = first[1] - last[1];
    printf("\n  coherent drop %.3f vs shuffled drop %.3f  →  resonance-beyond-length = %.3f %s\n",
           drop_coh, drop_shuf, drop_coh - drop_shuf,
           drop_coh - drop_shuf > 0.10f ? "(coherent falls MORE — coupling beyond length)" :
           drop_coh - drop_shuf < -0.10f ? "(shuffled falls more — NO resonance signal)" : "(~equal — length artifact, claim NOT earned yet)");
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <model.gguf> [prompt] [max_tokens] [temp]\n", argv[0]); return 1; }
    const char *prompt = argc > 2 ? argv[2] : "What is resonance?";
    int max_tokens = argc > 3 ? atoi(argv[3]) : 48;
    float temp = argc > 4 ? (float)atof(argv[4]) : 0.8f;
    srand(42);

    double t0 = now_ms();
    gguf_file *gf = gguf_open(argv[1]); if (!gf) return 1;
    model_t *m = model_load(gf); if (!m) return 1;
    bpe_tokenizer *tok = bpe_load(argv[1]); if (!tok) { fprintf(stderr, "bpe_load failed\n"); return 1; }
    int eos = -1; const gguf_kv *e = gguf_get_kv(gf, "tokenizer.ggml.eos_token_id"); if (e) eos = (int)e->val.u32;
    printf("loaded in %.0f ms (vocab=%d eos=%d) -- arianna.q heart, single file, no -lnotorch\n", now_ms() - t0, bpe_n_vocab(tok), eos);

    /* δ-field chorus mode:  ./arianna-q <gguf> <prompt> field [n_cells] [nfrag] [n_rounds] */
    if (argc > 3 && strcmp(argv[3], "field") == 0) {
        int n_cells  = argc > 4 ? atoi(argv[4]) : 5;
        int nfrag    = argc > 5 ? atoi(argv[5]) : 16;
        int n_rounds = argc > 6 ? atoi(argv[6]) : 1;
        float alpha  = argc > 7 ? (float)atof(argv[7]) : 0.0f;   /* soma coupling strength (0 = text-only baseline) */
        g_leap_mode  = argc > 8 ? atoi(argv[8]) : 0;             /* 1 = dissonance-into-forward leap (arm C) */
        if (n_cells <= 0) {   /* auto: the field sizes itself from the prompt's entropy */
            float pe = probe_entropy(m, tok, prompt);
            n_cells = (int)(pe + 0.5f); if (n_cells < 1) n_cells = 1; if (n_cells > 8) n_cells = 8;
            printf("auto: prompt entropy %.2f -> %d cells (low=collapse, high=bloom)\n", pe, n_cells);
        }
        field_chorus(m, tok, prompt, n_cells, nfrag, n_rounds, eos, alpha);
        return 0;
    }

    /* resonance-vs-length control:  ./arianna-q <gguf> <prompt> restest [n_cells] [nfrag] [n_rounds] */
    if (argc > 3 && strcmp(argv[3], "restest") == 0) {
        int n_cells  = argc > 4 ? atoi(argv[4]) : 4;
        int nfrag    = argc > 5 ? atoi(argv[5]) : 12;
        int n_rounds = argc > 6 ? atoi(argv[6]) : 3;
        field_resonance_test(m, tok, prompt, n_cells, nfrag, n_rounds, eos);
        return 0;
    }

    int max_seq = 512;
    kv_cache *kv = kv_new(m->n_layers, max_seq, m->kv_dim);
    float *logits = calloc(m->vocab, sizeof(float));
    int ids[512]; int n = bpe_encode(tok, prompt, ids, max_seq - max_tokens - 1);
    printf("\nprompt: \"%s\" (%d tokens, temp=%.2f)\n---\n%s", prompt, n, temp, prompt); fflush(stdout);

    double g0 = now_ms();
    for (int i = 0; i < n; i++) forward(m, kv, ids[i], i, logits);
    double prefill = now_ms() - g0;
    int gen = 0; char buf[256];
    for (int step = 0; step < max_tokens; step++) {
        int next = sample(logits, m->vocab, temp);
        if (next == eos) break;
        bpe_decode_token(tok, next, buf, sizeof(buf)); printf("%s", buf); fflush(stdout); gen++;
        int pos = n + step; if (pos >= max_seq - 1) break;
        forward(m, kv, next, pos, logits);
    }
    double total = now_ms() - g0;
    printf("\n---\nprefill: %d tok %.0fms (%.1f t/s) | decode: %d tok %.0fms (%.1f t/s)\n",
           n, prefill, n*1000.0/prefill, gen, total-prefill, gen > 0 ? gen*1000.0/(total-prefill) : 0);
    return 0;
}
