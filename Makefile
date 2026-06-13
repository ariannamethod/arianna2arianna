# arianna2arianna — single-file C heart. Punk: no external deps, just cc + libm.
# Build:  make            Run:  make run            Field:  make field
# Sweep:  make sweep      Test: make test           Portable: make portable

CC     ?= cc
CFLAGS ?= -O2 -Wall
LDLIBS := -lm
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifneq ($(PORTABLE),1)
  ifeq ($(UNAME_M),arm64)
    CFLAGS += -march=armv8.2-a+fp16+dotprod
  else ifeq ($(UNAME_M),x86_64)
    CFLAGS += -mavx2 -mfma -mf16c
  endif
  ifeq ($(UNAME_S),Darwin)
    CFLAGS += -DUSE_BLAS -DACCELERATE_NEW_LAPACK
    LDLIBS += -framework Accelerate
  else ifneq ($(shell pkg-config --exists openblas 2>/dev/null && echo yes),)
    CFLAGS += -DUSE_BLAS
    LDLIBS += -lopenblas
  endif
endif

BIN    := arianna2arianna
SRC    := arianna2arianna.c

HF_BASE = https://huggingface.co/ataeff/ariannamethod/resolve/main/weights
WEIGHTS = weights
MODEL  ?= $(WEIGHTS)/nanollama-arianna-full-v4-step2750-f16.gguf
PROMPT ?= What is resonance?

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDLIBS) -o $(BIN)

$(WEIGHTS)/nanollama-arianna-full-v4-step2750-f16.gguf:
	@mkdir -p $(WEIGHTS)
	curl -fL -o $@ $(HF_BASE)/nanollama-arianna-full-v4-step2750-f16.gguf

$(WEIGHTS)/nanollama-arianna-full-v4-step2750-q8_0.gguf:
	@mkdir -p $(WEIGHTS)
	curl -fL -o $@ $(HF_BASE)/nanollama-arianna-full-v4-step2750-q8_0.gguf

weights: $(MODEL)

portable:
	$(MAKE) PORTABLE=1 CFLAGS="-O2 -Wall -DA2A_SCALAR_ONLY" LDLIBS="-lm" clean $(BIN)

run: $(BIN) $(MODEL)
	./$(BIN) --16 "$(PROMPT)" 48 0.8

run-q8: $(BIN) $(WEIGHTS)/nanollama-arianna-full-v4-step2750-q8_0.gguf
	./$(BIN) --q8 "$(PROMPT)" 48 0.8

field: $(BIN) $(MODEL)
	./$(BIN) --16 "$(PROMPT)" field 4 12 3 0 2 0.25

# batch: 5 seeds × leap 0..3 → CSV on stdout
sweep: $(BIN) $(MODEL)
	./$(BIN) --16 --quiet "$(PROMPT)" sweep 5 4 12 3 0 0.25

clean:
	rm -f $(BIN)

.PHONY: run run-q8 field sweep clean weights test portable bench

test: $(BIN) $(MODEL)
	bash tests/run.sh

bench: $(BIN) $(MODEL)
	./$(BIN) --16 "What is resonance?" 24 0.8