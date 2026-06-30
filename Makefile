# arianna2arianna — single-file C heart. No external deps, just cc + libm + pthread.
# Build: make    Weights: make weights    Run: make run    Field: make field
# Life:  make life    Control: make restest    Test: make test

CC     ?= cc
CFLAGS ?= -O2 -Wall
LDLIBS := -lm -pthread
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifneq ($(PORTABLE),1)
  ifeq ($(UNAME_M),arm64)
    CFLAGS += -march=armv8.2-a+fp16+dotprod
  else ifeq ($(FAST_X86),1)
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
MODEL_F16 = $(WEIGHTS)/nanollama-arianna-full-v4-step2750-f16.gguf
MODEL_Q8  = $(WEIGHTS)/nanollama-arianna-full-v4-step2750-q8_0.gguf
MODEL  ?= $(MODEL_F16)
PROMPT ?= What is resonance?
TOKENS ?= 48
TEMP   ?= 0.8
CELLS  ?= 4
FRAG   ?= 12
ROUNDS ?= 3
ALPHA  ?= 0
LEAP   ?= 2
XCELL  ?= 0.30
TICKS  ?= 5
INIT   ?= 4
SWEEP_PROMPTS ?= prompts/kv_influence.txt
REPL_PROMPTS ?= prompts/repl_questions.txt

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $(SRC) $(LDLIBS) -o $(BIN)

$(MODEL_F16):
	@mkdir -p $(WEIGHTS)
	curl -fL -o $@ $(HF_BASE)/nanollama-arianna-full-v4-step2750-f16.gguf

$(MODEL_Q8):
	@mkdir -p $(WEIGHTS)
	curl -fL -o $@ $(HF_BASE)/nanollama-arianna-full-v4-step2750-q8_0.gguf

weights: $(MODEL_F16)
weights-q8: $(MODEL_Q8)

portable:
	$(MAKE) PORTABLE=1 CFLAGS="-O2 -Wall -DA2A_SCALAR_ONLY" LDLIBS="-lm -pthread" clean $(BIN)

fast-x86:
	$(MAKE) FAST_X86=1 clean $(BIN)

run: $(BIN) $(MODEL)
	./$(BIN) "$(MODEL)" "$(PROMPT)" $(TOKENS) $(TEMP)

run-q8: $(BIN) $(MODEL_Q8)
	./$(BIN) "$(MODEL_Q8)" "$(PROMPT)" $(TOKENS) $(TEMP)

repl: $(BIN) $(MODEL)
	./$(BIN) "$(MODEL)" repl $(CELLS) $(FRAG) $(ROUNDS)

field: $(BIN) $(MODEL)
	./$(BIN) "$(MODEL)" "$(PROMPT)" field $(CELLS) $(FRAG) $(ROUNDS) $(ALPHA) $(LEAP) $(XCELL)

restest: $(BIN) $(MODEL)
	./$(BIN) "$(MODEL)" "$(PROMPT)" restest $(CELLS) $(FRAG) $(ROUNDS)

life: $(BIN) $(MODEL)
	./$(BIN) "$(MODEL)" "$(PROMPT)" life $(TICKS) $(FRAG) $(INIT)

sweep-influence: $(BIN) $(MODEL)
	bash tools/kv_influence_sweep.sh "$(SWEEP_PROMPTS)"

repl-sweep: $(BIN) $(MODEL)
	bash tools/repl_question_sweep.sh "$(REPL_PROMPTS)"

clean:
	rm -f $(BIN)

.PHONY: run run-q8 repl field restest life sweep-influence repl-sweep clean weights weights-q8 test portable fast-x86 bench

test: $(BIN)
	bash tests/run.sh

bench: $(BIN) $(MODEL)
	./$(BIN) "$(MODEL)" "What is resonance?" 24 0.8
