# arianna2arianna — single-file C heart. Punk: no external deps, just cc + libm.
# (GGUF + SentencePiece + Llama forward + δ-field).
# Build:  make            Run one voice:  make run            δ-field chorus:  make field

CC     ?= cc
CFLAGS ?= -O2 -Wall
LDLIBS := -lm
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CFLAGS += -DUSE_BLAS -DACCELERATE_NEW_LAPACK
  LDLIBS += -framework Accelerate
else ifneq ($(shell pkg-config --exists openblas 2>/dev/null && echo yes),)
  CFLAGS += -DUSE_BLAS
  LDLIBS += -lopenblas
endif
BIN    := ariannameethod
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

# single voice (continuation) — packed f16 from HF
run: $(BIN) $(MODEL)
	./$(BIN) --16 "$(PROMPT)" 48 0.8

run-q8: $(BIN) $(WEIGHTS)/nanollama-arianna-full-v4-step2750-q8_0.gguf
	./$(BIN) --q8 "$(PROMPT)" 48 0.8

# δ-field chorus: N cells x R rounds, coupled + meta-recursive (cells hear each other)
field: $(BIN) $(MODEL)
	./$(BIN) --16 "$(PROMPT)" field 4 12 3

clean:
	rm -f $(BIN)

.PHONY: run run-q8 field clean weights test

test: $(BIN) $(MODEL)
	./$(BIN) --16 "What is resonance?" 6 0.8
	./$(BIN) --q8 "What is resonance?" 6 0.8