# arianna2arianna — single-file C heart. Punk: no external deps, just cc + libm.
# (GGUF + SentencePiece + Llama forward + δ-field).
# Build:  make            Run one voice:  make run            δ-field chorus:  make field

CC     ?= cc
CFLAGS ?= -O2 -Wall
LDLIBS := -lm
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

# single voice (continuation)
run: $(BIN) $(MODEL)
	./$(BIN) "$(MODEL)" "$(PROMPT)" 48 0.8

# δ-field chorus: N cells x R rounds, coupled + meta-recursive (cells hear each other)
field: $(BIN) $(MODEL)
	./$(BIN) "$(MODEL)" "$(PROMPT)" field 4 12 3

clean:
	rm -f $(BIN)

.PHONY: run field clean weights