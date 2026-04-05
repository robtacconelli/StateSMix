CC = gcc
CFLAGS = -O3 -march=native -mavx2 -mfma -ffast-math -fopenmp -Wall
LDFLAGS = -lm -fopenmp

SRCS = main.c ssm_math.c ssm_vocab.c ssm_tokenizer.c ssm_arithmetic.c ssm_model.c ssm_codec.c ssm_preprocess.c ssm_bwt.c
OBJS = $(SRCS:.c=.o)
TARGET = ssm_best_version2

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
