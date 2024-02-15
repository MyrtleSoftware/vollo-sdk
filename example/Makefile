all: vollo-example

CFLAGS = -Wall -Wextra -Wpedantic -Wconversion

.PHONY: all

vollo-example: main.c
	$(CC) main.c -O2 -std=gnu11 $(CFLAGS) -I $(VOLLO_SDK)/include -L $(VOLLO_SDK)/lib -lvollo_rt -o vollo-example
