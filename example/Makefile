all: vollo-example vollo-identity

CFLAGS = -Wall -Wextra -Wpedantic -Wconversion

.PHONY: all clean

vollo-example: example.c
	$(CC) example.c -O2 -std=gnu11 $(CFLAGS) -I $(VOLLO_SDK)/include -L $(VOLLO_SDK)/lib -lvollo_rt -o vollo-example

vollo-identity: identity.c
	$(CC) identity.c -O2 -std=gnu11 $(CFLAGS) -I $(VOLLO_SDK)/include -L $(VOLLO_SDK)/lib -lvollo_rt -o vollo-identity

clean:
	rm -f vollo-example vollo-identity
