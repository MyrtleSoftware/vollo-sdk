all: vollo-example vollo-identity vollo-partial-update

CFLAGS = -Wall -Wextra -Wpedantic -Wconversion

.PHONY: all clean

vollo-example: example.c npy.h utils.h
	$(CC) example.c -O2 -std=gnu11 $(CFLAGS) -I $(VOLLO_SDK)/include -L $(VOLLO_SDK)/lib -lvollo_rt -o vollo-example

vollo-identity: identity.c
	$(CC) identity.c -O2 -std=gnu11 $(CFLAGS) -I $(VOLLO_SDK)/include -L $(VOLLO_SDK)/lib -lvollo_rt -o vollo-identity

vollo-partial-update: partial_update.c npy.h utils.h
	$(CC) partial_update.c -O2 -std=gnu11 $(CFLAGS) -I $(VOLLO_SDK)/include -L $(VOLLO_SDK)/lib -lvollo_rt -o vollo-partial-update

clean:
	rm -f vollo-example vollo-identity vollo-partial-update
