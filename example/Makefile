all: vollo-example vollo-identity vollo-partial-update

CFLAGS += -std=c99 -O2 -flto -Wall -Wextra -Wpedantic -Wconversion -I$(VOLLO_SDK)/include
LDFLAGS += -flto -L$(VOLLO_SDK)/lib
LDLIBS += -lvollo_rt

.PHONY: all clean

utils.o: utils.c utils.h
npy.o: npy.c npy.h
identity.o: identity.c
example.o: example.c npy.h utils.h
partial_update.o: partial_update.c npy.h utils.h

vollo-identity: identity.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

vollo-example: example.o npy.o utils.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

vollo-partial-update: partial_update.o npy.o utils.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LDLIBS)

clean:
	rm -f vollo-example vollo-identity vollo-partial-update *.o
