CC = gcc
CFLAGS = -shared -fPIC -Wall -O3
LDFLAGS = -lraylib

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    TARGET = libblockblast.dylib
    # macOS Homebrew paths
    CFLAGS += -I/opt/homebrew/include
    LDFLAGS += -L/opt/homebrew/lib
else
    TARGET = libblockblast.so
endif

all: $(TARGET)

$(TARGET): blockblast_lib.c
	$(CC) $(CFLAGS) blockblast_lib.c -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
