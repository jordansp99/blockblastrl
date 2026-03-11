CC = gcc
CFLAGS = -shared -fPIC -Wall -O3
LDFLAGS = -lraylib

# Detect OS
ifeq ($(OS),Windows_NT)
    TARGET = libblockblast.dll
    LDFLAGS += -lopengl32 -lgdi32 -lwinmm
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        TARGET = libblockblast.dylib
        CFLAGS += -I/opt/homebrew/include
        LDFLAGS += -L/opt/homebrew/lib
    else
        TARGET = libblockblast.so
    endif
endif

all: $(TARGET)

$(TARGET): blockblast_lib.c
	$(CC) $(CFLAGS) blockblast_lib.c -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
