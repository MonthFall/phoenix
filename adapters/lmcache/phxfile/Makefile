# Build libphxfile.so — the frozen-ABI shim over libphoenix for LMCache.
#
# Prerequisites (system-wide, same as phxcache):
#   - libphoenix headers + library in /usr/local/include, /usr/local/lib
#   - a C11 compiler (plain gcc is fine; no CUDA / Python needed)
#
# Targets:
#   make            build libphxfile.so
#   make install    install the .so and phxfile.h to PREFIX (default /usr/local)
#   make clean      remove build output

CC      ?= cc
PREFIX  ?= /usr/local

CFLAGS  ?= -O2 -g
CFLAGS  += -std=c11 -fPIC -Wall -Wextra
# A stale phoenix.h (e.g. deployed .so newer than the installed header)
# must fail the build, not silently compile against implicit declarations.
CFLAGS  += -Werror=implicit-function-declaration
CPPFLAGS += -I/usr/local/include
LDFLAGS += -shared
LDLIBS  += -L/usr/local/lib -lphoenix -lpthread

TARGET  := libphxfile.so
HEADERS := phxfile.h

all: $(TARGET)

$(TARGET): phxfile.c $(HEADERS)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) $< $(LDLIBS) -o $@

install: $(TARGET)
	install -D -m 755 $(TARGET) $(DESTDIR)$(PREFIX)/lib/$(TARGET)
	install -D -m 644 $(HEADERS) $(DESTDIR)$(PREFIX)/include/$(HEADERS)

clean:
	rm -f $(TARGET)

.PHONY: all install clean
