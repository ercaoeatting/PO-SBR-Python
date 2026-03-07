CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

TARGET = po_rcs
SRCS = po_solver.c main_rcs.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.c po_solver.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
