# MAKEFLAGS += --silent

CC = g++
BUILD = build
INCLUDE = include
TESTS = tests
LIB = lib
CC_FLAGS = -g -std=c++2a -I ./$(INCLUDE)
# -g for gdb
# -std=c++2a for c++20

main: $(BUILD)/main.o $(LIB)/libneural.a
	@printf "building main\n"
	$(CC) $(CC_FLAGS) -pg $(BUILD)/main.o -I $(LIB) $(LIB)/libneural.a -o main

test: $(LIB)/libneural.a $(BUILD)/MLPTests.o
	@printf "building test\n"
	$(CC) $(CC_FLAGS) -pg $(BUILD)/MLPTests.o -I $(LIB) $(LIB)/libneural.a -o test

$(LIB)/libneural.a: $(BUILD)/Matrix.o $(BUILD)/MLP.o $(BUILD)/SGD.o
	rm $(LIB)/libneural.a
	ar rvs $(LIB)/libneural.a $(BUILD)/Matrix.o $(BUILD)/MLP.o $(BUILD)/SGD.o

$(BUILD)/main.o: Main.cpp
	@printf "building $<\n"
	$(CC) $(CC_FLAGS) Main.cpp -c -o $(BUILD)/main.o

$(BUILD)/MLPTests.o: $(TESTS)/MLPTests.cpp
	@printf "building $<\n"
	$(CC) $(CC_FLAGS) $(TESTS)/MLPTests.cpp -c -o $(BUILD)/MLPTests.o

$(BUILD)/%.o: $(LIB)/%.cpp $(INCLUDE)/%.h
	@printf "building $<\n"
	$(CC) $(CC_FLAGS) $< -c -o $@

clean:
	@printf "cleaning\n"
	-rm $(BUILD)/* -f
	-rm test main
