OBJS = nn.o config.o network.o layer.o matrix.o utils.o
CC = g++
FLAGS = -std=c++11

nn: $(OBJS)
	$(CC) $(FLAGS) -O3 -DNDEBUG $(OBJS) -o nn
	ar rcs nn.a network.o utils.o

%.o: %.cxx
	$(CC) $(FLAGS) -O3 -DNDEBUG -c $< -o $@

test: test.cxx matrix.cxx layer.cxx legacy/network.cxx network.cxx
	$(CC) $(FLAGS) -g test.cxx matrix.cxx layer.cxx network.cxx legacy/network.cxx -o test

clean:
	rm nn nn.a $(OBJS) test

