OBJS = nn.o config.o network.o utils.o
CC = g++
FLAGS = -std=c++11

nn: $(OBJS)
	$(CC) $(FLAGS) $(OBJS) -o nn
	ar rcs nn.a network.o utils.o

%.o: %.cxx
	$(CC) $(FLAGS) -c $< -o $@

clean:
	rm nn nn.a $(OBJS)

