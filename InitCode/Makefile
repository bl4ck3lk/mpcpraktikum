CC = g++
CFLAGS = 
EXECUTEABLE = initialize

OBJECTS = initialize.o imageHandler.o

all : $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $(EXECUTEABLE)

imageHandler.o : imageHandler.cpp
	$(CC) $(CFLAGS) -c $<

initialize.o : initialize.cpp imageHandler.h
	$(CC) $(CFLAGS) -c $<

.PHONY : clean
clean :
	rm  -rf $(OBJECTS) 

.PHONY : veryclean
veryclean :
	rm  -rf $(OBJECTS) $(EXECUTEABLE)
