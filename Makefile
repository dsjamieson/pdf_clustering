CXX = g++

INC = -I/gpfs/home/ajamieson/local/include/
LIB = -L/gpfs/home/ajamieson/local/lib/ -lfftw3_threads -lfftw3 -pthread -lm
CXXFLAGS = -Wall -O2 -fopenmp -std=c++17
TARGET = density_pdf_clustering
default: $(TARGET) 
all: $(TARGET)

density_pdf_clustering: main.cpp density_pdf_clustering.cpp
	$(CXX) $(CXXFLAGS) $(INC) -o $@ $^ $(LIB) 

install:
	cp $(TARGET) /gpfs/home/ajamieson/local/bin

clean:
	rm -f $(TARGET) core *.o


