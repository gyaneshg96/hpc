CXX = mpicxx # or mpicxx
CXXFLAGS = -std=c++11 -O3 -march=native

TARGETS = $(basename $(wildcard *.cpp)) $(basename $(wildcard *.c))


all : $(TARGETS)

%:%.cpp *.h
	$(CXX) $(CXXFLAGS) $< $(LIBS) -o $@


clean:
	-$(RM) $(TARGETS) *~

.PHONY: all, clean
