all:
	g++ -std=c++11 fast-sin.cpp -o fast-sin
	g++ -std=c++11 -fopenmp -march=native -O3 MMult1.cpp -o MMult1	
	g++ -std=c++11 val_test01.cpp -o val_test01_solved 
	g++ -std=c++11 val_test02.cpp -o val_test02_solved 

clean:
	rm fast-sin MMult1 val_test01_solved val_test02_solved

