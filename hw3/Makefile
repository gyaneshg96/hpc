all:
	g++ -fopenmp omp_bug2.c -o omp-solved2
	g++ -fopenmp omp_bug3.c -o omp-solved3
	g++ -fopenmp omp_bug2.c -o omp-solved4
	g++ -fopenmp omp_bug2.c -o omp-solved5
	g++ -fopenmp omp_bug2.c -o omp-solved6
	g++ -fopenmp omp-scan.cpp -o omp-scan
	g++ -fopenmp jacobi2D-omp.cpp -o jacobi2D-omp
	g++ -fopenmp gs2D-omp.cpp -o gs2D-omp

clean:
	rm omp-solved2 omp-solved3 omp-solved4 omp-solved5 omp-solved6 omp-scan