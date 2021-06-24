#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include "Exception.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"


void testfunc(){

	Vector b(3);
	b(1) = 2;
	b(2) = 5;
	b(3) = 12;
	
	Matrix m(3,3);
	m(1,1) = 1;
	m(1,2) = 2;
	m(1,3) = -1;
	m(2,1) = 1;
	m(2,2) = -1;
	m(2,3) = 2;
	m(3,1) = 2;
	m(3,2) = 2;
	m(3,3) = 2;

	Matrix m2(3,3);
	m2(1,1) = 1;
	m2(1,2) = 2;
	m2(1,3) = -1;
	m2(2,1) = 1;
	m2(2,2) = -1;
	m2(2,3) = 2;
	m2(3,1) = 2;
	m2(3,2) = 2;
	m2(3,3) = 2;

	Matrix c(3,3);
	c(1,1) = 2;
	c(1,2) = 4;
	c(1,3) = -2;
	c(2,1) = 2;
	c(2,2) = -2;
	c(2,3) = 4;
	c(3,1) = 4;
	c(3,2) = 4;
	c(3,3) = 4;

	Matrix c2(3,3);
	c2(1,1) = 2;
	c2(1,2) = -4;
	c2(1,3) = 2;
	c2(2,1) = 8;
	c2(2,2) = 14;
	c2(2,3) = 2;
	c2(3,1) = 16;
	c2(3,2) = 12;
	c2(3,3) = 12;

	Matrix d(3,3);

	if (!compare(m,m2)){
		throw Exception("Test not passed","Comparison operator failed.");
	}

	Matrix m3(m);
	if (!compare(m3,m)){
		throw Exception("Test not passed","Copy constructor failed.");
	}

	m3 = c;
	if (!compare(m3,c)){
		throw Exception("Test not passed","Assignment operator failed.");
	}

	m3 = m + m2;
	if (!compare(m3,c)){
		throw Exception("Test not passed","Addition operator failed.");
	}

	m3 = m - m2;
	if (!compare(m3,d)){
		throw Exception("Test not passed","Subtraction operator failed.");
	}

	m3 = m*c;
	if (!compare(m3,c2)){
		throw Exception("Test not passed","Matrix-matrix multiplication operator failed.");
	}

	Vector x(3);
	x(1) = 1;
	x(2) = 2;
	x(3) = 3;
	
	Vector v = m*x;

	if (!compare(v,b)){
		throw Exception("Test not passed","Matrix-vector multiplication operator failed.");
	}

	m3 = m*2;

	if (!compare(m3,c)){
		throw Exception("Test not passed","Matrix-vector multiplication operator failed.");
	}

	Vector s(2);
	s(1) = 3;
	s(2) = 3;

	if (!compare(s,size(m))){
		throw Exception("Test not passed","Matrix-vector multiplication operator failed.");
	}

	if (!(-6==determinant(m,3))){
		throw Exception("Test not passed","Determinant function failed.");
	}


	int l = 10;
	Matrix A(l,l);
	for (int i=1;i<=l;i++){
		for (int j=1;j<=l;j++){
			A(i,j) = std::rand()%10;
			//A(i,j)=i+j;
		}
	}

	double tol = 0.000001;
	Vector X(l);
	for (int i=1;i<=l;i++){
		X(i)=4*i;
	}


	Vector B=A*X;

	
	//backslash
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	v=B/A;
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	//std::cout << "Time difference backslash = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
	for (int i=1;i<=l;i++){
		if (abs(X(i)-v(i))>tol){
			throw Exception("Test not passed","Backslash operator failed.");
		}
	}

	//Gaussian Elimination
	begin = std::chrono::steady_clock::now();
	v = gausElim(A,B);
	end = std::chrono::steady_clock::now();
	//std::cout << "Time difference GMRES restarted = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
	for (int i=1;i<=l;i++){
		if (abs(X(i)-v(i))>tol){
			throw Exception("Test not passed","Gaussian elimination failed.");
		}
	}

	//restarted GMRES
	begin = std::chrono::steady_clock::now();
	v = GMRES(A,B,5,tol,100);
	end = std::chrono::steady_clock::now();
	//std::cout << "Time difference GMRES restarted = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
	for (int i=1;i<=l;i++){
		if (abs(X(i)-v(i))>tol){
			throw Exception("Test not passed","GMRES restarted failed.");
		}
	}

	//GMRES
	begin = std::chrono::steady_clock::now();
	v = GMRES(A,B);
	end = std::chrono::steady_clock::now();
	//std::cout << "Time difference GMRES = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
	for (int i=1;i<=l;i++){
		if (abs(X(i)-v(i))>tol){
			throw Exception("Test not passed","GMRES linsolve failed.");
		}
	}
	
	//CG
	A = A.transpose() * A; //A^TA is always symmetric positive definite
	B=A*X;
	begin = std::chrono::steady_clock::now();
	v = CG(A,B);
	end = std::chrono::steady_clock::now();
	//std::cout << "Time difference CG = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
	for (int i=1;i<=l;i++){
		if (abs(X(i)-v(i))>tol){
			throw Exception("Test not passed","CG failed.");
		}
	}
	std::cout << "All tests passed";
	
}

int main(int argc, char* argv[])
{

	try{
		testfunc();
	}
	catch  (Exception &ex)
	{
		ex.DebugPrint();
	}
}

