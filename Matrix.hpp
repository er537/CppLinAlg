#ifndef MATRIXDEF
#define MATRIXDEF



//  **********************
//  *  Class of matrix  *
//  **********************


//  Class written in such a way that code similar to Matlab
//  code may be written


#include <cmath>
#include "Exception.hpp"//  This class throws errors using the class "error"
#include "Vector.hpp"


class Vector;

class Matrix
{
private:
   // member variables
   double** mData;   // data stored in vector
   int mNumRow;      // number of rows
   int mNumCol;		 // number of columns
   //the following methods should only be called by operator/,
   //not as stand alone methods
   void swap_row(Matrix& m, int i, int j);
   int rowReduce(Matrix& m);
   void subMatrix(const Matrix& temp, const int& q, const int& n); //used by determinant function

public:
   // constructors
   //default
   Matrix();
   // overridden copy constructor
   Matrix(const Matrix& m1);
   // construct vector of given size
   Matrix(int sizeRow, int sizeCol);

   // destructor
   ~Matrix();


   // All "friend" external operators and functions are declared as friend inside the class (here)
   // but their actual prototype definitions occur outside the class.
   // Binary operators
   friend Matrix operator+(const Matrix& m1, const Matrix& m2);
   friend Matrix operator-(const Matrix& m1, const Matrix& m2);
   friend Matrix operator*(const Matrix& m1, const Matrix& m2); //matrix x matrix
   friend Matrix operator*(const Matrix& m, const double& a); //matrix x scaler
   friend Matrix operator*(const double& a, const Matrix& m); //scaler x matrix
   friend Matrix operator/(const Matrix& m, const double& a); // matrix/scaler
   friend Vector operator/(Vector& b, Matrix& A); //Solve linear system eg 'backslash' operator
   friend Matrix operator/(const Matrix& b, const Matrix& A);
   friend int find_pivot(Matrix A, int column);
   friend Vector backSub(Matrix& m); //used by operator/
   friend Vector operator*(const Matrix& m, const Vector& v); //matrix x vector
   friend Vector operator*(const Vector& v, const Matrix& m); //vector transpose x matrix
   friend int determinant(const Matrix& m, const int& n);
   friend Matrix resize(Matrix& M, const int& m, const int& n);
   friend Matrix reshape(const Matrix& M, const int& m, const int& n);
   friend Matrix extend(Matrix& M, int m, int n);
   friend Matrix eye(int n);
   friend Vector GMRES(Matrix& A, Vector& b, Vector& x_0, const double& tol, const int& max_iter);
   friend Vector GMRES(Matrix& A, Vector& b, const double& tol, const int& max_iter);
   friend Vector CG(Matrix& A, Vector& b, const Vector& x_0, const double& tol, const int& max_iter);
   friend Vector CG(Matrix& A, Vector& b, const double& tol, const int& max_iter);
   friend Vector gausElim(Matrix& A,  Vector& b);
   friend bool compare(const Matrix& m1,const Matrix& m2 );
   friend bool checkSymPos(Matrix& A);

   // Unary operator
   friend Matrix operator-(const Matrix& m);

   //other operators
   //assignment
   Matrix& operator=(const Matrix& m);
   Matrix transpose();
   //indexing
   double& operator()(int i,int j);
   //output
   friend std::ostream& operator<<(std::ostream& output, const Matrix& m);

   friend Vector size(const Matrix& m); //size of matrix

};


// All "friend" external operators and functions are declared as friend inside the class
// but their actual prototype definitions occur outside the class (here).
// Binary operators

Matrix operator+(const Matrix& m1, const Matrix& m2);
Matrix operator-(const Matrix& m1, const Matrix& m2);
Matrix operator*(const Matrix& m1, const Matrix& m2);
Matrix operator*(const Matrix& m, const double& a);
Matrix operator*(const Matrix& m, const double& a);
Vector operator*(const Matrix& m, const Vector& v);
Vector operator*(const Vector& v, const Matrix& m);
Matrix operator/(const Matrix& m, const double& a);
Vector operator/(Vector& b, Matrix& A); //backslash
Matrix operator/(const Matrix& b, const Matrix& A);
Vector gausElim(Matrix& A, const Vector& b);
int find_pivot(Matrix A, int column);
Vector GMRES(Matrix& A, Vector& b, Vector& x_0, const double& tol=0.000001, const int& max_iter=10000);
Vector GMRES(Matrix& A, Vector& b, const double& tol=0.000001, const int& max_iter=10000);
Vector GMRES(Matrix& A, Vector& b, const int& restart, const double& tol, const int& max_iter);
Vector backSub(Matrix& m);
// Unary operator
Matrix operator-(const Matrix& m);
int determinant(const Matrix& m, const int& n);
Matrix reshape(const Matrix& M, const int& m, const int& n);
Matrix resize(Matrix& M, const int& m, const int& n);
Matrix eye(int n);
bool compare(const Matrix& m1,const Matrix& m2 );

std::ostream& operator<<(std::ostream& output, const Matrix& m);
Vector size(const Matrix& m);

Vector CG(Matrix& A, Vector& b, const Vector& x_0, const double& tol=0.000001, const int& max_iter=1000);
Vector CG(Matrix& A, Vector& b, const double& tol=0.000001, const int& max_iter=1000);
bool checkSymPos(Matrix& A);


#endif

