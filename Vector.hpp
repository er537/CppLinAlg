#ifndef VECTORDEF
#define VECTORDEF



//  **********************
//  *  Class of vectors  *
//  **********************


//  Class written in such a way that code similar to Matlab
//  code may be written


#include <cmath>
#include "Exception.hpp"//  This class throws errors using the class "error"
#include "Matrix.hpp"

class Matrix;

class Vector
{
private:
   // member variables
   double* mData;   // data stored in vector
   int mSize;      // size of vector

public:
   // constructors
   // No default constructor
   // overridden copy constructor
   Vector();
   Vector(const Vector& v1);
   // construct vector of given size
   Vector(int sizeVal);

   // destructor
   ~Vector();


   // All "friend" external operators and functions are declared as friend inside the class (here)
   // but their actual prototype definitions occur outside the class.
   // Binary operators
   friend Vector operator+(const Vector& v1, const Vector& v2);
   friend Vector operator-(const Vector& v1, const Vector& v2);
   friend double operator*(const Vector& v1, const Vector& v2); //vector . vector
   friend Vector operator*(const Vector& v, const double& a); //vector x scaler
   friend Vector operator*(const double& a, const Vector& v); //scaler x vector
   friend Vector operator*(const Matrix& m, const Vector& v); //matrix x vector
   friend Vector operator*(const Vector& v, const Matrix& m); //vector transpose x matrix
   friend Vector operator/(const Vector& v, const double& a); // vector/scaler
   friend Vector operator/(Vector& b, Matrix& A); //Solve linear system eg 'backslash' operator
   // Unary operator
   friend Vector operator-(const Vector& v);
   friend Vector backSub(Matrix& m);
   friend Vector GMRES(Matrix& A, Vector& b, Vector& x_0, const double& tol, const int& max_iter);
   friend Vector GMRES(Matrix& A, Vector& b, const double& tol, const int& max_iter);
   friend Vector CG(Matrix& A, Vector& b, const Vector& x_0, const double& tol, const int& max_iter);
   friend Vector CG(Matrix& A, Vector& b, const double& tol, const int& max_iter);
   friend Vector gausElim(Matrix& A, Vector& b);
   friend bool compare(const Vector& v1, const Vector& v2);

   //other operators
   //assignment
   Vector& operator=(const Vector& v);
   //indexing
   double& operator()(int i);
   //output
   friend std::ostream& operator<<(std::ostream& output, const Vector& v);

   //norm (as a member method)
   double norm(int p=2) const;
   // functions that are friends
   friend double norm(Vector& v, int p);
   friend int length(const Vector& v);
   friend Vector resize(Vector& v, int n);
   friend Matrix reshape(const Matrix& M, const int& m, const int& n);
};


// All "friend" external operators and functions are declared as friend inside the class
// but their actual prototype definitions occur outside the class (here).
// Binary operators

Vector operator+(const Vector& v1, const Vector& v2);
Vector operator-(const Vector& v1, const Vector& v2);
double operator*(const Vector& v1, const Vector& v2);
Vector operator*(const Vector& v, const double& a);
Vector operator*(const double& a, const Vector& v);
Vector operator*(const Matrix& m, const Vector& v);
Vector operator*(const Vector& v, const Matrix& m);
Vector operator/(const Vector& v, const double& a);
Vector GMRES(Matrix& A, Vector& b, Vector& x_0, const double& tol, const int& max_iter);
Vector GMRES(Matrix& A, Vector& b, const double& tol, const int& max_iter);
Vector CG(Matrix& A, Vector& b, const double& tol, const int& max_iter);
Vector CG(Matrix& A, Vector& b, const Vector& x_0, const double& tol, const int& max_iter);
Vector gausElim(Matrix& A, Vector& b);
// Unary operator
Vector operator-(const Vector& v);
Vector resize(Vector& v, int n);
Matrix reshape(const Matrix& M, const int& m, const int& n);
bool compare(const Vector& v1, const Vector& v2);

// function prototypes
double norm(Vector& v, int p=2);
// Prototype signature of length() friend function
int length(const Vector& v);


#endif
