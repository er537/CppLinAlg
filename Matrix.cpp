#include <iostream>
#include <fstream>
#include "Matrix.hpp"
#include <chrono>
#include <algorithm>
using namespace std::chrono;

class Vector;

//constructor to construct empty object with 0 rows and 0 columns
Matrix::Matrix()
{
	mNumRow=0;
	mNumCol=0;
	mData = NULL;
}



// constructor that creates matrix of given size with
// double precision entries all initially set to zero
Matrix::Matrix(int sizeRow, int sizeCol)
{

  mNumRow = sizeRow;
  mNumCol = sizeCol;
  mData = new double* [mNumRow]; //double** mData

  for (int i=0; i<mNumRow; i++){
	  mData[i] = new double[mNumCol];
  }

  for (int i=0; i<mNumRow; i++){
	  for (int j=0; j<mNumCol;j++)
	  {
		  mData[i][j] = 0.0;
	  }
  }

}



// copy constructor - creates matrix with the same entries as m1
Matrix::Matrix(const Matrix& m1)
{
  mNumRow = m1.mNumRow;
  mNumCol = m1.mNumCol;

  mData = new double* [mNumRow];
  for (int i=0; i<mNumRow; i++){
  	  mData[i] = new double[mNumCol];
    }

  for (int i=0; i<mNumRow; i++){
  	  for (int j=0; j<mNumCol;j++)
  	  {
  		  mData[i][j] = m1.mData[i][j];
  	  }
  }


}



// destructor - deletes mData
Matrix::~Matrix()
{
	if ( mNumRow > 0 || mNumCol > 0 ){
		for (int i=0; i<mNumRow; i++){
			delete[] mData[i];
		}
		delete[] mData;
	}
}




//overlaod +operator
Matrix operator+(const Matrix& m1, const Matrix& m2){

  if (m1.mNumRow != m2.mNumRow | m1.mNumCol != m2.mNumCol){
	  std::cerr<<"matrix add - matrices are a different size\n";
	  std::cerr<<"extra entries of smaller matrix assumed to be 0.0\n";
  }

  int r,c;

  //  set r,c to be the largest number of rows,cols and create a matrix
  //  of that size to be returned
  if (m1.mNumRow > m2.mNumRow)
    {
      r = m1.mNumRow;
    }
  else
    {
      r = m2.mNumRow;
    }
  if (m1.mNumCol > m2.mNumCol)
      {
        c = m1.mNumCol;
      }
    else
      {
        c = m2.mNumCol;
      }


  Matrix w(r,c);
  Matrix temp1(r,c); //Temporary matrix of size (r,c) containing m1 padded with zeros.
  //  eg if one matrix is smaller than the other assume missing entries are 0
  Matrix temp2(r,c); //Temporary matrix of size (r,c) containing m2 padded with zeros

  // setting extra elements of temp1 to 0
  for (int i=0;i<m1.mNumRow;i++){
	  for (int j=0;j<m1.mNumCol;j++){
		  temp1.mData[i][j] = m1.mData[i][j];
	  }
	  for (int j=m1.mNumCol;j<c;j++){
		  temp1.mData[i][j]= 0;
	  }
  }
  for (int i=m1.mNumRow;i<r;i++){
  	  for (int j=0;j<c;j++){
  		  temp1.mData[i][j] = 0;
  	  }
  }

  // setting extra elements of temp2 to 0
  for (int i=0;i<m2.mNumRow;i++){
  	  for (int j=0;j<m2.mNumCol;j++){
  		  temp2.mData[i][j] = m2.mData[i][j];
  	  }
  	  for (int j=m2.mNumCol;j<c;j++){
  		  temp2.mData[i][j]= 0;
  	  }
  }
  for (int i=m2.mNumRow;i<r;i++){
      for (int j=0;j<c;j++){
    	  temp2.mData[i][j] = 0;
      }
  }


  //  add the temp matrices
  for (int i=0;i<r;i++){
	  for (int j=0;j<c;j++){
		  w.mData[i][j]=temp1.mData[i][j]+temp2.mData[i][j];
	  }
  }

  return w;
}







// definition of - between two matrices
Matrix operator-(const Matrix& m1, const Matrix& m2){

	if (m1.mNumRow != m2.mNumRow | m1.mNumCol != m2.mNumCol){
		  std::cerr<<"matrix subtraction - matrices are a different size\n";
		  std::cerr<<"extra entries of smaller matrix assumed to be 0.0\n";
	  }

	  int r,c;

	  //  set r,c to be the largest number of rows,cols and create a matrix
	  //  of that size to be returned
	  if (m1.mNumRow > m2.mNumRow)
	    {
	      r = m1.mNumRow;
	    }
	  else
	    {
	      r = m2.mNumRow;
	    }
	  if (m1.mNumCol > m2.mNumCol)
	      {
	        c = m1.mNumCol;
	      }
	    else
	      {
	        c = m2.mNumCol;
	      }


	  Matrix w(r,c);
	  Matrix temp1(r,c); //Temporary matrix of size (r,c) containing m1 padded with zeros.
	  //  eg if one matrix is smaller than the other assume missing entries are 0
	  Matrix temp2(r,c); //Temporary matrix of size (r,c) containing m2 padded with zeros


	  // setting extra elements of temp1 to 0
	    for (int i=0;i<m1.mNumRow;i++){
	  	  for (int j=0;j<m1.mNumCol;j++){
	  		  temp1.mData[i][j] = m1.mData[i][j];
	  	  }
	  	  for (int j=m1.mNumCol;j<c;j++){
	  		  temp1.mData[i][j]= 0;
	  	  }
	    }
	    for (int i=m1.mNumRow;i<r;i++){
	    	  for (int j=0;j<c;j++){
	    		  temp1.mData[i][j] = 0;
	    	  }
	    }

	    // setting extra elements of temp2 to 0
	    for (int i=0;i<m2.mNumRow;i++){
	    	  for (int j=0;j<m2.mNumCol;j++){
	    		  temp2.mData[i][j] = m2.mData[i][j];
	    	  }
	    	  for (int j=m2.mNumCol;j<c;j++){
	    		  temp2.mData[i][j]= 0;
	    	  }
	    }
	    for (int i=m2.mNumRow;i<r;i++){
	        for (int j=0;j<c;j++){
	      	  temp2.mData[i][j] = 0;
	        }
	    }


	    //  subtract the temp matrices
	    for (int i=0;i<r;i++){
	  	  for (int j=0;j<c;j++){
	  		  w.mData[i][j]=temp1.mData[i][j]-temp2.mData[i][j];
	  	  }
	    }

	    return w;

}


// definition of matrix product between two matrices
Matrix operator*(const Matrix& m1, const Matrix& m2)
{

	//matrices should be the correct dimensions for multiplication
	//else throw
	if (m1.mNumCol != m2.mNumRow){
		throw Exception("size mismatch", "matrix multiplication operator - matrices must be of size (mxn) and (nxk)");
	}

	int r = m1.mNumRow; //rows of output matrix
	int c = m2.mNumCol; //columns of output matrix
	//initialise output matrix
	Matrix w(r,c);

	//multiply matrices
	for(int i = 0; i < r; i++){
		for(int j = 0; j < c; j++){
			for(int k = 0; k < m1.mNumCol; k++){
				w.mData[i][j] += m1.mData[i][k] * m2.mData[k][j];
			}
		}
	}

  return w;

}

Vector operator*(const Matrix& m, const Vector& v){
	if (m.mNumCol != v.mSize){
		throw Exception("size mismatch","Matrix, vector multiplication operator - number of columns of matrix must equal size of vector");
	}
	Vector w(m.mNumRow);
	for (int i=0;i<m.mNumRow;i++){
		for (int j=0;j<v.mSize;j++){
			w.mData[i]+=m.mData[i][j]*v.mData[j];
		}
	}
	return w;
}

Vector operator*(const Vector& v, const Matrix& m){

	//vector matrix multiplication. Vector is transposed before multiplication.
	if (m.mNumRow != v.mSize){
			throw Exception("size mismatch","Matrix, vector multiplication operator - num of rows of matrix must equal size of vector");
	}
	std::cerr<<"Vector has been transposed before multiplication.\n";
	Vector w(m.mNumCol);
	for (int i=0;i<m.mNumCol;i++){
		for (int j=0;j<m.mNumRow;j++){
			w.mData[i]+=v.mData[j]*m.mData[j][i];
		}
	}
	return w;
}

Matrix operator*(const Matrix& m, const double& a){

	//create matrix of the same size as m, with entries m[i][j]*a
	Matrix w(m.mNumRow,m.mNumCol);

	for (int i=0;i<m.mNumRow;i++){
		for (int j=0;j<m.mNumCol;j++){
			w.mData[i][j]= m.mData[i][j] * a;
		}
	}
	return w;
}








Matrix operator*(const double& a, const Matrix& m){

	//create matrix of the same size as m, with entries m[i][j]*a
	Matrix w(m.mNumRow,m.mNumCol);

	for (int i=0;i<m.mNumRow;i++){
		for (int j=0;j<m.mNumCol;j++){
			w.mData[i][j]= m.mData[i][j] * a;
		}
	}
	return w;
}




Matrix operator/(const Matrix& m, const double& a){

	//create matrix of the same size as m, with entries m[i][j]*a
	Matrix w(m.mNumRow,m.mNumCol);

	for (int i=0;i<m.mNumRow;i++){
		for (int j=0;j<m.mNumCol;j++){
			w.mData[i][j]= m.mData[i][j] / a;
		}
	}
	return w;
}








// definition of the unary operator -
Matrix operator-(const Matrix& m)
{

	//create a matrix w with entries equal to -m
	Matrix w(m.mNumRow,m.mNumCol);

	for (int i=0;i<m.mNumRow;i++){
		for (int j=0;j<m.mNumCol;j++){
			w.mData[i][j]= - m.mData[i][j];
		}
	}

	return w;
}





// definition of matrix operator =
Matrix& Matrix::operator=(const Matrix& m)
{
	if (this == &m) // Same memory address?
	{
		return *this;  // Yes, so skip assignment, and just return *this.
	}

//  check both matrices are the same size
//  if rhs matrix is too small, assume missing entries are 0
//  if rhs vector is too big, then throw

	if (m.mNumCol != mNumCol | m.mNumRow != mNumRow){
		//std::cerr << "size mismatch, matrix assignment operator - matrices have different sizes";
	}
	for (int i=0;i<mNumRow;i++){
		delete[] mData[i];
	}
	delete[] mData;
	mNumRow = m.mNumRow;
	mNumCol = m.mNumCol;
	mData = new double* [m.mNumRow]; //double** mData
	for (int i=0; i<m.mNumRow; i++){
		mData[i] = new double[m.mNumCol];
	}

	for (int i=0; i<m.mNumRow; i++){
	  for (int j=0; j<m.mNumCol;j++)
	  {
		  mData[i][j] = m.mData[i][j];
	  }
	}

	return *this;
}




// definition of matrix operator ()
// allows m.mData[i][j] to be written as m(i+1,j+1), as in Matlab and FORTRAN
double& Matrix::operator()(int i, int j)
{

  if (i < 1 | j  < 1)
    {
      throw Exception("out of range", "accessing matrix through () - index too small");
    }
  else if (i > mNumRow | j > mNumCol)
    {
      throw Exception("length mismatch",
		  "accessing matrix through () - index too high");
    }

  return mData[i-1][j-1];

}




std::ostream& operator<<(std::ostream& output, const Matrix& m) {
  output << "(";
  for (int i=0; i<m.mNumRow; i++){
	  for (int j=0; j<m.mNumCol;j++){
		  output <<  m.mData[i][j]<<",";
	  }
      if (i != m.mNumRow-1){
    	  output  << "\n ";
      }
      else{
    	  output  << ")";
      }
  }
  return output;  // for multiple << operators.
}



// return size of Matrix
Vector size(const Matrix& m)
{
	Vector w(2);
	w(1) = m.mNumRow;
	w(2) =  m.mNumCol;
    return w;
}




void Matrix::subMatrix(const Matrix& m, const int& q, const int& n) {
   int i = 0, j = 0;
   // filling the sub matrix
   for (int row = 1; row < n; row++) {
      for (int col = 0; col < n; col++) {
         // skipping if the current row or column is not equal to the current
         // element row and column
         if (col != q) {
            mData[i][j++] = m.mData[row][col];
            if (j == n - 1) {
               j = 0;
               i++;
            }
         }
      }
   }
}



int determinant(const Matrix& m, const int& n) {
/* Calculate the determinant of matrix m. Note you must input both the matrix and the number of rows it contains.*/

   int det = 0;
   static int count = 0;
   assert(m.mNumRow == m.mNumCol);
   if (count == 0) {
	   assert(n == m.mNumRow);
   }
   count++;
   if (n == 1) {
      return m.mData[0][0];
   }
   if (n == 2) {
      return (m.mData[0][0] * m.mData[1][1]) - (m.mData[0][1] * m.mData[1][0]);
   }
   Matrix temp(n-1,n-1);
   int sign = 1;
   for (int i = 0; i < n; i++) {
      temp.subMatrix(m, i, n);
      //std::cout << "temp=" << temp << "\n";
      det += sign * m.mData[0][i] * determinant(temp, n-1); //note than n decreases by 1 every time determinant is called.
      sign = -sign;
   }
   return det;
}

Vector operator/(Vector& b, Matrix& A){
	Vector x;
	if (A.mNumRow != A.mNumCol | A.mNumRow != b.mSize){
			throw Exception("Size mismatch", "Gaussian elimination can only be used on square systems and rhs vector must be of correct size");
	}
	if (determinant(A,A.mNumRow)==0){
		std::cerr<< "Matrix is singular" << "\n";
	}
	else{
		if (A.mNumRow<=10){
			x=gausElim(A,b);
		}
		else if (checkSymPos(A)){
			x=CG(A,b);
		}
		else{
			x=GMRES(A,b);
		}
		return x;
	}
}

bool checkSymPos(Matrix& A){
	//check matrix is symmetric
	for (int i=1;i<=A.mNumRow;i++){
		for (int j=i; j<=A.mNumCol; j++){
			if (A(i,j)!=A(j,i)){
				return false;
			}
		}
	}
	//check matrix is positive definite
	/*for (int i=1;i<=A.mNumRow;i++){
		Matrix temp = resize(A,i,i);
		if (determinant(temp,i)<0){
			return false;
		}
	}*/
	return true;
}


Vector gausElim(Matrix&A, Vector &b){
/*Use Gaussian elimination to linear system Ax=b.
 * If the size of A is larger than 100x100, CG or GMRES should be used instead.
 */
	/*if (A.mNumRow >= 100){
		throw Exception("Linear system too large to be solved by direct method", "Please use CG or GMRES.");
	}*/

	/*if (determinant(A, A.mNumRow )==0) {
		std::cerr << "Singular Matrix passed to backslash operator.\n";
	}*/

	if (A.mNumRow != A.mNumCol | A.mNumRow != b.mSize){
    	throw Exception("Size mismatch", "Gaussian elimination can only be used on square systems and rhs vector must be of correct size");
    }
	for (int i=2;i<=A.mNumRow;i++){
		for (int j=1;j<i;j++){
			if (A(i,j)!=0){
				//append an extra column to A containing b
				Matrix m(A); 
				m = resize(m,A.mNumRow, A.mNumCol+1); 
				for (int i=0;i<m.mNumRow;i++){
					m.mData[i][m.mNumCol-1] = b.mData[i];
				}
				//reduce m to r.e.f
				int singular_flag = m.rowReduce(m);

				// if matrix is singular
				if (singular_flag != -1){

					/* if the RHS of equation corresponding to
					zero row  is 0, * system has infinitely
					many solutions, else inconsistent*/
					if (m.mData[singular_flag][A.mNumRow]){ //CHECK!!!!!
						std::cerr << "Inconsistent System.";
					}
					else{
						std::cerr <<"May have infinitely many solutions.";
					}
					Vector x(1);
					x(1) = -1;
					return x(1);

				}
				else{
					// get solution to system using backward substitution
					return backSub(m);
				}
			
			}
		}
	}
	//if A already upper triangular skip straight to back substitution
	//append an extra column to A containing b
	Matrix m(A); 
	m = resize(m,A.mNumRow, A.mNumCol+1);
	for (int i=0;i<m.mNumRow;i++){
		m.mData[i][m.mNumCol-1] = b.mData[i];
	}
	//use backSub to get the solution
	return backSub(m);	
}

// function for elementary operation of swapping two rows
void Matrix::swap_row(Matrix& m, int i, int j)
{

	for (int k=0; k<=m.mNumRow; k++)
    {
        double temp = m.mData[i][k];
        m.mData[i][k] = m.mData[j][k];
        m.mData[j][k] = temp;
    }
}



// function to reduce matrix to r.e.f.
int Matrix::rowReduce(Matrix& m)
{
    int N = m.mNumRow;
	for (int k=0; k<N; k++)
    {
        // Initialize maximum value and index for pivot
        int i_max = k;
        int v_max = m.mData[i_max][k];

        /* find greater amplitude for pivot if any */
        for (int i = k+1; i < N; i++)
            if (abs(m.mData[i][k]) > v_max)
                v_max = m.mData[i][k], i_max = i;

        /* if a prinicipal diagonal element  is zero,
         * matrix is singular, and*/
        if (m.mData[k][i_max]==0){
            return k; // Matrix is singular
        }

        /* Swap the greatest value row with current row */
        if (i_max != k){
            swap_row(m, k, i_max);
        }


        for (int i=k+1; i<N; i++)
        {
            // factor f required to set m.mData[i,k] to 0
            double f = m.mData[i][k]/m.mData[k][k];

            // subtract fth multiple of corresponding kth row element
            for (int j=k+1; j<=N; j++){
                m.mData[i][j] -= m.mData[k][j]*f;
            }

            //lower triangular matrix elements are 0
            m.mData[i][k] = 0;
        }

    }
    return -1;
}



// function to calculate the values of the unknowns
Vector backSub(Matrix& m)
{
    Vector x(m.mNumRow);  // An array to store solution

    //Back substitute to find x
    for (int i = m.mNumRow-1; i >= 0; i--)
    {
        /* start with the RHS of the equation */
        x.mData[i] = m.mData[i][m.mNumRow]; //check!!!!

        /* Initialize j to i+1 since matrix is upper
           triangular*/
        for (int j=i+1; j<m.mNumRow; j++)
        {
            /* subtract all the lhs values
             * except the coefficient of the variable
             * whose value is being calculated */
            x.mData[i] -= m.mData[i][j]*x.mData[j];
        }

        /* divide the RHS by the coefficient of the
           unknown being calculated */
        x.mData[i] = (x.mData[i])/(m.mData[i][i]);
    }

    return x;
}


Vector CG(Matrix& A, Vector& b, const Vector& x_0, const double& tol, const int& max_iter){
/*Function uses CG to solve the linear system Ax=b where A is symmetric positive definite. x_0 is an initial guess.
 * tol and max_iter are the maximum tolerance and number of iterations respectively.*/

	//check system is square
	if (A.mNumRow != A.mNumCol | A.mNumRow != b.mSize){
	    throw Exception("Size mismatch", "Gaussian elimination can only be used on square systems and rhs vector must be of correct size");
	}

	for (int i=1;i<=A.mNumRow;i++){
		for (int j=1; j<=A.mNumCol; j++){
			if (A(i,j)!=A(j,i)){
				throw Exception("Matrix not symmetric.","CG can only be used on symmetric, positive definite matrices.");
			}
		}
	}

	for (int i=1;i<=A.mNumRow;i++){
		Matrix temp = resize(A,i,i);
		if (determinant(temp,i)<0){
			throw Exception("Matrix not positive definite.","CG can only be used on symmetric, positive definite matrices.");
		}
	}

	std::ofstream out("residuals.txt");
	assert(out.is_open());

	Vector r(length(b));
	r = b - A*x_0;
	out << r.norm() << "\t" << 0 << "\n";
	Vector p(length(b));
	p = r;
	Vector x(length(b));
	x = x_0;
	Vector r_old(length(b));
	double alpha, beta;

	for (int k=0; k<max_iter; k++){
		alpha = (r*r)/(p*(A*p));
		x = x + (alpha*p);
		r_old = r;
		r = r - (alpha*(A*p));
		out << r.norm() << "\t" << k+1 << "\n";

		if (r.norm() <= tol){
			return x;
		}
		beta = (r*r)/(r_old*r_old);
		p = r + (beta*p);
	}
	out.close();

	return x;
}


Vector CG(Matrix& A, Vector& b, const double& tol, const int& max_iter){
	//check system is square
	if (A.mNumRow != A.mNumCol | A.mNumRow != b.mSize){
	    throw Exception("Size mismatch", "Gaussian elimination can only be used on square systems and rhs vector must be of correct size");
	}

	//check matrix is symmetric
	for (int i=1;i<=A.mNumRow;i++){
		for (int j=i; j<=A.mNumCol; j++){
			if (A(i,j)!=A(j,i)){
				throw Exception("Matrix not symmetric.","CG can only be used on symmetric, positive definite matrices.");
			}
		}
	}
	//check matrix is positive definite
	/*for (int i=1;i<=A.mNumRow;i++){
		Matrix temp = resize(A,i,i);
		if (determinant(temp,i)<0){
			std::cout << temp;
			std::cout << determinant(temp,i);
			throw Exception("Matrix not positive definite.","CG can only be used on symmetric, positive definite matrices.");
		}
	}*/
	//open file to save residuals for plotting
	std::ofstream out("CGresiduals.txt");
	assert(out.is_open());

	Vector x(length(b)); //constructor initialises the 0 vector
	Vector r(length(b));
	r = b - A*x;
	out << r.norm() << "\t" << 0 << "\n"; //write initial residual to file
	Vector p(length(b));
	p = r;
	Vector r_old(length(b));
	double alpha, beta;

	for (int k=0; k<max_iter; k++){
		alpha = (r*r)/(p*(A*p));
		x = x + (alpha*p);
		r_old = r;
		r = r - (alpha*(A*p));
		out << r.norm() << "\t" << k+1 << "\n"; //write new residual to file
		if (r.norm() <= tol){
			return x;
		}
		beta = (r*r)/(r_old*r_old);
		p = r + (beta*p);
	}
	out.close(); //close file
	return x;
}

Matrix reshape(const Matrix& M, const int& m, const int& n){
/*function to take a matrix M and reshape it to mxn.
reshape returns a matrix of size mxn, taking entries
sequentially along the rows of M, then filling the new matrix row by row.
This is analogous to Matlab's reshape function. */
	assert((m*n) >= (M.mNumRow*M.mNumCol));
	Vector temp(m*n);
	int k=0;
	Matrix Mout(m,n);
	//store entries in temporary vector
	for (int i=0;i<M.mNumRow;i++){
		for (int j=0; j<M.mNumCol; j++){
			temp.mData[k] = M.mData[i][j];
			k++;
		}
	}
	for (int i=k;i<(m*n);i++){
		temp.mData[i]=0;
	}

	k=0;
	for (int i=0;i<m;i++){
		for (int j=0;j<n;j++){
			Mout.mData[i][j]= temp.mData[k];
			k++;
		}
	}
	return Mout;
}

Matrix Matrix::transpose(){
	Matrix temp(mNumCol,mNumRow);
	for (int i=0;i<mNumRow;i++){
		for (int j=0;j<mNumCol;j++){
			temp.mData[j][i]=mData[i][j];
		}
	}
	return temp;
}

Matrix resize(Matrix& M, const int& m, const int& n){
/*function to take a matrix M and resize it to mxn.
	resize returns a matrix of size mxn, with the entires in the same place as M. Extra entries in the new matrix are 0,
	while extra entries in M are lost. This produces a warning. */
	Matrix Mout(m,n);
	int o,p;
	if (m<M.mNumRow | n<M.mNumCol){
		//std::cerr<< "Input matrix is larger than required size. Extra entries will be lost.";
	}
	if (m<M.mNumRow){
		o=m;
	}
	else{
		o=M.mNumRow;
	}
	if (n<M.mNumCol){
		p=n;
	}
	else{
		p=M.mNumCol;
	}
	for (int i=0;i<o;i++){
		for (int j=0;j<p;j++){
			Mout.mData[i][j] = M.mData[i][j];
		}
	}
	return Mout;
}



Matrix eye(int n){
//function to produce an nxn identity matrix
	Matrix m(n,n);
	for (int i=0;i<n;i++){
		m.mData[i][i]=1;
	}
	return m;
}


Vector GMRES(Matrix& A, Vector& b, const double& tol, const int& max_iter){
	//check system is square
	if (A.mNumRow != A.mNumCol | A.mNumRow != b.mSize){
		throw Exception("Size mismatch", "Gaussian elimination can only be used on square systems and rhs vector must be of correct size");
	}
	//open file to save residuals for plotting
	std::ofstream out("GMRESresiduals.txt");
	assert(out.is_open());
	
	Vector r0 = b;
	double normr0 = norm(r0);
	double res = norm(b);
	out << res << "\t" << 0 << "\n"; //write initial residual to file
	int k = 1;
	Vector r0e(1);
	r0e(1) = normr0;
	Vector v = r0/normr0;
	Vector vj(length(v));
	Matrix V(length(v),1);
	Matrix G=eye(2);
	Matrix H(1,1);
	Vector w,c;
	Matrix Htemp,J,R;
	int max = std::min(max_iter,A.mNumRow);
	//initialise first column of V
	for (int i=0;i<V.mNumRow;i++){
		V.mData[i][0]=v.mData[i];
	}

	while (res>tol && k<=max){
		H = resize(H,k+1,k); //k+2 and k+1 because k=0 initially, but we want the matrix to be of size(2,1)
		w = A*v;
		for (int j=1;j<=k;j++){
			for (int i=1;i<=V.mNumRow;i++){
				vj(i) = V(i,j);

			}
			H(j,k)=vj*w;
			w = w - H(j,k)*vj;
		}
		H(k+1,k)= norm(w);
		v = w/norm(w);
		V = resize(V,V.mNumRow,k+1);

		//append new column to V
		for (int i=1;i<=V.mNumRow;i++){
			V(i,k+1)= v(i);
		}

		if (k==1){
			Htemp = H;
		}
		else{
			G = resize(G,k+1,k+1); //when k==1, J is 3x3
			G(k+1,k+1) = 1;
			Htemp= G*H;
		}
		J = eye(k-1);
		J = resize(J,k+1,k+1);
		J(k,k)=(Htemp(k,k))/(pow(pow(Htemp(k,k),2)+pow(Htemp(k+1,k),2),0.5));
		J(k,k+1)=(Htemp(k+1,k))/(pow(pow(Htemp(k,k),2)+pow(Htemp(k+1,k),2),0.5));
		J(k+1,k)=-(Htemp(k+1,k))/(pow(pow(Htemp(k,k),2)+pow(Htemp(k+1,k),2),0.5));
		J(k+1,k+1)=(Htemp(k,k))/(pow(pow(Htemp(k,k),2)+pow(Htemp(k+1,k),2),0.5));
		
		G = J*G;
		R = G*H;
		r0e = resize(r0e,k+1);
		c = G*r0e;
		res = fabs(c(k+1));
		out << res << "\t" << k+1 << "\n"; //write new residual to file
		k++;
	}
	out.close(); //close file
	
	//fix rounding errors
	R = resize(R,R.mNumRow-1,R.mNumCol);
	for (int i=1; i<=R.mNumRow;i++){
		for (int j=1;j<=R.mNumCol;j++){
			if (abs(R(i,j))<0.000001){
				R(i,j)=0;
			}
		}
	}
	//solve system and return x
	c = resize(c,R.mNumRow);
	Vector y = c/R;
	V = resize(V,V.mNumRow,V.mNumCol-1);
	Vector x = (V*y);
	return x;
}

Vector GMRES(Matrix& A, Vector& b, const int& restart, const double& tol, const int& max_iter){
	//open file to save residuals for plotting
	std::ofstream out("GMRESrestart.txt");
	assert(out.is_open());
	out << norm(b) << "\t" << 0 << "\n"; //write initial residual to file
	int k =1;
	Vector x = GMRES(A,b,tol,max_iter);
	Vector res;
	res = (A*x) - b;
	out << norm(res) << "\t" << k*max_iter << "\n"; //write initial residual to file
	while (norm(res)>tol && k<restart){
		x = GMRES(A,b,x,tol,max_iter);
		res = (A*x) - b;
		out << norm(res) << "\t" << (k+1)*max_iter << "\n"; //write initial residual to file
		k++;
	}
	return x;
}

Vector GMRES(Matrix& A, Vector& b, Vector& x_0, const double& tol, const int& max_iter){
/*this function uses the GMRES method to compute the matrix x
 * that solves the linear system Ax=b. x_0 is an initial guess,
 * tol is the maximum size of the residual and max_iter is the max number of iterations to perform*/
	//check system is square
	if (A.mNumRow != A.mNumCol | A.mNumRow != b.mSize){
		throw Exception("Size mismatch", "Gaussian elimination can only be used on square systems and rhs vector must be of correct size");
	}

	Vector r0 = b-(A*x_0);
	double normr0 = norm(r0);
	double res = 10;
	int k = 1; //k is the rank of the Krylov subspace
	Vector r0e(1);
	r0e(1) = normr0;
	Vector q = r0/normr0;
	Vector qj(length(q));
	Matrix Q(length(q),1); //stores all Arnoldi vectors
	Matrix G=eye(2); //stores givens rotation of current step
	Matrix H(1,1);
	Vector v,c;
	Matrix Htemp,J,R; //J stores all previous givens rotations, R is upper triangular matrix in QR factorization.
	int max = std::min(max_iter,A.mNumRow);
	//initialise first column of V
	for (int i=0;i<Q.mNumRow;i++){
		Q.mData[i][0]=q.mData[i];
	}

	while (res>tol && k<=max){
		//arnoldi iteration k 
		H = resize(H,k+1,k); //upper Hessenberg matrix H
		v = A*q;
		for (int j=1;j<=k;j++){
			for (int i=1;i<=Q.mNumRow;i++){
				qj(i) = Q(i,j);

			}
			H(j,k)=qj*v;
			v = v - H(j,k)*qj;
		}

		H(k+1,k)= norm(v);
		q = v/norm(v);
		Q = resize(Q,Q.mNumRow,k+1);

		//append new column to V
		for (int i=1;i<=Q.mNumRow;i++){
			Q(i,k+1)= q(i);
		}
		
		if (k==1){
			Htemp = H;
		}
		//compute and apply givens rotations
		else{
			G = resize(G,k+1,k+1); //when k==1, J is 3x3
			G(k+1,k+1) = 1;
			Htemp= G*H;
		}
		J = eye(k-1);
		J = resize(J,k+1,k+1);
		J(k,k)=(Htemp(k,k))/(pow(pow(Htemp(k,k),2)+pow(Htemp(k+1,k),2),0.5));
		J(k,k+1)=(Htemp(k+1,k))/(pow(pow(Htemp(k,k),2)+pow(Htemp(k+1,k),2),0.5));
		J(k+1,k)=-(Htemp(k+1,k))/(pow(pow(Htemp(k,k),2)+pow(Htemp(k+1,k),2),0.5));
		J(k+1,k+1)=(Htemp(k,k))/(pow(pow(Htemp(k,k),2)+pow(Htemp(k+1,k),2),0.5));
		
		G = J*G;
		R = G*H;
		r0e = resize(r0e,k+1);
		c = G*r0e;
		res = fabs(c(k+1));
		k++;
	}
	/*note that we only need to compute y and x once res<tol. 
	Computing this on every iteration would be unneccesarry and computationally expensive*/
	R = resize(R,R.mNumRow-1,R.mNumCol);
	c = resize(c,R.mNumRow);
	Vector y = c/R;
	Q = resize(Q,Q.mNumRow,Q.mNumCol-1);
	Vector x = x_0+(Q*y);
	return x;
}

bool compare(const Matrix& m1,const Matrix& m2 ){
	if (m1.mNumRow == m2.mNumRow && m1.mNumCol==m2.mNumCol){
		for (int i = 0; i < m1.mNumRow;i++){
			for (int j=0; j< m1.mNumCol; j++){
				if (m1.mData[i][j]!=m2.mData[i][j]){
					return false;
				}
			}
		}
		return true;
	}
	return false;
}












