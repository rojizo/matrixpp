//
//  matrix.hpp
//  bddb
//
//  Created by Álvaro Lozano Rojo on 14/5/18.
//  Copyright © 2018 Álvaro Lozano Rojo. All rights reserved.
//

#include <memory>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <type_traits>

#ifndef matrix_h
#define matrix_h


template<class T>
class dumbdownvector { // A vector who does not care about initialization... for internal use
public:
    dumbdownvector() : _size{0}, data(nullptr) {}
    dumbdownvector(const size_t size) : _size{size}, data(new T[size]) {}
    dumbdownvector(const size_t size, const T& val) : _size{size}, data(new T[size]) {
        std::fill(begin(), end(), val);
    }
    dumbdownvector(const dumbdownvector& other) : _size{other._size}, data(new T[_size]) {
        std::copy(other.begin(), other.end(), begin());
    }
    dumbdownvector(dumbdownvector&& other) : _size{other._size}, data(std::move(other.data)) { }
    
    // Assign operators
    dumbdownvector<T>& operator=(dumbdownvector<T>&& rhs) {
        _size = rhs._size;
        data = std::move(rhs.data);
        return *this;
    }
    dumbdownvector<T>& operator=(const dumbdownvector<T>& rhs) {
        _size = rhs._size;
        std::copy(rhs.begin(), rhs.end(), begin());
        return *this;
    }
    
    // Iterators
    const T* begin() const {
        return data.get();
    }
    const T* end() const {
        return data.get() + _size;
    }
    T* begin() {
        return data.get();
    }
    T* end() {
        return data.get() + _size;
    }
    
    size_t size() const {
        return _size;
    }
    
protected:
    size_t _size;
    std::unique_ptr<T[]> data;
};

template<class T>
class Matrix : protected std::conditional<std::is_arithmetic<T>::value, dumbdownvector<T>, std::vector<T>>::type {

    // Base type name
    typedef typename std::conditional<std::is_arithmetic<T>::value, dumbdownvector<T>, std::vector<T>>::type BASE;

    
public:
    ////////////////////////////////////////////////////////////////////////////////
    //       Constructors
    ////////////////////////////////////////////////////////////////////////////////
    Matrix() : _cols{0}, _rows{0}, BASE() {
        std::cout << "contructor: default " << std::endl;
    }
    Matrix(const size_t rows, const size_t cols) : _rows{rows}, _cols{cols}, BASE(rows*cols) {
        std::cout << "contructor: r/c" << std::endl;
    }
    Matrix(const size_t rows, const size_t cols, const T& val) : _rows{rows}, _cols{cols}, BASE(rows*cols, val) {
        std::cout << "contructor: r/c/v" << std::endl;
    }
    Matrix(const Matrix<T>& A) : _rows{A._rows}, _cols{A._cols}, BASE(static_cast<const BASE&>(A)) {
        std::cout << "constructor: copy" << std::endl;
    }
    Matrix(Matrix<T>&& A) : _rows{A._rows}, _cols{A._cols}, BASE(static_cast<BASE&&>(A)) {
        std::cout << "constructor: move" << std::endl;
    }
    

//***************************************************************************
// Member methods...
//***************************************************************************

    ////////////////////////////////////////////////////////////////////////////////
    //       Col and row extractors
    ////////////////////////////////////////////////////////////////////////////////
    std::vector<T> row(const size_t row) const {
        std::vector<T> a(_cols);
        std::copy( BASE::begin() + row * _cols, BASE::begin() + (row+1) * _cols, a.begin());
        return a;
    }
    std::vector<T> col(const size_t col) const {
        std::vector<T> a(_rows);
        for(int i=0; i<_rows; i++) a[i] = operator()(i, col);
        return a;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //       Number of cols and rows
    ////////////////////////////////////////////////////////////////////////////////
    size_t cols() const {
        return _cols;
    }
    size_t rows() const {
        return _rows;
    }

    
//***************************************************************************
// Member operators...
//***************************************************************************
    
    ////////////////////////////////////////////////////////////////////////////////
    //       Assignament and move operators
    ////////////////////////////////////////////////////////////////////////////////
    Matrix<T>& operator=(Matrix<T>&& rhv) {
        _cols = rhv._cols;
        _rows = rhv._rows;
        BASE::operator=( static_cast<BASE&&>(rhv) );
        std::cout<< "assign: move" <<std::endl;
        return *this;
    }
    Matrix<T>& operator=(const Matrix<T>& rhv) {
        _cols = rhv._cols;
        _rows = rhv._rows;
        BASE::operator=( static_cast<const BASE&>(rhv) );
        std::cout<< "assign: copy" <<std::endl;
        return *this;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    //       'Accessing' operators
    ////////////////////////////////////////////////////////////////////////////////
    T& operator()(const size_t row, const size_t col) {
        return *(BASE::begin() + (row * _cols + col));
    }
    const T& operator()(const size_t row, const size_t col) const {
        return *(BASE::begin() + (row * _cols + col));
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    //       Inplace addition
    ////////////////////////////////////////////////////////////////////////////////
    Matrix<T>& operator+=(const Matrix<T>& rhs) {
        if((rhs._rows != _rows) or (rhs._cols != _cols)) throw "Addition: Dimensions mismatch";

        auto x = BASE::begin();
        auto b = rhs.begin();
        const auto end = BASE::end();
        while(x != end) *(x++) += *(b++);

        return *this;
    }
    
    
//***************************************************************************
// Static Member...
//***************************************************************************
    
    ////////////////////////////////////////////////////////////////////////////////
    //       Indentity construction
    ////////////////////////////////////////////////////////////////////////////////
    static Matrix<T> Identity(size_t n) {
        Matrix<T> I(n, n, T(0));
        for(size_t i=0; i<n; i++) I(i,i) = T(1);
        return I;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    //       Barreras & Peñas Solver
    ////////////////////////////////////////////////////////////////////////////////
    static std::vector<T> bpsolver(Matrix<T>& A, std::vector<T>& b){
        std::vector<T> r(A.rows);
        for(int i=0; i<A.rows; i++)
            for(int j=0; j<A._cols; j++)
                r[i] += A(i,j); // std::accumulate could be used...
        return bpsolver(A, b, r);
    }

    static std::vector<T> bpsolver(Matrix<T>& A, std::vector<T>& b, std::vector<T>& r){
        if(A.size() == 0) throw "BPSolver: Null matrix";
        if((A._cols != A._rows) or (b.size() != A._cols) or (r.size() != A._cols)) throw "BPSolver: Dimensions mismatch";
        
        const size_t N = b.size();

        // Return vector
        std::vector<T> X(N);

        // Memory allocation and initialization
        std::vector<size_t> p(N);
        std::iota(p.begin(), p.end(), 0);
        
        Matrix<T> L = Matrix<T>::Identity(N);
        Matrix<T> U(N, N, T(0));
        std::vector<T> D(N, T(0));
        std::vector<T> s(N, T(0)); // Off-diagonal col sums
        std::vector<T> h(N, T(0));
        
        // Initializations...
        for(int i = 0; i < N; i++){
            h[i] = A(i,i);
            for(int j = 0; j < N; j++)
                if(j != i) s[i] += A(j,i); // Off-diagonal col sums
        }
        
        
        // Look for the first permutation. We're assuming that such element exists,
        // since it is a diagonally dominant M-matrix!
        for(int i = 0; i < N; i++)
            if(h[i] >= -s[i]) {
                p[0] = i;
                p[i] = 0;
                break;
            }
        
        // this should be done after
        D[0] = h[p[0]];
        
        
        // Now, compute the LDU decomposition
        for(int k = 0; k < N-1; k++) {
            if(D[k] == T(0)) {
                for(int i = k+1; i < N; i++)
                    U(p[k],i) = L(p[i], p[k]) = T(0);
            } else {
                for(int i = k+1; i < N; i++) {
                    L(p[i], p[k]) = A(p[i], p[k]) / A(p[k], p[k]);
                    U(p[k], p[i]) = A(p[k], p[i]) / A(p[k], p[k]);
                    r[p[i]] -= L(p[i], p[k]) * r[p[k]];
                    h[p[i]] -= U(p[k], p[i]) * h[p[k]];
                    s[p[i]] -= U(p[k], p[i]) * s[p[k]];
                    if(L(p[i], p[k]) != T(0)) {
                        for(int j = k+1; j < N; j++) {
                            if(i != j) // Equivalent to p[i]!=p[j]
                                A(p[i], p[j]) -= L(p[i], p[k]) * A(p[k], p[j]);
                        }
                    }
                }
            }
            
            //Compute the new pivot... that is a permutation
            for(int i = k + 1 ; i < N; i++) {
                if(h[p[i]] >= -s[p[i]]) {
                    std::swap(p[i], p[k+1]);
                    break;
                }
            }
            
            //Final steps
            D[k+1] = r[p[k+1]];
            for(int i = k + 2; i < N; i++)
                D[k+1] -= A(p[k+1], p[i]);
            A(p[k+1],p[k+1]) = D[k+1];
        }
        
        // Add the ones to the diagonal of U. This step can be dropped
        for(int i=0; i < N; i++)
            U(p[i],p[i]) = T(1);
        
        // OK. Now, solve the system. If LDUx = b (with b reordered accordingly)
        // Let us start solving Ls=b with s = DUx. Since L has all entries less
        // than 0 (except the diagonal) there is no substractions in the algorithm.
        // We are reusing s... now it is not the sum of the off col-diagonals
        for(int i = 0; i < N; i++){
            s[p[i]] = b[p[i]];
            for(int j = 0; j < i; j++)
                if(L(p[i], p[j]) != T(0))
                    s[p[i]] -= L(p[i], p[j]) * s[p[j]];
        }
        
        // Now s = D s' with s' = Ux.
        for(int i = 0; i < N; i++)
            s[p[i]] /= D[i];
        
        // Finally, s = Ux so we can solve the system.
        for(int i = N-1; i >= 0; i--){
            X[p[i]] = s[p[i]];
            for(int j = N-1 ; j > i ; j-- )
                if( U(p[i], p[j]) != T(0))
                    X[p[i]] -= U(p[i], p[j]) * X[p[j]];
        }
        
        return X;
    }

    
    ////////////////////////////////////////////////////////////////////////////////
    //       Solve
    ////////////////////////////////////////////////////////////////////////////////
    static size_t gaussian_solve(Matrix<T>& A, std::vector<T> &b) {
        if(A.size() == 0) throw "Gaussian solve: Null matrix";
        if( (A._cols != A._rows) or (b.size() != A._cols) ) throw "Gaussian solve: Dimensions mismatch";
        
        size_t N = b.size();
        
        std::vector<size_t> p(N); // equations permutation
        std::iota(p.begin(), p.end(), 0); // filled from 0 to N-1
        
        // Triangulate!
        for(int i=0; i<N-1; i++) {
            // Search for the first non-zero element
            bool pivotFound = false;
            for(int j=i; j<N; j++)
                if( A(p[j],i) != T(0) ){
                    // Found... swap equations
                    if(i != j) // Equivalent to p[i] == p[j]
                        std::swap(p[j], p[i]);
                    pivotFound = true;
                    break;
                }
            
            if(not pivotFound) // There is some short of indetermination... report it!
                return N-i;
            
            // Now, elimination!
            auto& pivot = A(p[i], i);
            // Normalize i-esime equation
            for(int j=i+1; j<N; j++) A(p[i],j) /= pivot;
            // and the independent term
            b[p[i]] /= pivot;
            A(p[i],i) = T(1); // There is no need of doing that computation
            
            // Now, substract row i to the nexts
            for(int j=i+1; j<N; j++) {
                auto& pivot = A(p[j], i);
                if(pivot == 0) continue;
                for(int k=i+1; k<N; k++)
                    A(p[j], k) -= A(p[i], k) * pivot;
                b[p[j]] -= b[p[i]] * pivot;
                pivot = T(0); // It is zero after all
            }
        }
        
        // Last one!
        if(A(p[N-1], N-1) == T(0))
            // There is some short of indetermination... report it!
            return 1;
        
        // Ok... normalize equation
        b[p[N-1]] /= A(p[N-1], N-1);
        A(p[N-1], N-1) = T(1);
        
        // Once triangulate... we can use backward substitution
        for(int i=N-1; i>0; i--)
            for(int j=i-1; j>= 0; j--)
                b[p[j]] -= b[p[i]] * A(p[j], i);
        
        return 0;
    }
    
    

    
protected:
    size_t _rows;
    size_t _cols;
    
    
//***************************************************************************
// Non member operators...
//***************************************************************************
    
    ////////////////////////////////////////////////////////////////////////////////
    //       Output to stream
    ////////////////////////////////////////////////////////////////////////////////
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& A) {
        // Compute the longest thing to print
        std::vector<std::string> thingsToPrint(A.size());
        
        auto its = thingsToPrint.begin();
        size_t max = 0;
        for(const auto& a : A) {
            std::stringstream ss;
            ss << a <<std::flush;
            (*its) = ss.str();
            if(max < (*its).size()) max = (*its).size();
            its++;
        }
        
        max += 2; // Leave some space
        its = thingsToPrint.begin();
        for(int i=0; i<A.rows(); i++) {
            for(int j=0; j<A.cols(); j++) {
                os.width(max);
                os << std::right << *(its++);
            }
            os << std::endl;
        }
        
        return os;
    }

    
    ////////////////////////////////////////////////////////////////////////////////
    //       Scalar multiplication
    ////////////////////////////////////////////////////////////////////////////////
    // For arithmetic types only:
    template<typename U>
    friend Matrix<T> operator*(const typename std::enable_if<std::is_arithmetic<U>::value>::type& b, const Matrix<T> &A) {
        return A*T(b);
    }
    template<typename U>
    friend Matrix<T> operator*(const Matrix<T> &A, const typename std::enable_if<std::is_arithmetic<U>::value>::type& b) {
        return A*T(b);
    }
    template<typename U>
    friend Matrix<T> operator*(const typename std::enable_if<std::is_arithmetic<U>::value>::type& b, const Matrix<T> &&A) {
        return std::move(A)*T(b);
    }
    template<typename U>
    friend Matrix<T> operator*(const Matrix<T> &&A, const typename std::enable_if<std::is_arithmetic<U>::value>::type& b) {
        return std::move(A)*T(b);
    }

    // For the current "type" T
    friend Matrix<T> operator*(const T& b, Matrix<T> &&A) {
        return std::move(A)*b;
    }
    friend Matrix<T> operator*(Matrix<T> &&A, const T& b) {
        Matrix<T> B(std::move(A));
        for(auto& x : B) x *= b;
        return B;
    }
    friend Matrix<T> operator*(const T& b, const Matrix<T> &A) {
        return A*b;
    }
    friend Matrix<T> operator*(const Matrix<T>& A, const T& b) { // TODO : use a single real funcion w/ rvalues
        Matrix<T> B(A);
        for(auto& x : B) x *= b;
        return B;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    //       Matrix multiplication
    ////////////////////////////////////////////////////////////////////////////////
    friend Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B) {
        if(A._cols != B._rows) throw "Product: Dimensions mismatch";
        Matrix<T> AB(A._rows, B._cols, T(0));
        for(int i=0; i<A._rows; i++)
            for(int j=0; j<B._cols; j++)
                for(int k=0; k<A._cols; k++)
                    AB(i,j) += A(i,k) * B(k,j);
        return AB;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    //       Matrix "oposite"
    ////////////////////////////////////////////////////////////////////////////////
    friend Matrix<T> operator-(const Matrix<T>& A) {
        Matrix<T> mA(A._cols, A._rows);
        std::transform(A.begin(), A.end(), mA.begin(), [](const T& x)->T { return x; });
    };
    friend Matrix<T> operator-(const Matrix<T>&& A) {
        Matrix<T> mA(std::move(A));
        std::for_each(mA.begin(), mA.end(), [](T& x){ x = -x; });
    };
    
    // Matrix Substration
    friend Matrix<T> operator-(const Matrix<T>&& A, const Matrix<T> &&B) {
        return std::move(A) - B;
    }
    friend Matrix<T> operator-(const Matrix<T>& A, Matrix<T> &&B) {
        if((A._rows != B._rows) or (A._cols != B._cols)) throw "Substraction: Dimensions mismatch";
        
        Matrix AB(std::move(B));
        std::transform(A.begin(), A.end(), AB.begin(), AB.begin(), [](const T& a, T& b){ return a-b; });
        
        return AB;
    }
    friend Matrix<T> operator-(Matrix<T>&& A, const Matrix<T> &B) {
        if((A._rows != B._rows) or (A._cols != B._cols)) throw "Substraction: Dimensions mismatch";
        
        Matrix AB(std::move(A));
        
        auto b = B.begin();
        for(auto& x : AB)
            x -= *(b++);
        
        return AB;
    }
    friend Matrix<T> operator-(const Matrix<T>& A, const Matrix<T> &B) {
        if((A._rows != B._rows) or (A._cols != B._cols)) throw "Substraction: Dimensions mismatch";
        
        Matrix<T> AB(A._rows, A._cols);
        std::transform(A.begin(), A.end(), A.begin(), AB.begin(), [](const T& a, T& b){ return a-b; });
        
        return AB;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //       Matrix Addition
    ////////////////////////////////////////////////////////////////////////////////
    friend Matrix<T> operator+(Matrix<T>&& B, Matrix<T> &&A) { // For the ambiguity
        return std::move(A) + B;
    }
    friend Matrix<T> operator+(const Matrix<T>& B, Matrix<T> &&A) {
        return std::move(A) + B;
    }
    friend Matrix<T> operator+(Matrix<T> &&A, const Matrix<T>& B) {
        if((A._rows != B._rows) or (A._cols != B._cols)) throw "Addition: Dimensions mismatch";
        
        Matrix AB(std::move(A));
        auto x = AB.begin();
        auto b = B.begin();
        while(x != AB.end()) *(x++) += *(b++);

        return AB;
    }
    friend Matrix<T> operator+(const Matrix<T> &A, const Matrix<T>& B) {
        if((A._rows != B._rows) or (A._cols != B._cols)) throw "Addition: Dimensions mismatch";

        Matrix<T> AB(A._rows, A._cols);        
        std::transform(A.begin(), A.end(), A.begin(), AB.begin(), [](const T& a, const T& b)->T { return a + b; });

        return AB;
    }
    

};

        
        
#endif /* matrix_h */
