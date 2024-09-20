#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <Eigen/Dense>

using namespace std;

typedef vector<vector<double>>  myMatrix;
const double epsilon = 1e-5;

void displaymyMatrix(const myMatrix& A) {
    for (auto i : A) {
        for (auto j : i) {
            cout << left << setw(12) << j << " ";
        }
        cout << endl;
    }
}

// Tìm ma trận chuyển vị của A
myMatrix findTransposeMatrix(const myMatrix& A) {
    int m = A.size();
    int n = A[0].size();
    myMatrix result(n, vector<double>(m));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}


void multiplyMatrices(myMatrix A, myMatrix B, myMatrix& result) {
    int m = A.size();
    int n = A[0].size();
    int k = B[0].size();

    // Khởi tạo ma trận nếu ma trận chưa khởi tạo
    if (result.size() == 0) {
        result = myMatrix(m, vector<double>(k));
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            double temp = 0;
            for (int l = 0; l < n; l++) {
                temp += A[i][l] * B[l][j];
            }
            result[i][j] = temp;
        }
    }
}

Eigen::MatrixXd convertToMatrixXd(const myMatrix& inputMatrix) {
    int rows = inputMatrix.size();
    int cols = inputMatrix[0].size();
    Eigen::MatrixXd result(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = inputMatrix[i][j];
        }
    }

    return result;
}

myMatrix convertToMatrix(const Eigen::MatrixXd& inputMatrix) {
    int rows = inputMatrix.rows();
    int cols = inputMatrix.cols();
    myMatrix result(rows, vector<double>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = inputMatrix(i, j);
        }
    }

    return result;
}

void SVD_Decomposition(myMatrix A) {
    int m = A.size();
    int n = A[0].size();

    // Tìm trị riêng của ma trận A^T.A
    myMatrix transposeA = findTransposeMatrix(A);
    myMatrix temp1;
    multiplyMatrices(transposeA, A, temp1);
    // Tìm trị riêng, vector riêng
    Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(convertToMatrixXd(temp1));
    Eigen::VectorXd _eigenValues = eigenSolver.eigenvalues().real();
    vector<double>eigenValues;

    int indexOfZeroEigenValue = 0;
    for (int i = 0; i < _eigenValues.size(); i++) {
        eigenValues.push_back(round(_eigenValues[i] * 1000) / 1000);
        if (eigenValues[i] == 0) {
            indexOfZeroEigenValue = i;
        }
    }

    // Tìm các vector riêng để xây dựng ma trận V
    Eigen::MatrixXd eigenVectors = eigenSolver.eigenvectors().real();
    myMatrix V = convertToMatrix(eigenVectors);
    // Trực giao các từng vector (từng cột) của V, ta tìm được ma trận V
    for (int j = 0; j < V[0].size(); j++) {
        double norm = 0;
        for (int i = 0; i < V.size(); i++) {
            norm += pow(V[i][j], 2);
        }
        norm = sqrt(norm);
        for (int i = 0; i < V.size(); i++) {
            V[i][j] /= norm;
            V[i][j] = round(V[i][j] * 1000) / 1000;
        }
    }

    // Hoán đổi vị trí để đưa lamda = 0 về cuối
    swap(eigenValues[indexOfZeroEigenValue], eigenValues[eigenValues.size() - 1]);

    // Hoán đổi cột của V để đưa vector riêng ứng với lamda = 0 về cuối
    for (int i = 0; i < V.size(); i++) {
        swap(V[i][indexOfZeroEigenValue], V[i][V[0].size() - 1]);
    }

    // Số hàng, cột của ma trận delta chính là số hàng, cột của ma trận A
    myMatrix delta = myMatrix(m, vector<double>(n, 0));
    int k = 0;
    for (int i = 0; i < eigenValues.size(); i++) {
        // Nếu m < n thì bỏ đi trị riêng ứng với lamda = 0 (ở cuối)
        if (m < n && i == eigenValues.size() - 1)
            continue;

        delta[k][k] = sqrt(eigenValues[i]);
        // Làm tròn 3 chữ số thập phân
        delta[k][k] = round(delta[k][k] * 1000) / 1000;
        k++;
    }

    // Tìm ma trận U là ma trận mxm
    // Cột thứ i của ma trận U được tính: ui= A.vi/sqrt(eignValues[i])
    myMatrix U(m, vector<double>(m));
    k = 0;
    for (int i = 0; i < eigenValues.size(); i++) {
        // Nếu m < n thì bỏ đi trị riêng ứng với lamda = 0 (ở cuối)
        if (m < n && i == eigenValues.size() - 1)
            continue;

        // Lấy vector vi từ cột i của ma trận V. vi thực chất là ma trận v.size() x 1
        myMatrix vi = myMatrix(V.size(), vector<double>(1));
        for (int j = 0; j < V.size(); j++)
            vi[j][0] = V[j][i];
        myMatrix ui;
        multiplyMatrices(A, vi, ui);
        for (int j = 0; j < A.size(); j++) {
            ui[j][0] /= sqrt(eigenValues[i]);
            U[j][k] = round(ui[j][0] * 1000) / 1000;
        }
        k++;
    }
    if (m > n) {
        // Cần tìm thêm một vector riêng ui của ma trận A.A^T khi trị riêng bằng 0
        myMatrix temp2;
        multiplyMatrices(A, transposeA, temp2);

        // Tìmm danh sách các trị riêng, vector riêng của ma trận A.A^T
        Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(convertToMatrixXd(temp2));
        Eigen::VectorXd _eigenValues = eigenSolver.eigenvalues().real();
        Eigen::MatrixXd eigenVectors = eigenSolver.eigenvectors().real();

        // Tìm vị trí của vector riêng ứng với lamda = 0
        int indexOfZeroEigenValue = 0;
        for (int i = 0; i < _eigenValues.size(); i++) {
            if (_eigenValues[i] == 0) {
                indexOfZeroEigenValue = i;
                break;
            }
        }

        myMatrix ui = myMatrix(m, vector<double>(1, 0));
        for (int i = 0; i < m; i++) {
            ui[i][0] = eigenVectors(i, 0);
        }

        // Trực giao ma trận ui
        double norm = 0;
        for (int i = 0; i < m; i++) {
            norm += pow(ui[i][0], 2);
        }
        norm = sqrt(norm);
        for (int i = 0; i < m; i++) {
            ui[i][0] /= norm;

            // Gán vào cột cuối cùng của ma trận U
            U[i][m - 1] = round(ui[i][0] * 1000) / 1000;
        }
    }
    cout << "Ma tran U:" << endl;
    displaymyMatrix(U);
    cout << "Ma tran delta:" << endl;
    displaymyMatrix(delta);
    cout << "Ma tran V:" << endl;
    displaymyMatrix(V);
}

int main() {
    fstream inputFile("input.txt", ios::in);
    if (!inputFile.is_open()) {
		cout << "Can't open input.txt" << endl;
		return 0;
	}

    int m, n;
    inputFile >> m;
    cout << "m = " << m << "\n";
    inputFile >> n;
    cout << "n = " << n << "\n";
    myMatrix A(m, vector<double>(n));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            inputFile >> A[i][j];
        }
    }
    inputFile.close();
    displaymyMatrix(A);
    cout << endl;
    SVD_Decomposition(A);

    // Đợi nhập 1 ký tự bất kỳ để kết thúc chương trình (tránh đóng cửa sổ ngay lập tức)
    char c;
    cin >> c;
    return 0;
}
