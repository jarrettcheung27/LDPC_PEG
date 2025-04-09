#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <Eigen/Sparse>
#include <python.h>
#include <iomanip>  // ���ڿ��������ʽ
#include <cstdlib>  // ʹ�� atoi() ת���ַ���������
#include <chrono>
#include <ctime>

using namespace std;
using namespace Eigen;

const string filePathData = "D:\\DeSP-main\\Data\\Cost_Optimization_result\\Fix_indexing_cost\\Data\\Simu\\Exp_Ri_MinimalCost.csv";
const string HmatrixPath = "D:\\DeSP-main\\Inbox\\Classic_PEG\\x64\\Debug\\Hmatrix_test.txt";
const int sequenceNum = 320; //DNA���г���/bit
const int iter = 30; //Iteration for LDCP decode.
//========================================Helper Function=======================================//
int mod2Add(int a, int b) {
    return (a + b) % 2;
}

int HammingDistance(const vector<int> a, const vector<int> b) {
   if (a.size() != b.size()) {
       throw invalid_argument("Vectors must be of the same length.");
   }
   int distance = 0;
   for (size_t i = 0; i < a.size(); ++i) {
       if (a[i] != b[i]) {
           distance++;
       }
   }
   return distance;
}
void CreatFile(string filename) {
}

class Duration {
public:
    chrono::system_clock::time_point simu_start_time; //start time of the simulation.
    chrono::steady_clock::time_point exp_start;//start time of the experiment.
    chrono::steady_clock::time_point exp_end; //end time if the experiment.
    
    void simuStart() {
        simu_start_time = chrono::system_clock::now(); //��¼���濪ʼ��ʱ��
    }
    void expStart() {
        exp_start = chrono::high_resolution_clock::now();//��¼����ʵ�鿪ʼ��ʱ��
    }

    void completionTime(int RepetitionRequired) {
        /* Args:
        *   RepetitionRequired: ���η�����Ҫ�ظ�ʵ��Ĵ���
        */
        exp_end = chrono::high_resolution_clock::now();//��¼����ʵ�������ʱ��
        chrono::duration<double> elapsed = exp_end - exp_start; //����ʵ��ĳ���ʱ��
        cout << "Time spent in this experiment: " << elapsed.count() << "s" << endl;
        double total_time = elapsed.count() * RepetitionRequired; //�����ܺ�ʱ
        auto estimated_end_time = simu_start_time + chrono::seconds(static_cast<long>(total_time)); //����Ԥ�������ʱ��
        time_t end_time_t = chrono::system_clock::to_time_t(estimated_end_time); //ת����������ʱ�������ʽ
        char end_time_str[26];
        ctime_s(end_time_str, sizeof(end_time_str), &end_time_t);
        cout << "Estimated time for simulation completion: " << end_time_str << "Duariton: "<< total_time/3600 << "h";
    }

    
    

};

//==================================Generate Random Message==================================//
class MessageGenerator {
private:
    vector<int> Msg;


public:
    MessageGenerator() {
        cout << "Generating message..." << endl;
    }
    vector<int> genMsg(const int MsgLength) {
        /*Generate random bit stream of length n*/
        srand(static_cast<unsigned int>(time(0))); // Set the seed of random number generator according to time.
        Msg.clear();
        Msg.reserve(MsgLength);
        for (int i = 0; i < MsgLength; ++i) {
            int bit = rand() % 2;
            Msg.push_back(bit);
        }
        return move(Msg);
    }

    void PrintMsg() {
        cout << "Msg: " << endl;
        for (int bit : Msg) {
            cout << bit;
        }
        cout << endl;
    }
};

//==========================================ENCODER=========================================//
// ����ϡ��У�����H����
class CheckMatrix {
public:
    int M;       // The number of check nodes M
    int N;       // Code length N
    vector<vector<int>> H;// H matrix read from the file (Generate by PEG Algorithm.)
    vector<int> Permutation;        // ��¼�н�����Ϣ���任��H�е�i����ԭH�����е�����
    string filename;

    // Read the H matrix from file.

    void read_H_matrix(string filename) {
        cout << "Reading H matrix..." << endl;
        ifstream file(filename);
        if (!file.is_open()) {
            cerr << "Error: Could not open file " << filename << endl;
            exit(1);
        }
        string line;
        getline(file, line);
        istringstream iss(line);
        iss >> N >> M;

        for (int i = 0; i < M; ++i) {
            getline(file, line);
            istringstream row_iss(line);
            vector<int> row;
            for (int j = 0; j < N; ++j) {
                int val;
                row_iss >> val;
                if (val != 0 && val != 1) {
                    cerr << "Error: Hmatrix contains non-binary value " << val << endl;
                    exit(1);
                }
                row.push_back(val);
            }
            H.push_back(row); //H matrix for decoding.
        }
    }
    // ��˹��Ԫ��������GF(2)�ϣ�����H������ԪΪϵͳ����ʽ [P I]
    // ͬʱ��¼ÿ���н�����Ϣ�� Permutation �У���������ʱ�ָ�ԭʼ��Ϣ˳��
    void Gaussian_Elimination() {
        cout << "Processing Gaussian_Elimination" << endl;
        // ��ʼ�� Permutation Ϊ������У��� Permutation[i] = i
        Permutation.resize(N);
        for (int i = 0; i < N; ++i) {
            Permutation[i] = i;
        }

        // ����ÿһ�� r��Ŀ���ǽ��Ҳ� M �й���Ϊ��λ����
        // ������ r��Ŀ���� target = N - M + r
        for (int r = 0; r < M; ++r) {
            int target = N - M + r;
            int pivot_col = -1;
            // �ڵ� r ����Ѱ��һ��1��Ϊ��Ԫ���ɴ���������Ѱ�ң�
            for (int j = 0; j < N; j++) {
                if (H[r][j] == 1) {
                    pivot_col = j;
                    break;
                }
            }
            if (pivot_col == -1) {
                // ��ǰ����1�������ȿ��ܲ��㣬�޷�תΪ��ȫϵͳ����ʽ
                cerr << "Warning: Row " << r << " has no pivot. The matrix may be rank deficient." << endl;
                continue;
            }
            // ����ҵ�����Ԫ����Ŀ��λ�ã��򽻻��У�����Ԫ�Ƶ� target ��
            if (pivot_col != target) {
                for (int i = 0; i < M; i++) {
                    swap(H[i][pivot_col], H[i][target]);
                }
                // ͬʱ���� Permutation �����еĽ�����Ϣ
                swap(Permutation[pivot_col], Permutation[target]);
            }
            // ����Ԫλ�� H[r][target]���������������ϵ�1��ȥ
            for (int i = 0; i < M; i++) {
                if (i != r && H[i][target] == 1) {
                    for (int j = target; j < N; j++) {
                        H[i][j] ^= H[r][j];
                    }
                }
            }
        }
    }
    // ��ʹ���б任���и�˹��Ԫ���� H ת��Ϊϵͳ����ʽ [P I]
    void Gaussian_Elimination_RowTransform() {
        cout << "Processing Gaussian_Elimination" << endl;
        for (int r = 0; r < M; ++r) {
            int pivot_row = -1;

            // �ڵ�ǰ���ҵ���Ԫ��
            for (int i = r; i < M; ++i) {
                if (H[i][N - M + r] == 1) {
                    pivot_row = i;
                    break;
                }
            }
            if (pivot_row == -1) {
                cerr << "Warning: No pivot found in row " << r << "!" << endl;
                continue;
            }

            // ������ǰ�к���Ԫ��
            if (pivot_row != r) {
                swap(H[r], H[pivot_row]);
            }

            // ��ȥ�����еĸ��У�ʹ���ɵ�λ����
            for (int i = 0; i < M; ++i) {
                if (i != r && H[i][N - M + r] == 1) {
                    for (int j = 0; j < N; ++j) {
                        H[i][j] ^= H[r][j]; // GF(2) �ӷ�
                    }
                }
            }
        }
    }
    // Print H matrix for testing
    void PrintH() {
        // ���У�����
        cout << "H = " << endl;
        cout << "[" << endl;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                cout << H[i][j] << "";
            }
            cout << endl;
        }
        cout << "]" << endl;
    }

};

class LDPC_Encoder{
private:
    // ϵͳ�� H ���� [P I]��H Ϊ checkNum �С�codeLength �У�����
    // ��� Inflength ��Ϊ P �����Ҳ� checkNum ��Ϊ��λ���� I��
    vector<vector<int>> H;
    int checkNum;// Number of check node.
    int Inflength;// length of information bit.
    int codeLength;// Code Length
    vector<int> CodeWord;
    vector<int> syndrome;
public:
    // ���캯��
    LDPC_Encoder(CheckMatrix checkMatrix) {
        H = checkMatrix.H;
        checkNum = checkMatrix.M;
        codeLength = checkMatrix.N;
        Inflength = codeLength - checkNum;
        //printf("LDPC Encoding...\n");
    }
    // ���뺯��������ϵͳ�� H ���� [P I] ����Ϣ Msg ���б��룬
       // ����У��λ parity[j] = (sum_{i=0}^{Inflength-1} P[j][i]*Msg[i]) mod2��
       // Ȼ�� CodeWord = [Msg, parity]��
    vector<int> encode(vector<int> Msg) {
        // �������ִ�С��ȷ�� CodeWord ����Ϊ codeLength
        
        CodeWord.resize(codeLength, 0);

        // ����Ϣλ����������ǰ Inflength λ
        for (int i = 0; i < Inflength; i++) {
            CodeWord[i] = Msg[i];
        }

        // ����У��λ������ÿ��У��ڵ� j (0 <= j < checkNum)
        // ע�⣺H ������� j ��Ӧ�� P ������ H[j][0...Inflength-1]
        // У��λӦ���� H[j][0...Inflength-1] * Msg^T + parity[j] = 0 (mod 2)
        // ���� parity[j] = (sum_{i=0}^{Inflength-1} H[j][i]*Msg[i]) mod2.
        for (int j = 0; j < checkNum; j++) {
            int parity = 0;
            for (int i = 0; i < Inflength; i++) {
                // �˷��� GF(2) ��Ϊ����������Ϊ���
                parity ^= (H[j][i] & Msg[i]);
            }
            // ��У��λ�������ֵĺ� checkNum ��λ����
            CodeWord[Inflength + j] = parity;
        }
        return move(CodeWord);
    }
    // ��ӡ�����Ľ��
    void PrintResult() {
        cout << "Encoded CodeWord:" << endl;
        for (size_t i = 0; i < codeLength; i++) {
            cout << CodeWord[i];
        }
        cout << endl;
    }

    void CheckSyndrome() {
        // ���� H * c^T����������� syndrome ������
        int m = checkNum;
        int n = codeLength;
        syndrome.resize(m);

        for (int i = 0; i < m; i++) {
            int sum = 0;
            for (int j = 0; j < n; j++) {
                // ģ2�˷��ͼӷ�
                sum = mod2Add(sum, H[i][j] * CodeWord[j]);
            }
            syndrome[i] = sum % 2;
        }
        PrintSyndrome();
    }
    void PrintSyndrome(){
        cout << "Syndrome:" << endl;
        for (int bit : syndrome) {
            cout << bit;
        }
        cout << "\n" << endl;
    }
};

//==========================================DECODER=========================================//
class LDPC_Decoder {
private:
    int Z;                         // puncturing ����
    vector<vector<int>> H;         // У�������ΪУ��ڵ㣬ÿ�д洢 0/1��
    int numParBits;                // У��ڵ���� = H.size()
    int numTotBits;                // �ܱ����� = H[0].size() + 2*Z������ǰ�ò�0��
    int iterations;                // ��������
    int numInfBits;                // ��Ϣλ����
    vector<double> CodeWord_Rx;    // �ŵ����
    vector<double> llr;            // �������LLR���
    vector<int> Msg_Rx;            // �ָ�����Ϣ����
    vector<int> Permutation;         // ��˹��Ԫʱ�Ľ�����Ϣ
    
    // ���������ţ�x>=0 ����1�����򷵻�-1
    inline int sign(double x) {
        return (x >= 0) ? 1 : -1;
    }

    // Ϊ��ֹ log(0) ����һ����Сֵ��realmin��
    const double minVal = numeric_limits<double>::min();
public:
    // ���캯��
    LDPC_Decoder(CheckMatrix checkMatrix, int Maxiter, int z)
    {
        H = checkMatrix.H;
        numParBits = checkMatrix.M;
        numTotBits = checkMatrix.N;
        iterations = Maxiter;
        numInfBits = numTotBits - numParBits;
        Z = z;
        Permutation = checkMatrix.Permutation;
    }

    // Function to convert to voting score to LLR
    void to_LLR(vector<double> codeWrd_Rx) {
        CodeWord_Rx = codeWrd_Rx;
        int threshold_low = -5;
        int threshold_up = 5;
        int n = numTotBits;
        llr.resize(n);
        double L;
        for (int i = 0; i < n; ++i) {
            L = log((1 - CodeWord_Rx[i]) / CodeWord_Rx[i]);
            if (L <= threshold_low) {
                llr[i] = threshold_low;
            }
            else if (L > threshold_low and L < threshold_up) {
                llr[i] = L;
            }
            else {
                llr[i] = threshold_up;
            }
        }
    }

    // Print the LLr for testing.
    void PrintLLr() {
        //Output the result of the LRR calculation.
        cout << "LLR:" << endl;
        for (double bit : llr) {
            cout << bit<< " ";
        }
        cout << endl;
    }
    // SPA���뺯����llr Ϊ����� LLR ������δ����֮ǰ��
    vector<int> Decode() {
        // 1. Preprocessing: ���㣨����ǰ2*Z�� punctured bit��
        int preZeros = 2 * Z;
        vector<double> Qv(preZeros, 0.0); // ǰ preZeros ��Ԫ�ز�0
        Qv.insert(Qv.end(), llr.begin(), llr.end());

        // 2. ��ʼ����Ϣ���� Rcv��ά��Ϊ [numParBits x numTotBits]��ע�⣺numTotBits �Ѿ����ǲ��㣩
        vector<vector<double>> Rcv(numParBits, vector<double>(numTotBits, 0.0));

        for (int iter = 0; iter < iterations; ++iter) {
            // ����ÿ��У��ڵ㣨�У�
            for (int checkIdx = 0; checkIdx < numParBits; ++checkIdx) {
                // Ѱ�Ҹ�У��ڵ��Ӧ H ������Ϊ 1 �ı����ڵ�����
                vector<int> nbVarNodes;
                for (int varIdx = 0; varIdx < H[checkIdx].size(); ++varIdx) {
                    if (H[checkIdx][varIdx] == 1) {
                        nbVarNodes.push_back(varIdx);
                    }
                }

                // ���� tmpLlr������ÿ���ڽӵı����ڵ㣬
                // tmpLlr = Qv(varIdx) - Rcv(checkIdx, varIdx)
                vector<double> tmpLlr;
                for (int varIdx : nbVarNodes) {
                    tmpLlr.push_back(Qv[varIdx] - Rcv[checkIdx][varIdx]);
                }

                // ���� S �ķ�ֵ���֣�Smag = sum( -log(minVal + tanh(|tmpLlr|/2) ) )
                double Smag = 0.0;
                for (double val : tmpLlr) {
                    // ������� minVal ���� tanh(0) ���� log(0)
                    Smag += -log(minVal + tanh(fabs(val) / 2.0));
                }

                // ���� S �ķ��Ų��֣�ͳ�� tmpLlr �и����ĸ�������ż�� Ssign = +1������ -1
                int negCount = 0;
                for (double val : tmpLlr) {
                    if (val < 0) negCount++;
                }
                int Ssign = (negCount % 2 == 0) ? 1 : -1;

                // ����ÿ���ڽӵı����ڵ㣬������Ϣ
                for (int varIdx : nbVarNodes) {
                    double Qtmp = Qv[varIdx] - Rcv[checkIdx][varIdx];
                    double QtmpMag = -log(minVal + tanh(fabs(Qtmp) / 2.0));
                    int QtmpSign = sign(Qtmp); // ���ﲻ�� minVal���� minVal ��С

                    // ���� Rcv ��Ϣ���ο���ʽ Rcv = phi^-1( S - phi(Qtmp) )
                    // ������� -log(minVal + tanh(|S_mag - QtmpMag|/2)) ��Ϊ��Ϣ��ֵ
                    double newMsg = Ssign * QtmpSign * (-log(minVal + tanh(fabs(Smag - QtmpMag) / 2.0)));
                    Rcv[checkIdx][varIdx] = newMsg;

                    // ���� Qv�� Qv(varIdx) = Qtmp + Rcv(checkIdx, varIdx)
                    Qv[varIdx] = Qtmp + newMsg;
                }
            }
        }

        // 4. Ӳ�о���������Ϣ���ز��֣�������Ϣ���ش洢�� Qv ��ǰ numInfBits ��λ�ã�
        Msg_Rx.resize(numInfBits);
        for (int i = 0; i < numInfBits; ++i) {
            Msg_Rx[i] = (Qv[i] < 0) ? 1 : 0;
        }
        return move(Msg_Rx);
    }
    void Restore_Order() {
        vector<int> Msg_Rx_temp(Msg_Rx);
        // ����Permutation�ָ�ԭ����˳��
        for (size_t i = 0; i < numTotBits; ++i) {
            // Permutation[i] ��ʾϵͳ����� i ��������ԭʼ��Ϣ�е�λ��
            Msg_Rx_temp[Permutation[i]] = Msg_Rx[i];
        }
        Msg_Rx = Msg_Rx_temp;
        //delete & Msg_Rx_temp;
    }
    void PrintDecodeRes() {
        //Output the result of the decoder.
        
        cout << "The recovered messages:" << endl;
        for (int i = 0; i < numInfBits; ++i) {
            cout << Msg_Rx[i] << "";
        }
        cout << endl;
    }
};


//==========================================CHANNEL SIMU=========================================//

class Channel {
public:
    //Python����·�������
    PyObject* pName;
    PyObject* pModule;
    PyObject* pFunc;

    //Python��������
    PyObject* pList;
    PyObject* pArgs;
    PyObject* pValue;

    //����ָ��
    PyObject* pRow;
    PyObject* pVal;
    PyObject* pInner;
    PyObject* pNum;

    void InitializeChannel() {

        Py_Initialize();//ʹ��python֮ǰ��Ҫ����Py_Initialize();����������г�ʼ��

        if (!Py_IsInitialized())
        {
            throw runtime_error("��ʼ��ʧ�ܣ�");
        }

        // ���� Python ģ������·��
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append(r'D:\\DeSP-main')");  // ����ļ���·��

        /*����Ҫ���õ�python�ļ����ƣ���ǰ�����ļ����ƣ�DNAChannel.py*/
        pName = PyUnicode_DecodeFSDefault("LDPC-BCH_Two-Layer_cost_optimization");
        pModule = PyImport_Import(pName);
        Py_XDECREF(pName);

        if (pModule == NULL) {
            PyErr_Print();
            throw runtime_error("�޷�����ģ�� LDPC-BCH_Two-Layer_cost_optimization");
        }
        // ��ȡģ���еĺ��� PyDNAChannel
        pFunc = PyObject_GetAttrString(pModule, "DNAChannel");
        if (!pFunc || !PyCallable_Check(pFunc)) {
            PyErr_Print();
            throw runtime_error("�Ҳ������� DNAChannel");
        }
        return;
    }

    vector<vector<double>> DNAChal(vector<vector<int>> CodeWrd_Tx, double Noise_Lvl, int sequencingDepth, int innerRedundancy) {
        vector<vector<double>> msgRx;
        // �����һ���������� CodeWrd_Tx ת��Ϊ Python �� list-of-lists
        pList = PyList_New(CodeWrd_Tx.size());
        for (size_t i = 0; i < CodeWrd_Tx.size(); i++) {
            pRow = PyList_New(CodeWrd_Tx[i].size());
            for (size_t j = 0; j < CodeWrd_Tx[i].size(); j++) {
                pVal = PyLong_FromLong(CodeWrd_Tx[i][j]);
                PyList_SetItem(pRow, j, pVal); // ���ú� pVal ������Ȩ�� pRow ����
            }
            PyList_SetItem(pList, i, pRow);
        }

        // �������Ԫ�飬��������������list-of-lists �� Noise_Lvl��ת��Ϊ Python float��
        pArgs = PyTuple_New(4);
        PyTuple_SetItem(pArgs, 0, pList);  // pList ������ת�Ƶ�Ԫ����
        PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble(Noise_Lvl));
        PyTuple_SetItem(pArgs, 2, PyLong_FromLong(sequencingDepth));
        PyTuple_SetItem(pArgs, 3, PyLong_FromLong(innerRedundancy));
        //Py_XDECREF(pList);

        // ���� Python ���� PyDNAChannel(list-of-lists, Noise_Lvl)
        pValue = PyObject_CallObject(pFunc, pArgs);
        Py_XDECREF(pArgs);

        if (pValue != NULL) {
            // pValue Ӧ����һ�� list-of-lists���������ת��
            if (PyList_Check(pValue)) {
                Py_ssize_t outerSize = PyList_Size(pValue);
                msgRx.resize(outerSize);
                for (Py_ssize_t i = 0; i < outerSize; i++) {
                    pInner = PyList_GetItem(pValue, i); // ��������
                    if (PyList_Check(pInner)) {
                        Py_ssize_t innerSize = PyList_Size(pInner);
                        msgRx[i].resize(innerSize);
                        for (Py_ssize_t j = 0; j < innerSize; j++) {
                            pNum = PyList_GetItem(pInner, j);
                            msgRx[i][j] = PyFloat_AsDouble(pNum);
                        }
                    }
                }
            }
            else {
                PyErr_Print();
                throw runtime_error("���ؽ�������б��ʽ��");
            }
            Py_XDECREF(pValue);
        }
        else {
            PyErr_Print();
            throw runtime_error("���� DNAChannel ʧ�ܣ�");
        }
        return msgRx;
    }

    void CloseChannel() {
        Py_XDECREF(pFunc);
        Py_XDECREF(pModule);
        Py_XDECREF(pList); // ȷ���ͷ����ж�̬����� Python ����
        Py_Finalize();
    }

};


//######################################################################################//
//                                        main
// ���������� 4 ��������-NoiseLevel [Pe] -SequencingDepth [�������] -InnerRedundancy [��������������]
// 
//######################################################################################//

int main(int argc, char* argv[]) {
    // Reading parameter form command line.
    if (argc < 7) {  // ȷ��������7����������һ���ǳ�������
        std::cerr << "Usage: " << argv[0] << "Should have 3 parameters: Noise Level, Sequencing Depth and InnerRedundancy" << std::endl;
        return 1;  // ���ش�����
    }
    const double NoiseLvl = stod(argv[2]); // Nosie level of the DNA channel
    const int sequencingDepth = stoi(argv[4]);
    const int innerRedundancy = stoi(argv[6]);

    //======================Read the Parity Check Matrix======================//
    CheckMatrix CheckMtx;//����check matrix����
    CheckMtx.read_H_matrix(HmatrixPath);
    const int N = CheckMtx.N;
    const int M = CheckMtx.M;
    //CheckMtx.PrintH();//��ӡ�ļ�������H����
    CheckMtx.Gaussian_Elimination();//��ԭʼ������и�˹��Ԫ��
    //CheckMtx.PrintH();//��ӡ��˹��Ԫ���H����

    // Print the coding configuration to the terminal.
    double R_o = static_cast<double>(CheckMtx.N - CheckMtx.M) / CheckMtx.N;
    double R_i = static_cast<double>(334) / (334 + innerRedundancy);
    double sequencingCost = static_cast<double>(sequencingDepth * 0.5) / (R_o * R_i);
    printf("Coding Config:\n");
    printf("Outer Code: (%d, %d), %.2f\n", CheckMtx.N, CheckMtx.N - CheckMtx.M, R_o);
    printf("Inner Code: (%d, %d), %.2f\n", 334 + innerRedundancy, innerRedundancy, R_i);
    printf("Sequencing Cost: %.2f bit/base", sequencingCost);


    // ���������ļ������Ǿ��ļ���д�������
    /*
    ofstream file(filePathData);
    if (!file.is_open()) {
        cerr << "Failed to create Experiment.csv!" << endl;
        return 1;
    }
    file << "PE,Sequencing Depth,R_o,R_i,First Error Frame,Minimal Cost to Achieve 1e-6 FER\n";
    file.close();  // �ر��ļ����ͷ���Դ��������׷��ģʽ�򿪣�
    */

    //��ʼ�������Ϣ������
    const int K = N - M;
    MessageGenerator MsgGen;
    vector<vector<int>> msgsTx(sequenceNum,vector<int>(K)); //Contianer for the generated message

    //��ʼ��LDPC������
    LDPC_Encoder Encoder(CheckMtx);
    vector<vector<int>> codeWrds_Tx(sequenceNum, vector<int>(N));

    // ��������ʼ��Python Channel
    Channel DNAChannel;
    DNAChannel.InitializeChannel();
    int firsErrorFrame = -1; //��¼��һ������֡���ֵĵط�
    int RepetitionRequired = ceil(1e6 / (CheckMtx.N - CheckMtx.M)); //Minmal Experimental Repetition Required to achieve 1e-6 FER.
    vector<vector<double>> codeWrds_Rx(sequenceNum, vector<double>(N)); //Container for the channel output.
    // ��ʼ��LDPC������
    LDPC_Decoder Decoder(CheckMtx, 30, 0);
    vector<vector<int>> msgsRx(sequenceNum, vector<int>(K)); //Container for the decoded message.

    Duration Simu;
    Simu.simuStart();//��¼ʵ�鿪ʼ��ʱ��
    //��������ʵ��������������������С��Ҫ��Ĵ�����û�г��ִ���֡��
    for (int exp = 0; exp < RepetitionRequired  && firsErrorFrame == -1; ++exp) {
        Simu.expStart();//��¼����ʵ�鿪ʼ��ʱ��
        cout << "\n\n=========================== " << "Experiment " << exp + 1 << " / " << RepetitionRequired <<" ===========================" << endl;
        //======================Generate Random Messages======================//
        
        for (int i = 0; i < sequenceNum; ++i) {
            
            msgsTx[i] = MsgGen.genMsg(K);
        }
        //MsgGen.PrintMsg();

        //======================LDPC Encoding======================//
        for (int i = 0; i < sequenceNum; ++i) {
            codeWrds_Tx[i] = Encoder.encode(msgsTx[i]);
            cout << "LDPC Encoding"<<"("<<CheckMtx.N <<", "<< K << ")..." << i + 1 << " / " << sequenceNum << "\r";
        }
        cout << endl;
        //Encoder.PrintResult();
        //Encoder.CheckSyndrome();


        //======================Call DNA Channle in Python======================//
        codeWrds_Rx = DNAChannel.DNAChal(codeWrds_Tx, NoiseLvl, sequencingDepth, innerRedundancy);

        //======================LDPC Decoding and Calculation of FER========================//
        for (int i = 0; i < sequenceNum; ++i) {
            cout << "LDPC decoding..." << i + 1 << "/" << sequenceNum << "\r";
            Decoder.to_LLR(codeWrds_Rx[i]);
            msgsRx[i] = Decoder.Decode();
            if (HammingDistance(msgsTx[i], msgsRx[i]) != 0) {
                firsErrorFrame = exp * K + i + 1;
                cout << "\nFER != 0, " << i + 1 << "/" << sequenceNum << endl;
                cout << "End of this experiemnt." << endl;
                //system("pause");
                //return 0;
                break;
            }
        }
        //���������ɵ�ʱ��
        Simu.completionTime(RepetitionRequired);
    }
    // �رգ�Python
    DNAChannel.CloseChannel();
    // �����ļ�����ʹ��throw����浱ǰ�ĵ�����Ϣ��
    // ��׷��ģʽ���ļ���д��������
    ofstream outfile(filePathData, ios::app);
    if (!outfile.is_open()) {
        cerr << "Failed to open Experiment.csv for appending!" << endl;
        system("pause");
        return 1;
    }
    outfile << fixed << setprecision(2) << NoiseLvl << ","
        << sequencingDepth << ","
        << R_o << ","
        << R_i << ","
        << firsErrorFrame << ","
        << sequencingCost << "\n";
    outfile.close();
    cout << "Simulation data saved to Experiment.csv" << endl;
    system("pause");
    return 0;
}