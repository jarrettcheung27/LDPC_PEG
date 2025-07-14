#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <Eigen/Sparse>
#include <python.h>
#include <iomanip>  // 用于控制输出格式
#include <cstdlib>  // 使用 atoi() 转换字符串到整数
#include <chrono>
#include <ctime>
#include <filesystem> // C++17 文件系统库

using namespace std;
using namespace Eigen;

const string filePathData = "D:\\DeSP-main\\Data\\Cost_Optimization_result\\Fix_indexing_cost\\Data\\Simu";
//const string HmatrixPath = "D:\\DeSP-main\\Classic_PEG\\x64\\Debug\\Hmatrix_test.txt"
const int sequenceNum = 320; //DNA序列长度/bit
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

class Duration {
public:
    chrono::system_clock::time_point simu_start_time; //start time of the simulation.
    chrono::steady_clock::time_point exp_start;//start time of the experiment.
    chrono::steady_clock::time_point exp_end; //end time if the experiment.

    void simuStart() {
        simu_start_time = chrono::system_clock::now(); //记录仿真开始的时间
    }
    void expStart() {
        exp_start = chrono::high_resolution_clock::now();//记录本次实验开始的时间
    }

    void completionTime(int RepetitionRequired) {
        /* Args:
        *   RepetitionRequired: 本次仿真需要重复实验的次数
        */
        exp_end = chrono::high_resolution_clock::now();//记录本次实验结束的时间
        chrono::duration<double> elapsed = exp_end - exp_start; //本次实验的持续时间
        cout << "Time spent in this experiment: " << elapsed.count() << "s" << endl;
        double total_time = elapsed.count() * RepetitionRequired; //计算总耗时
        auto estimated_end_time = simu_start_time + chrono::seconds(static_cast<long>(total_time)); //计算预估的完成时间
        time_t end_time_t = chrono::system_clock::to_time_t(estimated_end_time); //转换成年月日时分秒的形式
        char end_time_str[26];
        ctime_s(end_time_str, sizeof(end_time_str), &end_time_t);
        cout << "Estimated time for simulation completion: " << end_time_str << "Duariton: " << total_time / 3600 << "h";
    }

    void expDuration() {
        exp_end = chrono::high_resolution_clock::now();//记录本次实验结束的时间
        chrono::duration<double> elapsed = exp_end - exp_start; //本次实验的持续时间
        cout << "Time spent in this experiment: " << elapsed.count() << "s" << endl;
    }

};

vector<vector<int>> ReadMsgs(const string& input_filepath,const int& K, const int& sequenceNum) { 
    // 从文件中读取消息序列，返回一个二维向量 msgsTx，其中每行代表一个消息序列。
    // Args:        
    //   input_filepath: 输入文件路径，包含消息序列。
    //   K: 每个消息序列的长度。
    //   sequenceNum: 需要读取的消息序列数量。
    vector<vector<int>> msgsTx(sequenceNum, vector<int>(K)); // 初始化 msgsTx 容器
    ifstream msgFile(input_filepath);
    if (!msgFile.is_open()) {
        cerr << "Failed to open input file: " << input_filepath << endl;
        return vector<vector<int>>();
    }
    string line;
    int msgIdx = 0;
    while (getline(msgFile, line) && msgIdx < sequenceNum) {
        vector<int> bits;
        for (char c : line) {
            if (c == '0' || c == '1') {
                bits.push_back(c - '0');
            }
        }
        if (bits.size() != K) {
            cerr << "Error: Line " << msgIdx + 1 << " does not contain exactly " << K << " bits." << endl;
            return vector<vector<int>>();
        }
        msgsTx[msgIdx] = bits;
        msgIdx++;
    }
    if (msgIdx < sequenceNum) {
        cerr << "Error: Not enough message sequences in input file." << endl;
        return vector<vector<int>>();
    }
    msgFile.close();
    return msgsTx;
}

void WriteEncodedMessages(const vector<vector<int>>& codeWrds_Tx, const string& output_filepath) {
        ofstream outFile(output_filepath);
        if (!outFile.is_open()) {
            cerr << "Error: Could not open output file: " << output_filepath << endl;
            return;
        }
        for (const auto& codeword : codeWrds_Tx) {
            for (int bit : codeword) {
                outFile << bit;
            }
            outFile << '\n';
        }
        outFile.close();
        cout << "Encoded messages written to " << output_filepath << endl;
    }

vector<vector<double>> ReadVotingscore(const string& input_filepath, const int& N, const int& sequenceNum) { 
    // 从文件中读取投票得分，返回一个二维向量 msgsRx，其中每行代表一个信道输出，用于后续计算LLR。
    // Args:        
    //   input_filepath: 输入文件路径，包含投票得分序列。
    //   N: 每个投票得分序列的长度。
    //   sequenceNum: 需要读取的投票得分序列数量。

    vector<vector<double>> msgsRx(sequenceNum, vector<double>(N));
    ifstream scoreFile(input_filepath);
    if (!scoreFile.is_open()) {
        cerr << "Failed to open input file: " << input_filepath << endl;
        return vector<vector<double>>();
    }
    string line;
    int msgIdx = 0;
    while (getline(scoreFile, line) && msgIdx < sequenceNum) {
        vector<double> scores;
        istringstream iss(line);
        double val;
        while (iss >> val) {
            scores.push_back(val);
        }
        if (scores.size() != N) {
            cerr << "Error: Line " << msgIdx + 1 << " does not contain exactly " << N << " scores." << endl;
            return vector<vector<double>>();
        }
        msgsRx[msgIdx] = scores;
        msgIdx++;
    }
    if (msgIdx < sequenceNum) {
        cerr << "Error: Not enough voting score sequences in input file." << endl;
        return vector<vector<double>>();
    }
    scoreFile.close();
    return msgsRx;
}  
    //==========================================ENCODER=========================================//
// 定义稀疏校验矩阵H的类
class CheckMatrix {
public:
    int M;       // The number of check nodes M
    int N;       // Code length N
    vector<vector<int>> H;// H matrix read from the file (Generate by PEG Algorithm.)
    vector<int> Permutation;        // 记录列交换信息：变换后H中第i列在原H矩阵中的索引
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
    // 高斯消元函数（在GF(2)上），将H矩阵消元为系统化形式 [P I]
    // 同时记录每次列交换信息到 Permutation 中，便于译码时恢复原始信息顺序。
    void Gaussian_Elimination() {
        cout << "Processing Gaussian_Elimination" << endl;
        // 初始化 Permutation 为身份排列，即 Permutation[i] = i
        Permutation.resize(N);
        for (int i = 0; i < N; ++i) {
            Permutation[i] = i;
        }

        // 对于每一行 r，目标是将右侧 M 列构造为单位矩阵
        // 即对行 r，目标列 target = N - M + r
        for (int r = 0; r < M; ++r) {
            int target = N - M + r;
            int pivot_col = -1;
            // 在第 r 行中寻找一个1作为主元（可从整个行中寻找）
            for (int j = 0; j < N; j++) {
                if (H[r][j] == 1) {
                    pivot_col = j;
                    break;
                }
            }
            if (pivot_col == -1) {
                // 当前行无1，矩阵秩可能不足，无法转为完全系统化形式
                cerr << "Warning: Row " << r << " has no pivot. The matrix may be rank deficient." << endl;
                continue;
            }
            // 如果找到的主元不在目标位置，则交换列，将主元移到 target 列
            if (pivot_col != target) {
                for (int i = 0; i < M; i++) {
                    swap(H[i][pivot_col], H[i][target]);
                }
                // 同时更新 Permutation 向量中的交换信息
                swap(Permutation[pivot_col], Permutation[target]);
            }
            // 现主元位于 H[r][target]，将该列其他行上的1消去
            for (int i = 0; i < M; i++) {
                if (i != r && H[i][target] == 1) {
                    for (int j = target; j < N; j++) {
                        H[i][j] ^= H[r][j];
                    }
                }
            }
        }
    }
    // 仅使用行变换进行高斯消元，将 H 转换为系统化形式 [P I]
    void Gaussian_Elimination_RowTransform() {
        cout << "Processing Gaussian_Elimination" << endl;
        for (int r = 0; r < M; ++r) {
            int pivot_row = -1;

            // 在当前列找到主元行
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

            // 交换当前行和主元行
            if (pivot_row != r) {
                swap(H[r], H[pivot_row]);
            }

            // 消去其他行的该列，使其变成单位矩阵
            for (int i = 0; i < M; ++i) {
                if (i != r && H[i][N - M + r] == 1) {
                    for (int j = 0; j < N; ++j) {
                        H[i][j] ^= H[r][j]; // GF(2) 加法
                    }
                }
            }
        }
    }
    // Print H matrix for testing
    void PrintH() {
        // 输出校验矩阵
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

class LDPC_Encoder {
private:
    // 系统化 H 矩阵 [P I]：H 为 checkNum 行、codeLength 列，其中
    // 左侧 Inflength 列为 P 矩阵，右侧 checkNum 列为单位矩阵 I。
    vector<vector<int>> H;
    int checkNum;// Number of check node.
    int Inflength;// length of information bit.
    int codeLength;// Code Length
    vector<int> CodeWord;
    vector<int> syndrome;
public:
    // 构造函数
    LDPC_Encoder(CheckMatrix checkMatrix) {
        H = checkMatrix.H;
        checkNum = checkMatrix.M;
        codeLength = checkMatrix.N;
        Inflength = codeLength - checkNum;
        //printf("LDPC Encoding...\n");
    }
    // 编码函数：利用系统化 H 矩阵 [P I] 对信息 Msg 进行编码，
       // 计算校验位 parity[j] = (sum_{i=0}^{Inflength-1} P[j][i]*Msg[i]) mod2，
       // 然后将 CodeWord = [Msg, parity]。
    vector<int> encode(vector<int> Msg) {
        // 调整码字大小，确保 CodeWord 长度为 codeLength

        CodeWord.resize(codeLength, 0);

        // 将信息位拷贝到码字前 Inflength 位
        for (int i = 0; i < Inflength; i++) {
            CodeWord[i] = Msg[i];
        }

        // 计算校验位。对于每个校验节点 j (0 <= j < checkNum)
        // 注意：H 矩阵的行 j 对应的 P 部分在 H[j][0...Inflength-1]
        // 校验位应满足 H[j][0...Inflength-1] * Msg^T + parity[j] = 0 (mod 2)
        // 故有 parity[j] = (sum_{i=0}^{Inflength-1} H[j][i]*Msg[i]) mod2.
        for (int j = 0; j < checkNum; j++) {
            int parity = 0;
            for (int i = 0; i < Inflength; i++) {
                // 乘法在 GF(2) 中为与操作，求和为异或
                parity ^= (H[j][i] & Msg[i]);
            }
            // 将校验位放在码字的后 checkNum 个位置中
            CodeWord[Inflength + j] = parity;
        }
        return move(CodeWord);
    }
    // 打印编码后的结果
    void PrintResult() {
        cout << "Encoded CodeWord:" << endl;
        for (size_t i = 0; i < codeLength; i++) {
            cout << CodeWord[i];
        }
        cout << endl;
    }

    void CheckSyndrome() {
        // 计算 H * c^T，结果保存在 syndrome 向量中
        int m = checkNum;
        int n = codeLength;
        syndrome.resize(m);

        for (int i = 0; i < m; i++) {
            int sum = 0;
            for (int j = 0; j < n; j++) {
                // 模2乘法和加法
                sum = mod2Add(sum, H[i][j] * CodeWord[j]);
            }
            syndrome[i] = sum % 2;
        }
        PrintSyndrome();
    }
    void PrintSyndrome() {
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
    int Z;                         // puncturing 参数
    vector<vector<int>> H;         // 校验矩阵（行为校验节点，每行存储 0/1）
    int numParBits;                // 校验节点个数 = H.size()
    int numTotBits;                // 总比特数 = H[0].size() + 2*Z（考虑前置补0）
    int iterations;                // 迭代次数
    int numInfBits;                // 信息位个数
    vector<double> CodeWord_Rx;    // 信道输出
    vector<double> llr;            // 计算出的LLR结果
    vector<int> Msg_Rx;            // 恢复的信息比特
    //vector<int> Permutation;         // 高斯消元时的交换信息

    // 返回正负号：x>=0 返回1，否则返回-1
    inline int sign(double x) {
        return (x >= 0) ? 1 : -1;
    }

    // 为防止 log(0) 采用一个极小值（realmin）
    const double minVal = numeric_limits<double>::min();
public:
    // 构造函数
    LDPC_Decoder(CheckMatrix checkMatrix, int Maxiter, int z)
    {
        H = checkMatrix.H;
        numParBits = checkMatrix.M;
        numTotBits = checkMatrix.N;
        iterations = Maxiter;
        numInfBits = numTotBits - numParBits;
        Z = z;
        //Permutation = checkMatrix.Permutation;
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
            cout << bit << " ";
        }
        cout << endl;
    }
    // SPA译码函数，llr 为输入的 LLR 向量（未补零之前）
    vector<int> Decode() {
        // 1. Preprocessing: 补零（对于前2*Z个 punctured bit）
        int preZeros = 2 * Z;
        vector<double> Qv(preZeros, 0.0); // 前 preZeros 个元素补0
        Qv.insert(Qv.end(), llr.begin(), llr.end());

        // 2. 初始化消息矩阵 Rcv：维度为 [numParBits x numTotBits]（注意：numTotBits 已经考虑补零）
        vector<vector<double>> Rcv(numParBits, vector<double>(numTotBits, 0.0));

        for (int iter = 0; iter < iterations; ++iter) {
            // 遍历每个校验节点（行）
            for (int checkIdx = 0; checkIdx < numParBits; ++checkIdx) {
                // 寻找该校验节点对应 H 的行中为 1 的变量节点索引
                vector<int> nbVarNodes;
                for (int varIdx = 0; varIdx < H[checkIdx].size(); ++varIdx) {
                    if (H[checkIdx][varIdx] == 1) {
                        nbVarNodes.push_back(varIdx);
                    }
                }

                // 计算 tmpLlr：对于每个邻接的变量节点，
                // tmpLlr = Qv(varIdx) - Rcv(checkIdx, varIdx)
                vector<double> tmpLlr;
                for (int varIdx : nbVarNodes) {
                    tmpLlr.push_back(Qv[varIdx] - Rcv[checkIdx][varIdx]);
                }

                // 计算 S 的幅值部分：Smag = sum( -log(minVal + tanh(|tmpLlr|/2) ) )
                double Smag = 0.0;
                for (double val : tmpLlr) {
                    // 这里加上 minVal 避免 tanh(0) 导致 log(0)
                    Smag += -log(minVal + tanh(fabs(val) / 2.0));
                }

                // 计算 S 的符号部分：统计 tmpLlr 中负数的个数，若偶数 Ssign = +1，否则 -1
                int negCount = 0;
                for (double val : tmpLlr) {
                    if (val < 0) negCount++;
                }
                int Ssign = (negCount % 2 == 0) ? 1 : -1;

                // 对于每个邻接的变量节点，更新消息
                for (int varIdx : nbVarNodes) {
                    double Qtmp = Qv[varIdx] - Rcv[checkIdx][varIdx];
                    double QtmpMag = -log(minVal + tanh(fabs(Qtmp) / 2.0));
                    int QtmpSign = sign(Qtmp); // 这里不加 minVal，因 minVal 很小

                    // 更新 Rcv 消息：参考公式 Rcv = phi^-1( S - phi(Qtmp) )
                    // 这里采用 -log(minVal + tanh(|S_mag - QtmpMag|/2)) 作为消息幅值
                    double newMsg = Ssign * QtmpSign * (-log(minVal + tanh(fabs(Smag - QtmpMag) / 2.0)));
                    Rcv[checkIdx][varIdx] = newMsg;

                    // 更新 Qv： Qv(varIdx) = Qtmp + Rcv(checkIdx, varIdx)
                    Qv[varIdx] = Qtmp + newMsg;
                }
            }
        }

        // 4. 硬判决：对于信息比特部分（假设信息比特存储在 Qv 的前 numInfBits 个位置）
        Msg_Rx.resize(numInfBits);
        for (int i = 0; i < numInfBits; ++i) {
            Msg_Rx[i] = (Qv[i] < 0) ? 1 : 0;
        }
        return move(Msg_Rx);
    }
    /*
    void Restore_Order() {
        vector<int> Msg_Rx_temp(Msg_Rx);
        // 根据Permutation恢复原来的顺序
        for (size_t i = 0; i < numTotBits; ++i) {
            // Permutation[i] 表示系统化后第 i 个比特在原始消息中的位置
            Msg_Rx_temp[Permutation[i]] = Msg_Rx[i];
        }
        Msg_Rx = Msg_Rx_temp;
        //delete & Msg_Rx_temp;
    }
    */
    void PrintDecodeRes() {
        //Output the result of the decoder.

        cout << "The recovered messages:" << endl;
        for (int i = 0; i < numInfBits; ++i) {
            cout << Msg_Rx[i] << "";
        }
        cout << endl;
    }
};



//######################################################################################//
//                                        main
// 命令行输入 4 个参数：-NoiseLevel [Pe] -SequencingDepth [测序深度] -InnerRedundancy [内码总冗余数量]
// 
//######################################################################################//

int main(int argc, char* argv[]) {
    // Reading parameter form command line.
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <mode> <input_filepath> <output_filepath> <HmatrixPath>" << std::endl;
        return 1;
    }
    string mode = argv[1];
    string input_filepath = argv[2];
    string output_filepath = argv[3];
    string HmatrixPath = argv[4];

    //======================Read the Parity Check Matrix======================//
    CheckMatrix CheckMtx;//定义check matrix的类
    CheckMtx.read_H_matrix(HmatrixPath);
    const int N = CheckMtx.N; // N为码字长度
    const int M = CheckMtx.M; // M为校验位长度
    const int K = N - M; // K为信息位长度
    //CheckMtx.PrintH();//打印文件读出的H矩阵。
    CheckMtx.Gaussian_Elimination();//对原始矩阵进行高斯消元。
    //CheckMtx.PrintH();//打印高斯消元后的H矩阵。

    if (mode == "encode")
    {
        //======================从 input_filepath 读取待编码的信息序列======================//
        vector<vector<int>> msgsTx(sequenceNum, vector<int>(K)); // Container for the generated message
        msgsTx = ReadMsgs(input_filepath, K, sequenceNum); // 从文件中读取消息序列
        if (msgsTx.empty()) {
            cerr << "Error: No messages read from input file." << endl;
            return 1;
        }

        // 初始化LDPC编码器
        LDPC_Encoder Encoder(CheckMtx);
        vector<vector<int>> codeWrds_Tx(sequenceNum, vector<int>(N));

        //======================LDPC Encoding======================//
        for (int i = 0; i < sequenceNum; ++i) {
            codeWrds_Tx[i] = Encoder.encode(msgsTx[i]);
            cout << "LDPC Encoding" << "(" << CheckMtx.N << ", " << K << ")..." << i + 1 << " / " << sequenceNum << "\r";
        }
        cout << endl;

        //======================输出编码后的结果到 output_filepath======================//
        WriteEncodedMessages(codeWrds_Tx, output_filepath);
    }
    else {
        //======================LDPC Decoding ========================//
        //======================从 input_filepath 读取待编码的投票得分序列======================//
        vector<vector<double>> codeWrds_Rx(sequenceNum, vector<double>(N)); // Container for the received codewords
        codeWrds_Rx= ReadVotingscore(input_filepath, N, sequenceNum); // 从文件中读取消息序列
        if (codeWrds_Rx.empty()) {
            cerr << "Error: No messages read from input file." << endl;
            return 1;
        }
		 vector<vector<int>> msgsRx(sequenceNum, vector<int>(K)); // 用于存储解码后的消息序列
        // 初始化LDPC解码器
        LDPC_Decoder Decoder(CheckMtx, iter, 0); // 这里的 Z 设置为 0，因为不需要前置补零
        for (int i = 0; i < sequenceNum; ++i) {
            cout << "LDPC decoding..." << i + 1 << "/" << sequenceNum << "\r";
            Decoder.to_LLR(codeWrds_Rx[i]);
            msgsRx[i] = Decoder.Decode();
        }
        int errorCount = 0; // 用于统计错误帧的数量
        // 遍历每一列
        for (int col = 0; col < msgsRx[0].size(); ++col) {
            bool hasError = false; // 标记当前列是否存在错误比特

            // 遍历每一行，检查当前列是否有错误比特
            for (int row = 0; row < sequenceNum; ++row) {
                if (msgsRx[row][col] != msgsRx[row][col]) {
                    hasError = true; // 如果发现错误比特，标记为 true
                    break; // 退出当前列的检查
                }
            }
            if (hasError) {
                errorCount++; // 如果当前列有错误比特，错误帧计数器加 1
            }
        }
        cout    << "Total number of error frames: " << errorCount << endl;
        WriteEncodedMessages(msgsRx, output_filepath);
    }   
    return 0;
}