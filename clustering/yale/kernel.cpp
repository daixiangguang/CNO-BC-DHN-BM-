#include"mex.h"

#include"kernel.h"

// nlhs: ��������ĸ���(lefthand side,��������������)

// plhs�������mxArray�����ͷָ��

// nrhs: �����������(righthand side,��������������)

// prhs�������mxArray�����ͷָ��

// ��������������������ôprhs[0]ָ���һ������

//prhs[1]ָ��ڶ�������

// Matlab��arrayʹ��mxArray��������ʾ��

//plhs��hrhs����ָ��mxArray���͵�ָ������

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])

{
    double *initialx = (double*)mxGetData(prhs[0]);
//     int numRowsB = (int)mxGetM(prhs[1]);//prhs[1]ָ��ڶ�������
//     int numColsB = (int)mxGetN(prhs[1]);

//mxGetData ��ȡ���������е�����
//     double*A = (double*)mxGetData(prhs[0]);
//     double*B = (double*)mxGetData(prhs[1]);
// �������������mxArray�ṹ��
    plhs[0]= mxCreateDoubleMatrix(100000,1, mxREAL);
    // plhs[0]= mxCreateNumericMatrix(4000,1, mxSINGLE_CLASS, mxREAL);
    plhs[1]=mxCreateDoubleMatrix(15,165,  mxREAL);
// ��ȡ���������ָ��
    double*y = (double*)mxGetData(plhs[0]);
    double*gbestx = (double*)mxGetData(plhs[1]);
// �����ӳ���
    
    Hopfield_syn_cuda(y,gbestx,initialx);
}