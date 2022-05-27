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
#define INITIALX_NUMBER 32*4096
#define D_NUMBER 2048*2048
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])

{
    double *tx = (double*)mxGetData(prhs[0]);
    double *td = (double*)mxGetData(prhs[1]);
    
    float initialx[INITIALX_NUMBER],d[D_NUMBER];
    int i;
    for(i=0;i<INITIALX_NUMBER;i++)
        initialx[i]=(float)tx[i];
     for(i=0;i<D_NUMBER;i++)
        d[i]=(float)td[i];
//     int numRowsB = (int)mxGetM(prhs[1]);//prhs[1]ָ��ڶ�������
//     int numColsB = (int)mxGetN(prhs[1]);

//mxGetData ��ȡ���������е�����
//     double*A = (double*)mxGetData(prhs[0]);
//     double*B = (double*)mxGetData(prhs[1]);
// �������������mxArray�ṹ��
    plhs[0]= mxCreateNumericMatrix(32,20000, mxSINGLE_CLASS,mxREAL);
    // plhs[0]= mxCreateNumericMatrix(4000,1, mxSINGLE_CLASS, mxREAL);
    plhs[1]=mxCreateNumericMatrix(2,2048,  mxSINGLE_CLASS,mxREAL);
// ��ȡ���������ָ��
    float*y = (float*)mxGetData(plhs[0]);
    float*gbestx = (float*)mxGetData(plhs[1]);
// �����ӳ���
    
   Hopfield_syn_cuda(d,y,gbestx,initialx);
}