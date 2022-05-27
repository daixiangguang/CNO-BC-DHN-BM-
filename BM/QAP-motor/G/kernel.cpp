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
#define INITIALX_NUMBER 1*4096
#define D_NUMBER 2048*2048
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])

{
    double *tx = (double*)mxGetData(prhs[0]);
    double *td = (double*)mxGetData(prhs[1]);
    
    float initialx[INITIALX_NUMBER],d[D_NUMBER];
    int i;
    for(i=0;i<INITIALX_NUMBER;i++)
        initialx[i]=(float)tx[i];
     for(i=0;i<D_NUMBER;i++)
        d[i]=(float)td[i];
    float M=mxGetScalar(prhs[2]);
    //mexPrintf("%f",M);
//     int numRowsB = (int)mxGetM(prhs[1]);//prhs[1]ָ��ڶ�������
//     int numColsB = (int)mxGetN(prhs[1]);

//mxGetData ��ȡ���������е�����
//     double*A = (double*)mxGetData(prhs[0]);
//     double*B = (double*)mxGetData(prhs[1]);
// �������������mxArray�ṹ��
        //mexPrintf("%f %f %f",initialx[0],initialx[1],initialx[2]);
    //mexPrintf("%f %f %f",d[0],d[1],d[2]);
    plhs[0]= mxCreateNumericMatrix(100000,1, mxSINGLE_CLASS,mxREAL);

    plhs[1]=mxCreateNumericMatrix(2,2048,  mxSINGLE_CLASS,mxREAL);
// ��ȡ���������ָ��

         float*y = (float*)mxGetData(plhs[0]);
    float*gbestx = (float*)mxGetData(plhs[1]);
    
    /*double*y ;
    double*gbestx;
  y = mxGetPr( plhs[0]);
  gbestx = mxGetPr( plhs[1]);*/
   // y[0]=1;
   Hopfield_syn_cuda(d,y,gbestx,initialx,M);

}