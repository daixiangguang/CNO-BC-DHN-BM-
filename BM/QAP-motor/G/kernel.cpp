#include"mex.h"

#include"kernel.h"

// nlhs: 输出变量的个数(lefthand side,调用语句的左手面)

// plhs：输出的mxArray矩阵的头指针

// nrhs: 输入变量个数(righthand side,调用语句的右手面)

// prhs：输入的mxArray矩阵的头指针

// 如果有两个输入变量，那么prhs[0]指向第一个变量

//prhs[1]指向第二个变量

// Matlab的array使用mxArray类型来表示。

//plhs和hrhs都是指向mxArray类型的指针数组
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
//     int numRowsB = (int)mxGetM(prhs[1]);//prhs[1]指向第二个变量
//     int numColsB = (int)mxGetN(prhs[1]);

//mxGetData 获取数据阵列中的数据
//     double*A = (double*)mxGetData(prhs[0]);
//     double*B = (double*)mxGetData(prhs[1]);
// 生成输入参数的mxArray结构体
        //mexPrintf("%f %f %f",initialx[0],initialx[1],initialx[2]);
    //mexPrintf("%f %f %f",d[0],d[1],d[2]);
    plhs[0]= mxCreateNumericMatrix(100000,1, mxSINGLE_CLASS,mxREAL);

    plhs[1]=mxCreateNumericMatrix(2,2048,  mxSINGLE_CLASS,mxREAL);
// 获取输出参数的指针

         float*y = (float*)mxGetData(plhs[0]);
    float*gbestx = (float*)mxGetData(plhs[1]);
    
    /*double*y ;
    double*gbestx;
  y = mxGetPr( plhs[0]);
  gbestx = mxGetPr( plhs[1]);*/
   // y[0]=1;
   Hopfield_syn_cuda(d,y,gbestx,initialx,M);

}