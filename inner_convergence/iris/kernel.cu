#include "kernel.h"
#include "curand_kernel.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include "E:\Matlab R2018a\extern\include\mex.h"


#include <stdio.h>


#define N   150 //��������
#define P   3  //��������
#define POP 1 //��Ⱥ���� 
#define SIZE N*P*POP 
#define ALPHA 5 //�ͷ�����1
#define BETA  5//�ͷ�����2

//�������BLOCKS��HREAD_NUM 
#define BLOCKS POP
#define THREAD_NUM P //THREAD_NUM С�ڵ���POP����POP�ܳ���THREAD_NUM,THREAD_NUM���1024��THREAD_NUM�����32,һ�����ó�P*POP,���������1024������BLOCKS
//xתxt��HREAD_NUM����ʱ��ҪBLOCKS


#define PSO_THREAD_NUM 1 //����߳�����������ܳ���POP�����ó�POP
#define PSO_BLOCKS SIZE/PSO_THREAD_NUM 

#define RD_THREAD_NUM 1
#define RD_BLOCKS P*POP/RD_THREAD_NUM
#define W 1
#define C1 2
#define C2 2
#define STOPNUM 500

#define data "iris.txt"
#define U N/P
#define ITER 10
#define MAXITER 20000
//cublas����˷��ھ���,A[m,k], B[k,n],C[n,m],alpha=1.0,beta=0.0
//C=A*B
//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

//��ע
//���ڴ�Ƕȣ�c���Եľ����ǰ������У�cublas�ľ����ǰ�������
//x[n,p*pop],����n������������p�����pop����Ⱥ��С
//�����к�s1,�轫xת��xt[n*pop,p],I1[p,1],��������s1=xt*I1,s1[n*pop,1]
//�����к�s2,I2[1,n],s2=I2*x,s2[1,p*pop]
//����t=d*x,d[n,n],x[n,p*pop],t[p*pop,n]
//index[n,p*pop],�������¾���
void generate_rd_num(int* a)
{
	int i, j;

	int b[N];
	// int* b = (int*)malloc(sizeof(int) * N);
	for (i = 0; i < N; i++)
		a[i] = i;
	srand(time(NULL));
	int temp;
	for (i = 0; i < N; i++)
	{
		//j = (int)((double)((N - i) * rand()) / (RAND_MAX + 1.0));
		j = rand() % N;
		temp = a[i];
		a[i] = a[j];
		a[j] = temp;
	}
}

//�첽���£�����batch�����浽����index[n,p*pop]��index��ÿ���е���Ԫ����ͬ�����£�ÿ��ֻ���첽����
void generate_batch(int *index)
{
	int pos;
	int *block = (int*)malloc(sizeof(int) * N*P);
	int* rd = (int*)malloc(sizeof(int) * N);
	int i, j, loop;
	for (i = 0; i < N; i++)
		for (j = 0; j < P; j++)
			block[i*P + j] = (i*P + j * (P + 1)) % (N*P);

	//print_matrix_int(block, N, P);

	for (loop = 0; loop < POP; loop++)
	{
		generate_rd_num(rd);

		//for (int ii=0;ii<N;ii++)
		//	printf("%d ",rd[ii]);
		//	printf("\n");
		//	printf("\n");
		for (i = 0; i < N; i++)
			for (j = 0; j < P; j++)
			{
				index[i*P*POP + loop * P + j] = block[rd[i] * P + j];
			}
		//print_index(index, N, P,POP,loop);
	}
	free(rd);
	free(block);
}

__global__ static void tt()
{
}

//����ͬ������ 
__global__ static void parallel_updated_GPU(double *y,double *obj, double *lbest, double *lbestx, double*initialx, double *x, double *d, double *s1, double *s2, double *t, int *index, int gen, curandState *globalState)
{
	curandGenerator_t gen1;  //�������������
	int tid, i, k, b, j;
	int loop;
	//curandState state;
	//curand_init(seed, tid, 0, &state);
	tid = blockIdx.x *blockDim.x + threadIdx.x; //��ȡ�̺߳�0~blocks*THREAD_NUM-1
	int rd[N], temp, ped;
	curandState localState = globalState[tid];
	//tt<<<1,20>>>();
    int count=0;
	if (tid%P == 0)
	{

		//������һ�г�ʼ����
		int col = (curand(&localState)) % P;
		for (int i = 0; i < N; i++)
			rd[i] = i * P;
		//���ҵ�һ��

		for (int i = 0; i < N; i++)
		{


			j = (curand(&localState)) % N;
			temp = rd[i];
			rd[i] = rd[j];
			rd[j] = temp;
		}
		//printf("%d,%d\n", tid, rd[0]);
		//���������࣬��಻������P���Ҵ���P
		ped = P + 1;
		for (i = 0; i < N; i++)
			for (j = 0; j < P; j++)
				index[i*P*POP + tid + j] = (ped * j + rd[i]) % (N*P);
	}
	for (i = 0; i < N; i++)
	{
		x[i*P*POP + tid] = initialx[i*P*POP + tid];
	}
	__syncthreads();





	b = tid / P;
	double flag = 1.0;
	int it = 0;
	localState = globalState[tid / P];
	while (it < ITER)
	{
		flag = 0;
		//double total=0.0,const1=0.0,const2=0.0;
				//������һ�г�ʼ����
		for (int i = 0; i < N; i++)
			rd[i] = i;
		//���ҵ�һ��

		for (int i = 0; i < N; i++)
		{


			j = (curand(&localState)) % N;
			temp = rd[i];
			rd[i] = rd[j];
			rd[j] = temp;
		}
        double total = 0.0;
		for (loop = 0; loop < N; loop++)
		{

			temp = index[rd[loop] * POP*P + tid];
			i = temp / P;
			k = temp % P + b * P;

			//loopȡֵ0~N-1
			//int tid =  threadIdx.x;
			int pos = P * POP*i + k; //�����̺߳ż����index����
			double x_pos = x[pos];//��������ȡ��x�����ݣ�׼������x(pos)
			double half = 0.5;
			//double s1_i = s1[b*N + i];//����s1(i)
			double s1_i = s1[i*POP + b];//����s1(i)
			double s2_k = s2[k];//����s2(k)
			double dedx = 0.5*t[pos] + ALPHA * (s1_i - x_pos - 2 + P) + BETA * (s2_k - x_pos - 2 * U + N);
			//double dedx=t[k*N+i]+ALPHA*(s1_i-half-x_pos)+BETA*(s2_k-x_pos-U+half);
			//double dedx=ALPHA*(s1_i-half-x_pos)+BETA*(s2_k-x_pos-U+half);

			if (-dedx >= 0)
				x[pos] = 1.0;
			else
				x[pos] = -1.0;

			//��������Ż���if x[i][k]==0 t[k][j]=0
			//for(int j=0;j<N;j++)
				//t[k*N+j]=d[j*N+i]*x[pos];
			if (x[pos] != x_pos)
			{
				s1[i*POP + b] = s1_i - x_pos + x[pos];
				s2[k] = s2_k - x_pos + x[pos];
				double deltx = x[pos] - x_pos;
				for (int j = 0; j < N; j++)
					t[j*P*POP + k] = t[j*P*POP + k] + d[j*N + i] * deltx;
			}
			__syncthreads();
            if(tid==0)
            {
                double const1 = 0.0, const2 = 0.0;
                int ii,kk,id;
                for (kk = tid; kk < P + tid; kk++)
                for (ii = 0; ii < N; ii++)
                {
                    id = ii * P*POP + kk;
                        if (x[id] == 1.0)
                            total = total + t[id];
                        else
                            total = total - t[id];
                }
                for (int ii = 0; ii < N; ii++)
                    const1 = const1 + ALPHA * (s1[ii*POP + b] - 2 + P)*(s1[ii*POP + b] - 2 + P);
                for (int kk = 0; kk < P; kk++)
                    const2 = const2 + BETA * (s2[b*P + kk] - 2 * U + N)*(s2[b*P + kk] - 2 * U + N);
                total = 0.5*(total  + const1 + const2);
            }
			//flag = flag + (s1[i*POP + b] - 2 + P)*(s1[i*POP + b] - 2 + P) + (s2[k] - 2 * U + N)*(s2[k] - 2 * U + N);
		}
		__syncthreads();
        y[count]=total;
                count++;
		it++;
		//if (flag == 0)
		//	break;
	}
	//__syncthreads();
	if (tid%P == 0)
	{
		double total = 0.0, const1 = 0.0, const2 = 0.0;
		int id;
		for (k = tid; k < P + tid; k++)
			for (i = 0; i < N; i++)
			{
				id = i * P*POP + k;
				if (x[id] == 1.0)
					total = total + t[id];
				else
					total = total - t[id];
			}
		for (i = 0; i < N; i++)
			const1 = const1 + ALPHA * (s1[i*POP + b] - 2 + P)*(s1[i*POP + b] - 2 + P);
		for (k = 0; k < P; k++)
			const2 = const2 + BETA * (s2[b*P + k] - 2 * U + N)*(s2[b*P + k] - 2 * U + N);
		total = 0.5*(total  + const1 + const2);
		obj[tid / P] = total;
		//����ǵ�һ�ε�������ô������Ⱥ�õ���Ŀ�꺯��ֵ���϶������ŵ�
		if (gen == 0)
		{
			lbest[tid / P] = total;
			for (k = tid; k < tid + P; k++)
				for (i = 0; i < N; i++)
				{
					id = i * P*POP + k;
					lbestx[id] = x[id];
				}
		}
		else
		{
			if (obj[tid / P] < lbest[tid / P])
			{

				lbest[tid / P] = obj[tid / P];
				for (k = tid; k < tid + P; k++)
					for (i = 0; i < N; i++)
					{
						id = i * P*POP + k;
						lbestx[id] = x[id];
					}
			}
		}
	}



}


//���ļ���ȡd
void generate_d(double* d, char* str)
{
	FILE* fp;            /*�ļ�ָ��*/
	errno_t error;
	double* TemporaryD = d;
	error = fopen_s(&fp, str, "r");
	if (error != 0)
	{
		perror("fail to read");
		exit(1);
	}
	int i = 0;
	while (!feof(fp))
	{
		fscanf_s(fp, "%lf", &d[i++]);
		//fseek(fp, 1L, SEEK_CUR);   /*fpָ��ӵ�ǰλ������ƶ�*/
	}

	fclose(fp);                     //�ر��ļ�
}
void generate_s1(cublasHandle_t handle, double* s1, double *x, double* I1)
{

	//���Դ��е�xת����xt,һ��THREAD_NUM����POP�����һ���Կ����߳���������POP������forѭ������
	//for (int loop = 0; loop < REPEAT; loop++)
	//	trans_x_xt << <1, TRANS_THREAD_NUM >> > (xt, x, loop);


	//x[n,p*sizepop]
	//����s1,s1=x*I1,x���кͱ��浽s1

	const double alpha = 1.0f;
	const double beta = 0.0f;
	//�����к�s1,s1=xt*I1,I1[p,1],�൱��xt[n*pop,p]��A[m,k]��I1[p,1]��B[k,n]��
	//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, N*POP, P, &alpha, I1, 1, xt, P, &beta, s1, 1);
	//�����к�s1,s1=x*I1,I1[p,1],�൱��x[n,p*pop]��A[m,k]��I1[p*pop,pop]��B[k,n]��
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, POP, N, P*POP, &alpha, I1, POP, x, P*POP, &beta, s1, POP);
}

void generate_s2(cublasHandle_t handle, double* s2, double* x, double* I2)
{
	//x[n,p*sizepop]
	//����s1,s1=x*I1,x���кͱ��浽s1


	const double alpha = 1.0f;
	const double beta = 0.0f;
	//����s2,s2=I2*x,x���кͱ��浽s2
	//�����к�s1,s2=I2*x,I2[1,n],�൱��I2[1,n]��A[m,k]��x[n,p*pop]��B[k,n]��
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P*POP, 1, N, &alpha, x, P*POP, I2, N, &beta, s2, P*POP);

}

void generate_t(cublasHandle_t handle, double* t, double* d, double* x)
{
	const double alpha = 1.0f;
	const double beta = 0.0f;
	//����t=d*x,d[n,n],x[n,p*pop],t[p*pop,n],�൱��d[n,n]��A[m,k]��x[n,p*pop]��B[k,n]��
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, P*POP, N, N, &alpha, x, P*POP, d, N, &beta, t, P*POP);

}

void generate_I1(double* I)
{
	int i, j;
	for (i = 0; i < POP; i++)
	{
		for (j = 0; j < P*POP; j++)
			I[j*POP + i] = 0.0;
		for (j = i * P; j < (i + 1)*P; j++)
			I[j*POP + i] = 1.0;
	}
}
void generate_I2(double* I, int num)
{
	int i;
	for (i = 0; i < num; i++)
		I[i] = 1.0;
}

void cp_initialx_x(double *x,double *initialx)
{
	int i;
	double rd;
	srand((unsigned)time(NULL));
	for (i = 0; i < SIZE; i++) {

		rd = 2.0 * rand() / RAND_MAX - 1; //����-1��1�������
		if (rd > 0)
			x[i] = 1.0;
		else
			x[i] = -1.0;
	}
}
void initial_x(double* x)
{
	int i;
	double rd;
	srand((unsigned)time(NULL));
	for (i = 0; i < SIZE; i++) {

		rd = 2.0 * rand() / RAND_MAX - 1; //����-1��1�������
		if (rd > 0)
			x[i] = 1.0;
		else
			x[i] = -1.0;
	}
}

void initial_obj(double* x)
{
	int i;
	for (i = 0; i < POP; i++)
		x[i] = 0.0;
}

//����ȫ������Ŀ�꺯��ֵ�����Ž�,
__global__ static void generate_global_best(double *y, double *gbest, double *gbestx, double * lbest, double *lbestx, int it)
{

	int i, j, k, flag = 0;
	//������Ⱥ0��Ŀ�꺯��ֵ��С�Ҿ������Ž�
	double gb = lbest[0];
	int id = 0;
	//��ȡȫ�����Ž�
	for (i = 0; i < POP; i++)
		if (lbest[i] < gb)
		{
			gb = lbest[i];
			id = i;
		}
	if (it == 0 || gb < *gbest)
	{
		*gbest = gb;
		y[it] = gb;
		k = 0;
		int count = 0;
		for (j = 0; j < N; j++)
			for (i = id * P; i < (id + 1)*P; i++)
				gbestx[k++] = lbestx[j*P*POP + i];

		/*
			for (j = 0; j < N; j++)
			{
				for (i = id * P; i < (id + 1)*P; i++)
					printf("%f ", lbestx[j*P*POP + i]);
				printf("\n");
			}*/
	}
	else
	{
		y[it] = *gbest;
	}

	//printf("%d=%lf\n", it, y[it]);


}
__global__ static void pso(double *x, double *initialx, double *initialv, double *gbestx, double *lbestx, double *rd1, double *rd2, double *rd3)
{

	int i, k;
	int tid = blockIdx.x *blockDim.x + threadIdx.x; //��ȡ�̺߳�0~blocks*THREAD_NUM-1


	//printf("%f %f\n", rd1,rd2);
	//double rd1 = 0.7;
	//double rd2 = 0.5;
	/*
	if (tid == 0)
	{
		printf("\n");
		for (i = 0; i < N; i++)
		{
			for (k = 0; k < P; k++)
				printf("% f", initialx[i*P + k]);
			printf("\n");
		}
	}*/
	i = tid / (P*POP);
	k = tid % (P*POP) % P;

	/*
	   initial_v{j}=w1*initial_v{j}+beta1*rd1*(pbest_x{j}-initial_x{j})+beta2*rd2*(zbest_x-initial_x{j});
	   initial_x{j}=initial_x{j}+initial_v{j};
	   initial_x{j}=round(min(1,max(0,initial_x{j})));  %for zero one
	*/


	/*
	initialv[tid] = W * initialv[tid] + C1 * rd1[tid % (P*POP) / P] * (lbestx[tid] - initialx[tid]) + C2 * rd2[tid % (P*POP) / P] * (gbestx[i*P + k] - initialx[tid]);
	initialx[tid] = initialx[tid] + initialv[tid];
	if (initialx[tid] <= -1.0)
		initialx[tid] = -1.0;
	if (initialx[tid] >= 1.0)
		initialx[tid] = 1.0;
	if (initialx[tid] >= 0)
		initialx[tid] = 1.0;
	else
		initialx[tid] = -1.0;*/


		//initialv[tid] = W * initialv[tid] + C1 * rd1[tid % (P*POP) / P] * ((lbestx[tid] + 1) / 2.0 - (initialx[tid] + 1) / 2.0) + C2 * rd2[tid % (P*POP) / P] * ((gbestx[i*P + k] + 1) / 2.0 - (initialx[tid] + 1) / 2.0);
	initialv[tid] = W * initialv[tid] + C1 * rd1[tid] * ((lbestx[tid] + 1) / 2.0 - (initialx[tid] + 1) / 2.0) + C2 * rd1[tid] * ((gbestx[i*P + k] + 1) / 2.0 - (initialx[tid] + 1) / 2.0);
	double s;


	s = 1 / (1 + exp(-initialv[tid]));

	if (s > rd3[tid])
		initialx[tid] = 1.0;
	else
		initialx[tid] = -1.0;

	//if (5 / (N*P) > rd2[tid])
		//initialx[tid] = 1 - 2 * (1 - (initialx[tid] + 1) / 2.0);


}

void initial_v(double* x)
{
	int i;
	double rd;
	srand((unsigned)time(NULL));
	for (i = 0; i < SIZE; i++) {

		rd = 2.0 * rand() / RAND_MAX - 1; //����-1��1�������
		x[i] = rd;
		/*
				if (rd > 0)
					x[i] =0;
				else
					x[i] = 0;
		*/
	}
}
__global__ void setup_kernel(curandState *state, unsigned long seed)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x; //��ȡ�̺߳�0~blocks*THREAD_NUM-1
	curand_init(seed, tid, 0, &state[tid]);// initialize the state
}
__global__ void generate_rd1_rd2(double *rd1, double *rd2, double *rd3, curandState *globalState)
{
	int tid = blockIdx.x *blockDim.x + threadIdx.x; //��ȡ�̺߳�0~blocks*THREAD_NUM-1
	int i;
	curandState localState;
	if (tid%P == 0)
	{
		//localState = globalState[tid / P];
		//rd1[tid / P] = curand_uniform(&localState);
		//rd2[tid / P] = curand_uniform(&localState);
	}
	localState = globalState[tid];
	for (i = 0; i < N; i++)
	{
		rd1[i*P*POP + tid] = curand_uniform(&localState);
		rd2[i*P*POP + tid] = curand_uniform(&localState);
		rd3[i*P*POP + tid] = curand_uniform(&localState);
		//printf("%d %lf", i*P*POP + tid, rd3[i*P*POP + tid]);
	}
}
__global__ void stop(double *y, int *is_stop, int it)
{
	*is_stop = 0;
	if (y[it] == y[it - STOPNUM])
		*is_stop = 1;
}
int Hopfield_syn_cuda(double *y,double *gbestx,double *initialx)
{
	
	//ѡ���Կ�,ubuntu�û���3��2080ti�Կ������ϴ�ѧ��2��rtx8000�Կ�
	cudaSetDevice(1);


	double *xh, *xth, *xd, *xtd;
	xh = (double*)malloc(sizeof(double) * SIZE); //x���ڴ�����������ڴ�ռ�
	xth = (double*)malloc(sizeof(double)*SIZE); //xt���ڴ�����������ڴ�ռ�
	cp_initialx_x(xh,initialx);//printf("\n");print_matrix(xh, N, P*POP);//��ʼ��x
	cudaMalloc((void**)&xd, sizeof(double) * SIZE); //x���Դ�����������Դ�ռ�
	cudaMalloc((void**)&xtd, sizeof(double) * SIZE); //xtd���Դ�����������Դ�ռ�
	cudaMemcpy(xd, xh, sizeof(int) * SIZE, cudaMemcpyHostToDevice); //�ڴ��е�x���Ƶ��Դ�

	//���Կ����㣬��ʱ����#pragma omp parallel for num_threads(3)


	//����x��xt�Ƿ���ȷ����ʱ����
	/*print_matrix(xh, N, P*POP);
	cudaMemcpy(xth, xtd, sizeof(int)*SIZE, cudaMemcpyDeviceToHost); //�Դ����ݿ������ڴ�
	print_matrix(xth, N* POP, P);*/

	//����obj,index,I1��I2��s1��s2��t��d���ڴ����Դ���� 
	int *indexd, *indexh;
	double *sd1, *sd2, *sh1, *sh2, *Id1, *Id2, *Ih1, *Ih2, *td, *th, *dd, *dh;
	double *objd, *objh;
	double *initialxh, *initialxd, *initialvh, *initialvd;
	initialxh = (double*)malloc(sizeof(double)*SIZE);
	initialvh = (double*)malloc(sizeof(double)*SIZE); initial_v(initialvh);
	cudaMalloc((void**)&initialvd, sizeof(double) * SIZE);
	indexh = (int*)malloc(sizeof(int) *SIZE); generate_batch(indexh);//index���ڴ�����������ڴ�ռ�
	Ih1 = (double*)malloc(sizeof(double)  * POP*P*POP); generate_I1(Ih1);//I1���ڴ�����������ڴ�ռ�
	Ih2 = (double*)malloc(sizeof(double) * N); generate_I2(Ih2, N);//I2���ڴ�����������ڴ�ռ�
	sh1 = (double*)malloc(sizeof(double) * N*POP);//s1���ڴ�����������ڴ�ռ�
	sh2 = (double*)malloc(sizeof(double) * P*POP);//s2���ڴ�����������ڴ�ռ�
	th = (double*)malloc(sizeof(double) * SIZE);//t���ڴ�����������ڴ�ռ�
	dh = (double*)malloc(sizeof(double) * N*N); generate_d(dh, data);//d���ڴ�����������ڴ�ռ�
	objh = (double*)malloc(sizeof(double) * POP); initial_obj(objh);//obj���ڴ�����������ڴ�ռ�

	cudaMalloc((void**)&indexd, sizeof(int) * SIZE);//I1���ڴ�����������Դ�ռ�
	cudaMalloc((void**)&Id1, sizeof(double) * POP*P*POP);//I1���ڴ�����������Դ�ռ�
	cudaMalloc((void**)&Id2, sizeof(double) * N);//I2���Դ�����������Դ�ռ�
	cudaMalloc((void**)&sd1, sizeof(double) * N*POP);//I1���ڴ�����������Դ�ռ�
	cudaMalloc((void**)&sd2, sizeof(double) * P*POP);//I2���Դ�����������Դ�ռ�
	cudaMalloc((void**)&td, sizeof(double) * SIZE);//t���Դ�����������Դ�ռ�
	cudaMalloc((void**)&dd, sizeof(double) * N*N);//d���Դ�����������Դ�ռ�
	cudaMalloc((void**)&objd, sizeof(double) * POP);//obj���Դ�����������Դ�ռ�
	cudaMalloc((void**)&initialxd, sizeof(double) * SIZE);
	cudaMemcpy(initialxd, xh, sizeof(double) * SIZE, cudaMemcpyHostToDevice); //�ڴ��е�x���Ƶ��Դ�
	cudaMemcpy(initialvd, initialvh, sizeof(double) * SIZE, cudaMemcpyHostToDevice); //�ڴ��е�x���Ƶ��Դ�

	cudaMemcpy(indexd, indexh, sizeof(int) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(Id1, Ih1, sizeof(double) * POP*P*POP, cudaMemcpyHostToDevice);
	cudaMemcpy(Id2, Ih2, sizeof(double) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(td, th, sizeof(double) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dd, dh, sizeof(double) * N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(objd, objh, sizeof(double) * POP, cudaMemcpyHostToDevice);



	double *yh, *yd, *gbestd, *gbesth, *gbestxd, *gbestxh, *lbesth, *lbestxh, *lbestd, *lbestxd;
	yh = (double*)malloc(sizeof(double) * 100000);
	gbesth = (double*)malloc(sizeof(double));
	gbestxh = (double*)malloc(sizeof(double)*N*P);
	lbestxh = (double*)malloc(sizeof(double)*SIZE);
	lbesth = (double*)malloc(sizeof(double)*POP);
	cudaMalloc((void**)&yd, sizeof(double) * 100000);
	cudaMalloc((void**)&gbestd, sizeof(double));
	cudaMalloc((void**)&gbestxd, sizeof(double)*N*P);
	cudaMalloc((void**)&lbestd, sizeof(double)*POP);
	cudaMalloc((void**)&lbestxd, sizeof(double)*SIZE);

	double *rdh1, *rdh2, *rdh3;
	double *rdd1, *rdd2, *rdd3;
	rdh1 = (double*)malloc(sizeof(double) *SIZE);
	rdh2 = (double*)malloc(sizeof(double) *SIZE);
	rdh3 = (double*)malloc(sizeof(double) *SIZE);
	cudaMalloc((void**)&rdd1, sizeof(double) * SIZE);
	cudaMalloc((void**)&rdd2, sizeof(double) *SIZE);
	cudaMalloc((void**)&rdd3, sizeof(double) * SIZE);
	cublasHandle_t handle;
	cublasCreate(&handle);
	srand((unsigned int)time(NULL));
	curandState* devStates;

	cudaMalloc(&devStates, P*POP * sizeof(curandState));

	srand(time(0));
	int gen;
	int is_stoph, *is_stopd, stop_number = STOPNUM;
	cudaMalloc((void**)&is_stopd, sizeof(int));
	for (gen = 0; gen < 1; gen++)
	{



		generate_s1(handle, sd1, initialxd, Id1);
		generate_s2(handle, sd2, initialxd, Id2);
		generate_t(handle, td, dd, initialxd);
		//�������ӹ��࣬�����Ż���
		parallel_updated_GPU << <BLOCKS, THREAD_NUM >> > (yd,objd, lbestd, lbestxd, initialxd, xd, dd, sd1, sd2, td, indexd, gen, devStates);//����ÿ��

		

	}

	cudaMemcpy(gbestx, gbestxd, sizeof(double) * N*P, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, yd, sizeof(double)*100000, cudaMemcpyDeviceToHost);





	free(xh);
	free(indexh);
	free(sh1);
	free(sh2);
	free(Ih1);
	free(Ih2);
	free(th);
	free(dh);
	free(objh);
	free(initialxh);
	free(initialvh);
	free(yh);
	free(gbesth);
	free(gbestxh);
	free(lbesth);
	free(lbestxh);
	free(rdh1);
	free(rdh2);


	cudaFree(xd);
	cudaFree(indexd);
	cudaFree(sd1);
	cudaFree(sd2);
	cudaFree(Id1);
	cudaFree(Id2);
	cudaFree(td);
	cudaFree(dd);
	cudaFree(objd);
	cudaFree(initialxd);
	cudaFree(initialvd);
	cudaFree(yd);
	cudaFree(gbestd);
	cudaFree(gbestxd);
	cudaFree(lbestd);
	cudaFree(lbestxd);
	cudaFree(rdd1);
	cudaFree(rdd2);
	cudaFree(is_stopd);
	cublasDestroy(handle);
    return 0;
}