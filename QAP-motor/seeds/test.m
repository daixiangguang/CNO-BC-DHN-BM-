
system('nvcc -c kernel.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64"')
mex kernel.cpp kernel.obj  -lcublas -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64"


%A=[3 2 3;7 8 9];
%A=gpuArray(A);
%B=[4 5 6];
 %C = AddVectorsCuda(A',B);

