#include "cuPrintf.cuh"
#include "cuPrintf.cu"
#include<stdio.h>

#include<math.h>

#include<time.h>

#include<stdlib.h>

#include<cuda.h>

__device__ float su;

/////////////////////////////////NAIVE KERNEL///////////////////////////////////////////////////////////////////


__global__ void naive(float* x, float * y, int n, float* sumx, float* sumy, float* xiyi, float* xi2, float* yi2, float* sx, float* sy){
int index = blockIdx.x*blockDim.x + threadIdx.x;

float tempx=x[index];
float tempy=y[index];

if(index<n){
xi2[index]=tempx*tempx;
yi2[index]=tempy*tempy;
xiyi[index]=tempx*tempy;
__syncthreads();

for(int stride=n-1;stride>=1;stride=stride/2){
if(index<=stride/2 && index!=stride-index)
{
x[index]=x[index]+x[stride-index];
y[index]=y[index]+y[stride-index];
xiyi[index]=xiyi[index]+xiyi[stride-index];
xi2[index]=xi2[index]+xi2[stride-index];
yi2[index]=yi2[index]+yi2[stride-index];
}
__syncthreads();
}
                                   float xavg = x[0]/n;
float yavg = y[0]/n;

sx[index]=(tempx-xavg)*(tempx-xavg);
sy[index]=(tempy-yavg)*(tempy-yavg);
__syncthreads();

for(int stride=n-1;stride>=1;stride=stride/2){
if(index<=stride/2 && index!=stride-index)
{
sx[index]=sx[index]+sx[stride-index];
sy[index]=sy[index]+sy[stride-index];
}
__syncthreads();
}
}
}



////////////////////////////////////////SHARED KERNEL ///////////////////////////////////////////////
                                                                                                                                    
__global__ void shared(float *a,float *b,int n,int k,float *sumx,float *sumy,float *xi2,float *yi2,float *xiyi,float *sumxi2,float *sumyi2, float* sumxiyi){


     __shared__    float sharedx[1024];
     __shared__    float sharedy[1024];
                  int tid= threadIdx.x;
                 int i=blockIdx.x*blockDim.x+threadIdx.x;

if(i<n){
sharedx[tid]    = a[i];
sharedy[tid]    = b[i];
__syncthreads();
float tempx=a[i];
float tempy=b[i];
xi2[i]=tempx*tempx;
yi2[i]=tempy*tempy;
xiyi[i]=tempx*tempy;
 __syncthreads();
if(blockIdx.x==k){
for(int stride=n-1024*k-1;stride>=1;stride=stride/2){
if(tid<=stride/2 && tid!=stride-tid)
{
sharedx[tid]=sharedx[tid]+sharedx[stride-tid];
sharedy[tid]=sharedy[tid]+sharedy[stride-tid];
}


__syncthreads();
}
}
else
{     for(int stride=1023;stride>=1;stride=stride/2){

if(tid<=stride/2 && tid!=stride-tid)
{
sharedx[tid]=sharedx[tid]+sharedx[stride-tid];
sharedy[tid]=sharedy[tid]+sharedy[stride-tid];
}
 __syncthreads();
}
}
if(tid==0){
sumx[blockIdx.x]=sharedx[0];
sumy[blockIdx.x]=sharedy[0];
                        }
        __syncthreads();

                                  sharedx[tid]=xi2[i];

                                  sharedy[tid]=yi2[i];


                                        __syncthreads();

                                     if(blockIdx.x==k){
                                 for(int stride=n-1024*k-1;stride>=1;stride=stride/2){

                                      if(tid<=stride/2 && tid!=stride-tid)
                                        {
                                                 sharedx[tid]=sharedx[tid]+sharedx[stride-tid];
                                                 sharedy[tid]=sharedy[tid]+sharedy[stride-tid];
                                        }


                                                __syncthreads();
                                         }

                               }
                            else
                        {



                                for(int stride=1023;stride>=1;stride=stride/2){

                                      if(tid<=stride/2 && tid!=stride-tid)
                                        {
                                                 sharedx[tid]=sharedx[tid]+sharedx[stride-tid];
                                                 sharedy[tid]=sharedy[tid]+sharedy[stride-tid];
 }


                                                __syncthreads();
                                         }


                                        }

                      if(tid==0)
                        {

                              sumxi2[blockIdx.x]=sharedx[0];
                              sumyi2[blockIdx.x]=sharedy[0];


                        }
__syncthreads();

                     sharedx[tid]=xiyi[i];

                    __syncthreads();

                                    if(blockIdx.x==k){
                          for(int stride=n-1024*k-1;stride>=1;stride=stride/2){

                                      if(tid<=stride/2 && tid!=stride-tid)
                                        {
                                                 sharedx[tid]=sharedx[tid]+sharedx[stride-tid];

                                        }


                                                __syncthreads();
                                         }
                                 }
                             else
                                {
    for(int stride=1023;stride>=1;stride=stride/2){

                                      if(tid<=stride/2 && tid!=stride-tid)
                                        {
                                                 sharedx[tid]=sharedx[tid]+sharedx[stride-tid];
                                                 sharedy[tid]=sharedy[tid]+sharedy[stride-tid];
                                        }


                                                __syncthreads();
                                         }

                                        }

                      if(tid==0)
                        {

                              sumxiyi[blockIdx.x]=sharedx[0];

                        }


}

}





//////////////////////////////////////SERIAL FUNCTIONS///////////////////////////////////


float uniform(float a, float b)

{
return rand() / (RAND_MAX + 1.0) * (b - a) + a;
}

////////////////////////////////////////////////////////////////////////////////////////



int main(){



float* xavg;
xavg=(float*)malloc(sizeof(float));


float* yavg;
yavg=(float*)malloc(sizeof(float));



float* sumx;
sumx=(float*)malloc(sizeof(float));


float* sumy;
sumy=(float*)malloc(sizeof(float));


float* xiyi;
xiyi=(float*)malloc(sizeof(float));



float* xi2;
xi2=(float*)malloc(sizeof(float));


float* yi2;
yi2=(float*)malloc(sizeof(float));
float* crc;
crc=(float*)malloc(sizeof(float));



float* sx;
sx=(float*)malloc(sizeof(float));


float* sy;
sy=(float*)malloc(sizeof(float));
float* m;

m=(float*)malloc(sizeof(float));

float* b;
b=(float*)malloc(sizeof(float));




time_t t;

srand((unsigned) time(&t));



int n;



scanf("%d",&n);



float *x;
x=(float *)malloc(n*sizeof(float));
float *y;
y=(float *)malloc(n*sizeof(float));
for(int i=0;i<n;i++) {

x[i]=uniform(1.0,1000.0);

y[i]=uniform(1.0,1000.0);


}



////////copying to file/////////////////////////////////////

FILE *fp;



   fp = fopen("data.txt", "w+");

for(int i=0;i<n;i++){



fprintf(fp,"%f ,  %f\n ",x[i],y[i]);



}



fclose(fp);

//printf("%f",x[0]);

////////////////////closing file/////////////////////////

//////////////////serial code///////////////////////////



cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

for(int i=0;i<n;i++){

*sumx+=x[i];
*sumy+=y[i];
*xiyi+=x[i]*y[i];
*xi2+=x[i]*x[i];
*yi2+=y[i]*y[i];

}

*xavg=*sumx/n;
*yavg=*sumy/n;

*crc = ((*xiyi)-(n*(*xavg)*(*yavg)))/(sqrt((*xi2)-n*(*xavg)*(*xavg))*sqrt((*yi2)-n*(*yavg)*(*yavg)));

printf("serial crc%f\n",*crc);

for(int i=0;i<n;i++){

*sx+=(x[i]-*xavg)*(x[i]-*xavg);
*sy+=(y[i]-*yavg)*(y[i]-*yavg);

}
*sx=sqrt(*sx/n);
*sy=sqrt(*sy/n);
*m=(*crc)*(*sy/(*sx));
*b=(*yavg)-(*m)*(*xavg);
printf("serial regressiony=%fx+%f\n",*m,*b);


cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("millis %f\n",milliseconds);




////////////////////////////////////////////////////////////
///////////////////////naive parallel//////////////////////



cudaEvent_t start1, stop1;


float *dev_x;

float *dev_y;

float *dev_sumx;

float *dev_sumy;

float *dev_xiyi;

float *dev_xi2;

float *dev_yi2;

float *dev_sx;

float *dev_sy;

//  float* sumx, float* sumy, float* xiyi, float* xi2, float* yi2, float* sx, float* sy



xiyi=(float*)malloc(n*sizeof(float));
xi2=(float*)malloc(n*sizeof(float));
yi2=(float*)malloc(n*sizeof(float));
sx=(float*)malloc(n*sizeof(float));
sy=(float*)malloc(n*sizeof(float));




cudaMalloc((void**)&dev_x,n*sizeof(float));

cudaMalloc((void**)&dev_y,n*sizeof(float));

cudaMalloc((void**)&dev_sumx,sizeof(float));

cudaMalloc((void**)&dev_sumy,sizeof(float));

cudaMalloc((void**)&dev_xiyi,n*sizeof(float));

cudaMalloc((void**)&dev_xi2,n*sizeof(float));

cudaMalloc((void**)&dev_yi2,n*sizeof(float));

cudaMalloc((void**)&dev_sx,n*sizeof(float));

cudaMalloc((void**)&dev_sy,n*sizeof(float));


cudaMemcpy(dev_x,x,n*sizeof(float),cudaMemcpyHostToDevice);

cudaMemcpy(dev_y,y,n*sizeof(float),cudaMemcpyHostToDevice);


//printf("%f\n",dev_x[0]);

cudaEventCreate(&start1);
cudaEventCreate(&stop1);

cudaEventRecord(start1);

naive<<<(n/1024)+1,1024>>>(dev_x,dev_y,n,dev_sumx,dev_sumy,dev_xiyi,dev_xi2,dev_yi2,dev_sx,dev_sy);


cudaEventRecord(stop1);

cudaEventSynchronize(stop1);
float milliseconds1 = 0;
cudaEventElapsedTime(&milliseconds1, start1, stop1);
printf("naive parallel millis %f\n",milliseconds1);



cudaMemcpy(sumx,&dev_x[0],sizeof(float),cudaMemcpyDeviceToHost);

cudaMemcpy(sumy,&dev_y[0],sizeof(float),cudaMemcpyDeviceToHost);

cudaMemcpy(xiyi,&dev_xiyi[0],sizeof(float),cudaMemcpyDeviceToHost);

cudaMemcpy(xi2,&dev_xi2[0],sizeof(float),cudaMemcpyDeviceToHost);

cudaMemcpy(yi2,&dev_yi2[0],sizeof(float),cudaMemcpyDeviceToHost);

cudaMemcpy(sy,&dev_sy[0],sizeof(float),cudaMemcpyDeviceToHost);

cudaMemcpy(sx,&dev_sx[0],sizeof(float),cudaMemcpyDeviceToHost);

//cudaMemcpy(sy,dev_sy,sizeof(float),cudaMemcpyDeviceToHost);

*sx=sqrt(*sx/n);
*sy=sqrt(*sy/n);

/*
printf("\n%f\n",*sumx);
//printf("\n%f\n",*sumy);
printf("\n%f\n",*xiyi);
printf("\n%f\n",*xi2);
printf("%f\n",*yi2);
printf("%f\n",*sx);
printf("%f\n",*sy);
*/
*crc = ((*xiyi)-(*sumx)*((*sumy)/n))/(sqrt(*xi2-(*sumx)*((*sumx)/n))*sqrt(*yi2-(*sumy)*((*sumy)/n)));



printf("naive %f\n",*crc);

*m=(*crc)*(*sy/(*sx));
*b=(*sumy)/n-(*m)*(*sumx)/n;



printf("naive regression y=%fx+%f\n",*m,*b);



cudaFree(dev_x);
cudaFree(dev_y);
cudaFree(dev_sumx);
cudaFree(dev_sumy);
cudaFree(dev_xi2);
cudaFree(dev_yi2);
cudaFree(dev_sx);
cudaFree(dev_sy);
cudaFree(dev_xiyi);






///////////////////////////////////SHARED MEMORY IMPLEMENTATIION/////////////////////////////////////////

cudaEvent_t start2, stop2;


*sx=0;
*sy=0;
*crc=0;
*m=0;
*b=0;



/*
float *dev_x1;

float *dev_y1;*/

float *dev_sumxi2;
float *dev_sumyi2;
float *dev_sumx1;
float *dev_sumy1;
float *dev_sumxiyi1;
//float *dev_sumxiyi;
cudaMalloc((void**)&dev_x,n*sizeof(float));

cudaMalloc((void**)&dev_y,n*sizeof(float));

cudaMalloc((void**)&dev_sumx1,((n/1024)+1)*sizeof(float));

cudaMalloc((void**)&dev_sumy1,((n/1024)+1)*sizeof(float));

//cudaMalloc((void**)&dev_xiyi1,((n/1024)+1)*sizeof(float));



cudaMalloc((void**)&dev_xi2,n*sizeof(float));
cudaMalloc((void**)&dev_yi2,n*sizeof(float));
cudaMalloc((void**)&dev_xiyi,n*sizeof(float));






cudaMalloc((void**)&dev_sumxi2,((n/1024)+1)*sizeof(float));

cudaMalloc((void**)&dev_sumyi2,((n/1024)+1)*sizeof(float));
cudaMalloc((void**)&dev_sumxiyi1,((n/1024)+1)*sizeof(float));


//cudaMalloc((void**)&dev_sx,sizeof(float));

//cudaMalloc((void**)&dev_sy,sizeof(float));

cudaMemcpy(dev_x,x,n*sizeof(float),cudaMemcpyHostToDevice);

cudaMemcpy(dev_y,y,n*sizeof(float),cudaMemcpyHostToDevice);



int k=floor(n/1024);
//printf("floor %d\n",k);
//shared(float *a,float *b,int n,float *sumx,float *sumy)


cudaEventCreate(&start2);
cudaEventCreate(&stop2);

cudaEventRecord(start2);

shared<<<(n/1024)+1,1024>>>(dev_x,dev_y,n,k,dev_sumx1,dev_sumy1,dev_xi2,dev_yi2,dev_xiyi,dev_sumxi2,dev_sumyi2,dev_sumxiyi1);



cudaEventRecord(stop2);

cudaEventSynchronize(stop2);
float milliseconds2 = 0;
cudaEventElapsedTime(&milliseconds2, start2, stop2);
printf("shared parallel millis %f\n",milliseconds2);





float *p;
p=(float *)malloc(((n/1024)+1)*sizeof(float));

float *q;
q=(float *)malloc(((n/1024)+1)*sizeof(float));

float *r;
r=(float *)malloc(((n/1024)+1)*sizeof(float));

float *s;
s=(float *)malloc(((n/1024)+1)*sizeof(float));

float *z;
z=(float *)malloc(((n/1024)+1)*sizeof(float));

cudaMemcpy(p,dev_sumx,((n/1024)+1)*sizeof(float),cudaMemcpyDeviceToHost);

cudaMemcpy(q,dev_sumy,((n/1024)+1)*sizeof(float),cudaMemcpyDeviceToHost);

cudaMemcpy(r,dev_sumxi2,((n/1024)+1)*sizeof(float),cudaMemcpyDeviceToHost);

cudaMemcpy(s,dev_sumyi2,((n/1024)+1)*sizeof(float),cudaMemcpyDeviceToHost);

cudaMemcpy(z,dev_sumxiyi1,((n/1024)+1)*sizeof(float),cudaMemcpyDeviceToHost);
float finalsumx=0;
float  finalsumy=0;
float finalxiyi=0;
float finalxi2=0;
float finalyi2=0;
for(int i=0;i<=k;i++)
{
finalsumx+=p[i];
finalsumy+=q[i];
finalxiyi+=z[i];
finalxi2+=r[i];
finalyi2+=s[i];

}


float finalxavg=finalsumx/n;
float finalyavg=finalsumy/n;


//float crc1;


//*crc = ((*xiyi)-(n*(*xavg)*(*yavg)))/(sqrt((*xi2)-n*(*xavg)*(*xavg))*sqrt((*yi2)-n*(*yavg)*(*yavg)));




*crc = (finalxiyi-(n*finalxavg*finalyavg))/(sqrt(finalxi2-n*finalxavg*finalxavg)*sqrt(finalyi2-n*finalyavg*finalyavg));

printf("shared crc %f\n",*crc);


for(int i=0;i<n;i++){



*sx+=(x[i]-finalxavg)*(x[i]-finalxavg);

*sy+=(y[i]-finalyavg)*(y[i]-finalyavg);


}
*sx=sqrt(*sx/n);

*sy=sqrt(*sy/n);



*m=(*crc)*(*sy/(*sx));

*b=finalyavg-(*m)*finalxavg;



printf("shared regression y=%fx+%f\n",*m,*b);





}











