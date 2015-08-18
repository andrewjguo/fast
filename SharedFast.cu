
#include <iostream>
#include <opencv2/opencv.hpp>
#include <helper_cuda.h>
#include <timer.h>
#include <string>
#include "corner.h"
#include "fast_cuda.h"

//__device__
//int position(int m,int n,int width)
//{
//	int idx=m+n*width;
//	return idx;
//}

//__global__
//void fast(uchar* image, int width, int height,Corner* d_corner,int gridsize_x, int gridsize_y, const int threshold)
//{
//	__shared__ uchar patch[22][22];
//	uint sp=0,sn=0;
//	int m=blockDim.x*blockIdx.x+threadIdx.x;
//	int n=blockDim.y*blockIdx.y+threadIdx.y;
//	uint idx = m+n*width;
//	uint idx_block=threadIdx.y*blockDim.x+threadIdx.x;               //offset for pixel in patch
//	d_corner[idx]={0,0};                                            //initialize the class member
//	int patch_top_x=blockDim.x*blockIdx.x-3;
//	int patch_top_y=blockDim.y*blockIdx.y-3;
//	int idx_block_256=idx_block+256;
//
//    //load into patch
//	patch[idx_block%22][idx_block/22]=image[position(patch_top_x+idx_block%22,patch_top_y+idx_block/22,width)];
//	if(idx_block_256<484)
//		patch[(idx_block+256)%22][(idx_block+256)/22]=image[position(patch_top_x+idx_block_256%22,patch_top_y+idx_block_256/22,width)];
//	__syncthreads();
//
//	//detect
//	int x=3+threadIdx.x;
//    int y=3+threadIdx.y;
//	if(m>2&&m<(width-3)&&n>2&&n<(height-3))            //detect the points away from the edges
//	{
//		uchar center_value = patch[x][y];
//		sp |=(patch[x][y-3]>(center_value + threshold))<< 0;
//		sp |=(patch[x+1][y-3]>(center_value + threshold))<< 1;
//		sp |=(patch[x+2][y-2]>(center_value + threshold))<< 2;
//		sp |=(patch[x+3][y-1]>(center_value + threshold))<< 3;
//		sp |=(patch[x+3][y]>(center_value + threshold))<< 4;
//		sp |=(patch[x+3][y+1]>(center_value + threshold))<< 5;
//		sp |=(patch[x+2][y+2]>(center_value + threshold))<< 6;
//		sp |=(patch[x+1][y+3]>(center_value + threshold))<< 7;
//		sp |=(patch[x][y+3]>(center_value + threshold))<< 8;
//		sp |=(patch[x-1][y+3]>(center_value + threshold))<< 9;
//		sp |=(patch[x-2][y+2]>(center_value + threshold))<< 10;
//		sp |=(patch[x-3][y+1]>(center_value + threshold))<< 11;
//		sp |=(patch[x-3][y]>(center_value + threshold))<< 12;
//		sp |=(patch[x-3][y-1]>(center_value + threshold))<< 13;
//		sp |=(patch[x-2][y-2]>(center_value + threshold))<< 14;
//		sp |=(patch[x-1][y-3]>(center_value + threshold))<< 15;
//
//		sp+=sp<<16;
//		uint sp1=sp&(sp<<1);
//		uint sp2=sp1&(sp1<<2);
//		uint sp3=sp2&(sp2<<4);
//		uint sp4=sp3&(sp<<8);
//		if(sp4!=0)
//		{
//			int value=abs(center_value-patch[x-1][y-1])+abs(center_value-patch[x][y-1])+abs(center_value-patch[x+1][y-1])+
//					abs(center_value-patch[x-1][y])+abs(center_value-patch[x+1][y])+abs(center_value-patch[x+1][y-1])+
//					abs(center_value-patch[x+1][y])+abs(center_value-patch[x+1][y+1]);
//			d_corner[idx].value=value;
//			d_corner[idx].set=1;
//		}
//		else
//		{
//			sn |=(patch[x][y-3]<(center_value - threshold))<< 0;
//			sn |=(patch[x+1][y-3]<(center_value - threshold))<< 1;
//			sn |=(patch[x+2][y-2]<(center_value - threshold))<< 2;
//			sn |=(patch[x+3][y-1]<(center_value - threshold))<< 3;
//			sn |=(patch[x+3][y]<(center_value - threshold))<< 4;
//			sn |=(patch[x+3][y+1]<(center_value - threshold))<< 5;
//			sn |=(patch[x+2][y+2]<(center_value - threshold))<< 6;
//			sn |=(patch[x+1][y+3]<(center_value - threshold))<< 7;
//			sn |=(patch[x][y+3]>(center_value - threshold))<< 8;
//			sn |=(patch[x-1][y+3]<(center_value - threshold))<< 9;
//			sn |=(patch[x-2][y+2]<(center_value - threshold))<< 10;
//			sn |=(patch[x-3][y+1]<(center_value - threshold))<< 11;
//			sn |=(patch[x-3][y]<(center_value - threshold))<< 12;
//			sn |=(patch[x-3][y-1]<(center_value - threshold))<< 13;
//			sn |=(patch[x-2][y-2]<(center_value - threshold))<< 14;
//			sn |=(patch[x-1][y-3]<(center_value - threshold))<< 15;
//			sn+=sn<<16;
//			uint sn1=sn&(sn<<1);
//			uint sn2=sn1&(sn1<<2);
//			uint sn3=sn2&(sn2<<4);
//			uint sn4=sn3&(sn<<8);
//			if(sn4!=0)
//			{
//				int value=abs(center_value-patch[x-1][y-1])+abs(center_value-patch[x][y-1])+abs(center_value-patch[x+1][y-1])+
//						abs(center_value-patch[x-1][y])+abs(center_value-patch[x+1][y])+abs(center_value-patch[x+1][y-1])+
//						abs(center_value-patch[x+1][y])+abs(center_value-patch[x+1][y+1]);
//				d_corner[idx].value=value;
//				d_corner[idx].set=1;
//			}
//		}
//	}
//
//}
//__global__
//void nms(uchar* image, Corner* d_corner,int width, int height)
//{
//	int m=blockDim.x*blockIdx.x+threadIdx.x;
//	int n=blockDim.y*blockIdx.y+threadIdx.y;
//	int idx=n*width+m;
//	if(d_corner[idx].set==1)
//	{
//		if(d_corner[position(m-1,n-1,width)].value>d_corner[idx].value)
//		{d_corner[idx].set=0;return;}
//		if(d_corner[position(m,n-1,width)].value>d_corner[idx].value)
//		{d_corner[idx].set=0;return;}
//		if(d_corner[position(m+1,n-1,width)].value>d_corner[idx].value)
//		{d_corner[idx].set=0;return;}
//		if(d_corner[position(m-1,n,width)].value>d_corner[idx].value)
//		{d_corner[idx].set=0;return;}
//		if(d_corner[position(m+1,n,width)].value>d_corner[idx].value)
//		{d_corner[idx].set=0;return;}
//		if(d_corner[position(m+1,n-1,width)].value>d_corner[idx].value)
//		{d_corner[idx].set=0;return;}
//		if(d_corner[position(m+1,n,width)].value>d_corner[idx].value)
//		{d_corner[idx].set=0;return;}
//		if(d_corner[position(m+1,n+1,width)].value>d_corner[idx].value)
//		{d_corner[idx].set=0;return;}
//
//	}
//}

int main( void )
{
	using namespace std;
	using namespace cv;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    const int threshold=100;
    string filename="/Users/macbookpro/Downloads/monash.jpg";
	Mat image;
	image = cv::imread(filename,0);   // Read the file
	if(! image.data )                              // Check for invalid input
	{
		cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}
	uchar* d_data;                // create a pointer
	size_t imSize=image.cols*image.rows;
	Corner* h_corner=new Corner[imSize];
	Corner* d_corner;
	checkCudaErrors(cudaMalloc((void**) &d_corner,sizeof(Corner)*imSize));
	checkCudaErrors(cudaMalloc((void**) &d_data, sizeof(uchar)*imSize)); // create memory on the gpu and pass a pointer to the host
	checkCudaErrors(cudaMemcpy(d_data, image.data, sizeof(uchar)*imSize, cudaMemcpyHostToDevice));// copy from the image data to the gpu memory you reserved
	dim3 blocksize(16,16);
	dim3 gridsize((image.cols-1)/blocksize.x+1, (image.rows-1)/blocksize.y+1, 1);
	cudaEventRecord(start);
	fast<<<gridsize,blocksize>>>(d_data, image.cols, image.rows,d_corner,gridsize.x,gridsize.y,threshold); // processed data on the gpu
	//checkCudaErrors(cudaDeviceSynchronize());
	cudaEventRecord(stop);	cudaEventSynchronize(stop);
	nms<<<gridsize,blocksize>>>(d_data,d_corner,image.cols,image.rows);
	checkCudaErrors(cudaMemcpy(h_corner,d_corner,sizeof(Corner)*imSize,cudaMemcpyDeviceToHost));
	float elptime;
	cudaEventElapsedTime(&elptime,start,stop);
	//show the corner in the image
	Mat image_color = imread(filename,1);
    int point=0;
	for(int i=0;i<imSize;i++)
	{
		if(h_corner[i].set!=0)
		{
			int x=i%image.cols;
			int y=i/image.cols;
			circle(image_color,Point(x,y),1,Scalar(0,255,0),-1,8,0);
			point++;
		}
	}
	cout<<"points:"<<point<<endl;
	cout<<"Elapsed time:"<<elptime<<"ms"<<endl;
	//printf("%x\n",0x7|((10>1)<<3));
	//cout<<"the size of: "<<sizeof(corner)<<endl;
	namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
	imshow( "Display window", image_color );                   // Show our image inside it.
	waitKey(0);                                          // Wait for a keystroke in the window
	delete[] h_corner;
	cudaFree(d_corner);
	cudaFree(d_data);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;
}
