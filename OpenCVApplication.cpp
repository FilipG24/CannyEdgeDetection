// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <queue>
#include <random>
#define point pair<int,int>

using namespace std;

Mat Color2Gray(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height,width,CV_8UC1);

	for (int i=0; i<height; i++)
	{
		for (int j=0; j<width; j++)
		{
			Vec3b v3 = src.at<Vec3b>(i,j);
			uchar b = v3[0];
			uchar g = v3[1];
			uchar r = v3[2];
			dst.at<uchar>(i,j) = (r+g+b)/3;
		}
	}

	return dst;
		
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}


void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Mat_<uchar> pad(int k, Mat_<uchar> img)
{
	Mat copy(img.rows + k, img.cols + k, CV_8UC1);
	for (int i = 0; i < copy.rows; i++)
		for (int j = 0; j < copy.cols; j++)
			copy.at<uchar>(i, j) = 0;
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			copy.at<uchar>(i + k / 2, j + k / 2) = img.at<uchar>(i, j);
	return copy;
}

void convolution(Mat_<float> &filter, Mat_<uchar> &img, Mat_<uchar> &output)
{
	output.create(img.size());
	output.setTo(0);
	float scalingCoeff = 1;
	float additionFactor = 0;
	int pos_elem = 0;
	int neg_elem = 0;
	float pos_sum = 0;
	float neg_sum = 0;
	for (int i = 0; i < filter.rows; i++)
	{
		for (int j = 0; j < filter.cols; j++)
		{
			if (filter.at<float>(i, j) >= 0)
			{
				pos_elem++;
				pos_sum += filter.at<float>(i, j);
			}
			else
			{
				neg_elem++;
				neg_sum += filter.at<float>(i, j);
			}
		}
	}
	if (pos_elem == filter.rows*filter.rows)
	{ //low pass
		additionFactor = 0;
		scalingCoeff = pos_sum;
	}
	else
	{ // highpass
		if (pos_sum > abs(neg_sum))
			scalingCoeff = 2 * pos_sum;
		else
			scalingCoeff = 2 * abs(neg_sum);
		additionFactor = 127;
	}
	int di[9] = { -1,-1,-1,0,0,0,1,1,1 };
	int dj[9] = { -1,0,1,-1,0,1,-1,0,1 };


	for (int i = filter.rows / 2; i < img.rows - filter.rows / 2; i++) {
		for (int j = filter.rows / 2; j < img.cols - filter.rows / 2; j++) {
			float sum = 0;
			for (int k = 0; k < filter.rows; k++)
				for (int l = 0; l < filter.cols; l++)
					sum += img(i + k - filter.rows / 2, j + l - filter.cols / 2) * filter(k, l);
			sum = min(abs(sum / scalingCoeff), 255);
			output(i, j) = sum;
		}
	}
}

Mat gaussianFilter(Mat img)
{
	double t = (double)getTickCount();
	int dimension = 3;
	img = pad(dimension, img);
	Mat img_filtered;
	img.copyTo(img_filtered);
		
	float sigma = dimension / 6.0;
	Mat_<float> filter(dimension, dimension);
	for (int i = 0; i < dimension; i++)
		for (int j = 0; j < dimension; j++) {
			filter(i, j) = exp(-(pow(i - dimension / 2, 2) + pow(j - dimension / 2, 2)) / (2 * sigma*sigma)) / (2 * PI*sigma*sigma);
			t = ((double)getTickCount() - t) / getTickFrequency();
		}
	convolution(filter, (Mat_<uchar>)img, (Mat_<uchar>)img_filtered);
	
	imshow("filtered", img_filtered);
	
	return img_filtered;
}

void fillWithWhite(Mat result, Mat src, Mat visited, int i, int j)
{
	if (i >= 0 && j >= 0 && i < src.rows && j < src.cols)
		if (src.at<uchar>(i, j) != 0 && visited.at<uchar>(i, j) == 0)
		{
			visited.at<uchar>(i, j) = 255;
			result.at<uchar>(i, j) = 255;
			fillWithWhite(result, src, visited, i + 1, j);
			fillWithWhite(result, src, visited, i - 1, j);
			fillWithWhite(result, src, visited, i, j + 1);
			fillWithWhite(result, src, visited, i, j - 1);
			fillWithWhite(result, src, visited, i + 1, j - 1);
			fillWithWhite(result, src, visited, i - 1, j - 1);
			fillWithWhite(result, src, visited, i + 1, j + 1);
			fillWithWhite(result, src, visited, i - 1, j + 1);
		}
}

void fillWithBlack(Mat result, Mat src, Mat visited, int i, int j)
{
	if (i >= 0 && j >= 0 && i < src.rows && j < src.cols)
		if (src.at<uchar>(i, j) != 0 && visited.at<uchar>(i, j) == 0)
		{
			visited.at<uchar>(i, j) = 255;
			result.at<uchar>(i, j) = 0;
			fillWithBlack(result, src, visited, i + 1, j);
			fillWithBlack(result, src, visited, i - 1, j);
			fillWithBlack(result, src, visited, i, j + 1);
			fillWithBlack(result, src, visited, i, j - 1);
			fillWithBlack(result, src, visited, i + 1, j - 1);
			fillWithBlack(result, src, visited, i - 1, j - 1);
			fillWithBlack(result, src, visited, i + 1, j + 1);
			fillWithBlack(result, src, visited, i - 1, j + 1);
		}
}


Mat fillGrey(Mat src)
{
	Mat_<uchar> result = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));
	Mat_<uchar> visited = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (visited.at<uchar>(i, j) == 0 && src.at<uchar>(i, j) == 255)
				fillWithWhite(result, src, visited, i, j);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
			if (visited.at<uchar>(i, j) == 0 && src.at<uchar>(i, j) == 128)
				fillWithBlack(result, src, visited, i, j);
	//imshow("visited", visited);
	return result;
}

void convolutionINT(Mat_<int> &filter, Mat_<uchar> &img, Mat_<int> &output)
{
	output.create(img.size());
	output.setTo(0);
	int kk = (filter.rows - 1) / 2;
	for (int i = kk; i < img.rows - kk; i++)
		for (int j = kk; j < img.cols - kk; j++)
		{
			float sum = 0;
			for (int k = 0; k < filter.rows; k++)
				for (int l = 0; l < filter.cols; l++)
					sum += img(i + k - kk, j + l - kk) * filter(k, l);

			output(i, j) = sum;
		}

}

Mat Sobelx(Mat src)
{
	Mat_<int> m = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	Mat_<int> ret;
	convolutionINT(m, (Mat_<uchar>)src, ret);
	return ret;
}

Mat Sobely(Mat src)
{
	Mat_<int> m = (Mat_<int>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	Mat_<int> ret;
	convolutionINT(m, (Mat_<uchar>)src, ret);
	return ret;
}


Mat getEdges(Mat src)
{
	Mat_<uchar> dest = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));
	Mat_<uchar> orientation = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));
	Mat_<uchar> magnitude = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));

	Mat_<int> x = Sobelx(src);
	Mat_<int> y = Sobely(src);


	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			uchar res = sqrt(x.at<int>(i, j) * x.at<int>(i, j) + y.at<int>(i, j) * y.at<int>(i, j)) / (4 * sqrt(2));
			magnitude.at<uchar>(i, j) = res;
			double at = atan2(y.at<int>(i, j), x.at<int>(i, j));
			if (at < 0)
				at = at + PI;
			if (at < PI / 8 || at >= 7 * PI / 8)
				orientation.at<uchar>(i, j) = 1;
			else if (at >= PI / 8 && at < 3 * PI / 8)
				orientation.at<uchar>(i, j) = 2;
			else if (at >= 3 * PI / 8 && at < 5 * PI / 8)
				orientation.at<uchar>(i, j) = 3;
			else if (at >= 5 * PI / 8 && at < 7 * PI / 8)
				orientation.at<uchar>(i, j) = 4;

		}
	imshow("Gradientul initial", magnitude);

	for (int r = 1; r < src.rows - 1; r++)
		for (int c = 1; c < src.cols - 1; c++)
		{
			dest[r][c] = magnitude[r][c];
			switch (orientation[r][c])
			{
			case 1:
				if (magnitude[r][c] < magnitude[r][c + 1] || magnitude[r][c] < magnitude[r][c - 1])
					dest[r][c] = 0;
				break;
			case 2:
				if (magnitude[r][c] < magnitude[r - 1][c + 1] || magnitude[r][c] < magnitude[r + 1][c - 1])
					dest[r][c] = 0;
				break;
			case 3:
				if (magnitude[r][c] < magnitude[r - 1][c] || magnitude[r][c] < magnitude[r + 1][c])
					dest[r][c] = 0;
				break;
			case 4:
				if (magnitude[r][c] < magnitude[r + 1][c + 1] || magnitude[r][c] < magnitude[r - 1][c - 1])
					dest[r][c] = 0;
				break;
			}
		}
	return dest;
}

Mat getEdgesAdaptiveThresholding(Mat src, int percent)
{
	Mat srcCpy = src.clone();
	Mat result = src.clone();
	src = Color2Gray(srcCpy);

	int Thigh = 0, Tlow;
	Mat filtered;
	GaussianBlur(src, filtered, Size(5, 5), 0.8, 0.8);

	imshow("Dupa aplicarea filtrului Gaussian", filtered);

	Mat_<uchar> magnitude = getEdges(filtered);
	imshow("Reducerea non-maximelor", magnitude);
	Mat_<uchar> threeColourMat = Mat(src.rows, src.cols, CV_8UC1, Scalar(0));
	int histo[256];
	for (int i = 0; i <= 255; i++)
		histo[i] = 0;
	for (int i = 0; i < magnitude.rows; i++)
		for (int j = 0; j < magnitude.cols; j++)
			histo[(int)magnitude.at<uchar>(i, j)]++;
	//showHistogram("histo test", histo, src.rows, src.cols);

	int nonEdgePixels = (1 - (percent/100))*((magnitude.rows - 2)*(magnitude.cols - 2) - histo[0]);

	int sum = 0;
	for (Thigh = 1; Thigh <= 255 && sum <= nonEdgePixels; Thigh++)
		sum += histo[Thigh];
	Tlow = 0.4*Thigh;
	
	for (int i = 0; i < magnitude.rows; i++)
		for (int j = 0; j < magnitude.cols; j++)
		{
			if (magnitude.at<uchar>(i, j) > Thigh)
				threeColourMat.at<uchar>(i, j) = 255;
			else if (magnitude.at<uchar>(i, j) > Tlow)
				threeColourMat.at<uchar>(i, j) = 128;
			else
				threeColourMat.at<uchar>(i, j) = 0;
		}
	imshow("Dupa binarizarea adaptiva", threeColourMat);
	Mat mat = fillGrey(threeColourMat);
	Vec3b color = (0, 255, 255);

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			if (threeColourMat.at<uchar>(i, j) != 0) {
				srcCpy.at<Vec3b>(i, j) = color;
			}
			
			if (mat.at<uchar>(i, j) != 0) {
				result.at<Vec3b>(i, j) = color;
			}
		}

	imshow("Binarizare adaptiva fara histerezis", srcCpy);

	return result;
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Canny algorithm\n");
		printf(" 2 - Default Canny\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				char fname[MAX_PATH];
				while (openFileDlg(fname))
				{
					Mat img = imread(fname, CV_LOAD_IMAGE_COLOR);
					imshow("image", img);
					int percent = 4;
					Mat edges = getEdgesAdaptiveThresholding(img, percent);
					imshow("poza finala", edges);
					waitKey(0);
				}
				break;
			case 2:
				testCanny();
				break;
		}
	}
	while (op!=0);
	return 0;
}