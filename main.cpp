#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "eigen3/Eigen/Core"
#include "eigen3/Eigen/SVD"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	
	cv::Mat U;
	cv::Mat	S;
	cv::Mat V;
	cv::Mat A;

	cv::Mat R, G, B;
	
	

	// Show individual channels
    cv::Mat g, fin_img;
    

	// "channels" is a vector of 3 Mat arrays:
	std::vector<cv::Mat> channels(3);

	/// Load source image and convert it to gray
  	A = cv::imread( argv[1],  cv::IMREAD_COLOR );
  	std::cout << "A size: " << A.rows << " x " << A.cols << std::endl;
  	// cv::Mat B = cv::Mat(A.rows, A.cols, CV_32F);
  	
  	cv::Mat UB, SB, VB;
  	g = cv::Mat::zeros(cv::Size(A.cols, A.rows), CV_8UC1);
	cv::split(A, channels);
	// get the channels (dont forget they follow BGR order in OpenCV)
	B = channels[0];
	G = channels[1];
	R = channels[2];
	B.convertTo(B,CV_32FC1,1.0/255.0);
	G.convertTo(G,CV_32FC1,1.0/255.0);
	R.convertTo(R,CV_32FC1,1.0/255.0);

	// float dataH[3][3] = {{ 25079040., -2.22029547e+005, -8.73665188e+005 }, 
                     // { 2811925.,  -2441455.,        -68310968. },
                     // { 0.,        0.,               0. }};
	// cv::Mat H( 3, 3, CV_32F, dataH );
	//cv::SVD svd_obj;
	//svd_obj.compute(, S, U, V);

	std::cout << "[start] OpenCV SVD decomposition" << std::endl;
	cv::SVD::compute(B, SB, UB, VB, cv::SVD::FULL_UV);
	std::cout << "[end] OpenCV SVD decomposition" << std::endl;
	// std::cout << "U" << std::endl << SB << std::endl << std::endl;
	std::cout << "Singular values size (S): " << SB.rows << " x " << SB.cols << std::endl;
	/// Keep the largest singular values, and nullify others
	cv::Mat sigma_mB = cv::Mat::zeros(SB.rows,SB.rows,CV_32FC1);       
    for(int i=0; i<SB.rows; i++){
    	sigma_mB.at<float>(i,i) = SB.at<float>(i);
    }
    std::cout << "Singular values size (S): " << sigma_mB.rows << " x " << sigma_mB.rows << std::endl;
    // Reduce Rank to k
    int k = 3;
    ///Form the singular values matrix, paddes as necessary
    sigma_mB = sigma_mB(cv::Range(0,k),cv::Range(0,k));
    UB = UB(cv::Range::all(),cv::Range(0,k));
    VB = VB(cv::Range(0,k),cv::Range::all());
    // Compute low-rank approximation by multiplying out component matrices 
    cv::Mat Result = UB*sigma_mB*VB;
    // std::cout << "Result size:" << UB.rows*UB.cols+k+VB.rows*VB.cols <<" elements "<< std::endl;
    std::cout << "Result size:" << Result.rows << "x" << Result.cols <<" elements "<< std::endl;

	{
	    std::vector<cv::Mat> multi_channel_image;
	    multi_channel_image.push_back(Result);
	    multi_channel_image.push_back(g);
	    multi_channel_image.push_back(g);
	    /// Merge the three channels
	    cv::merge(multi_channel_image, fin_img);	    
	}
	
	std::cout << "Processing done." << std::endl;
	namedWindow( "Original Image", cv::WINDOW_NORMAL );
	cv::imshow("Original Image", Result);

	cv::waitKey(0);//Wait for a keystroke in the window
	return 0;
}
