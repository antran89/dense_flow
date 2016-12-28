#ifndef CTENSOROPENCV_H_
#define CTENSOROPENCV_H_

#include <opencv2/core.hpp>

#include "CTensor.h"

template < typename T >
class CTensorOpencv : public CTensor< T >
{
public:
    // standard constructor
    inline CTensorOpencv() : CTensor< T >() {}
    // constructor
    inline CTensorOpencv( const int xSize, const int ySize, const int zSize ): CTensor< T >( xSize, ySize, zSize ) {}
    // copy constructor
    CTensorOpencv( const CTensor<T> &aCopyFrom ): CTensor< T >( aCopyFrom ) {}
    // constructor with implicit filling
    CTensorOpencv( const int xSize, const int ySize, const int zSize, const T value ): CTensor< T >( xSize, ySize, zSize, value ) {}

    /** @brief Copy Mat image to CTensor image.

     The function transforms a Mat image to a CTensor image format for computing LDOF.

     @param input An input image.
     */
    void copyFromMat( const cv::Mat input )
    {
        int xSize, ySize, zSize;

        int nChannels = input.channels() ;
        int nRows = input.rows;
        int nCols = input.cols * nChannels;

        xSize = input.cols;
        ySize = nRows;
        zSize = nChannels;

        if( nChannels != 1 && nChannels != 3 )
        {
            //throw MxMatrixSizeIncompatible( dimensions );
            std::cout << "crap" << std::endl;
            return;
        }

        if( xSize != CTensor< T >::mXSize || ySize != CTensor< T >::mYSize || zSize != CTensor< T >::mZSize )
        {
            delete [] CTensor< T >::mData;

            CTensor< T >::mXSize = xSize;
            CTensor< T >::mYSize = ySize;
            CTensor< T >::mZSize = zSize;
            CTensor< T >::mData = new T[ CTensor< T >::mXSize * CTensor< T >::mYSize * CTensor< T >::mZSize ];
        }

        // We assume the Mat contains CV_8U data
        if ( input.isContinuous() )
        {
            nCols *= nRows;
            nRows = 1;
        }
        int ind = 0;
        for ( int ch = 0; ch < nChannels; ch++ )
            for (int i = 0; i < nRows; i++ )
            {
                const uchar* p = input.ptr<uchar>( i );
                for ( int j = 0; j < nCols; j += nChannels )
                    CTensor< T >::mData[ ind++ ] = ( T )p[j + ch];
            }

        CV_Assert( ind == xSize * ySize * zSize );

    }

    /** @brief Copy CTensor image to Mat image.

    The function transforms a CTensor image format to an Mat image for further processing in opencv.

    @param output Output image.
    */
    void copyToMat( cv::Mat& output )
    {
        int nCols = CTensor< T >::mXSize;
        int nRows = CTensor< T >::mYSize;
        int nFlows = CTensor< T >::mZSize;
        CV_Assert( nFlows == 2 );

        cv::Mat flows[2];

        int ind = 0;
        for ( int k = 0; k < nFlows; k++ ) {
            flows[k].create( nRows, nCols, CV_32FC1 );
            for ( int i = 0; i < nRows; i++ )
            {
                // the solution right now, because no trivial way to get CV_32F from float
                float* p = flows[k].ptr< float >( i );
                for ( int j = 0; j < nCols; j++ )
                    p[j] = CTensor< T >::mData[ ind++ ];
            }
        }

        CV_Assert( ind == nRows * nCols * nFlows );

        // merge two flow into a Mat
        cv::merge(flows, nFlows, output);

    }

};

#endif
