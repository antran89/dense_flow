#ifndef LDOF_H_
#define LDOF_H_
#include <CTensor.h>

void ldof(CTensor<float> aImage1, CTensor<float> aImage2, CTensor<float>& aForward, CTensor<float>& aBackward);

void ldof(CTensor<float> aImage1, CTensor<float> aImage2, CTensor<float>& aForward, CTensor<float>& aBackward, float sigma, float alpha, float beta, float gamma, float eta, int fp_iterations, int linsolve_iterations);

void runFlowGPU(CTensor<float>& aImage1, CTensor<float>& aImage2, CTensor<float>& aResult, float sigma, float alpha, float beta, float gamma, float eta, int fp_iterations, int linsolve_iterations) ;

//float aae(CTensor<float> flow, CTensor<float> refflow);

/*
 * function for computing large displacement optical flow using CPU
 *
 * Arguments "aImage1" and "aImage2" are color images (3-dimensional arrays).
 * They are required to have the same dimension.
 * The computed optical flow is stored in "aResult".
 * The array will be resized to fit the image dimension.
 *
 * The other arguments are tuning parameters
 * (the number in parentheses is the default value)
 *
 *       sigma:        (0.8) presmoothing of the input images
 *       alpha:        (30)  smoothness of the flow field
 *       beta:         (300) influence of the descriptor matching
 *       gamma:        (5)   influence of the gradient constancy assumption
 *
 * The default values work best with image values in the range [0,255].
 *
 * Thomas Brox
 * U.C. Berkeley
 * Apr, 2010
 * All rights reserved
 */
void runFlow(CTensor<float>& aImage1, CTensor<float>& aImage2, CTensor<float>& aResult, float sigma=0.8f, float alpha=30.f, float beta=300.f, float gamma=5.f);

#endif /* LDOF_H_ */
