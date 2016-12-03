#include "pooling.h"

MatrixXd pool(MatrixXd convolved, int f, int s, int im_width, int im_height) {
    const int w = ((im_width - f) / s) + 1;
	const int h = ((im_height - f) / s) + 1;
    MatrixXd pooled(convolved.rows(), w*h);
    for (int i=0; i < convolved.rows(); i++) {
        // Take filter from stack of filters.
        MatrixXd slice = convolved.row(i);
        // Shape into image.
        Map<MatrixXd> box(slice.data(), im_width, im_height);
        // for each block from image:
        MatrixXd pooling(w, h);
        for (int m=0; m < w; m++) {
            for (int n=0; n < h; n++) {
                MatrixXd smallblock = box.block(0+(m*s),0+(n*s),f,f); 
                float max = smallblock.maxCoeff();       
                pooling(m,n) = max;                                  
            }
        }     
        // Flatten pooling:
        Map<VectorXd> pcollapsed(pooling.data(),pooling.size()); 
        // Add back to rows of filters.  
        pooled.row(i) = pcollapsed;     	    	
    }
    return pooled;
}
