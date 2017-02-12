#include "convolution.h"
#include "lowp.h"

MatrixXd col2im(MatrixXd c) {
    MatrixXd ct = c.transpose();

    return ct;
}

MatrixXd add_padding(MatrixXd box, int im_height, int im_width, int p1, int p2) {
    int pa1 = im_width;
    int pa2 = im_height+(2*p1);
    MatrixXd padding1 = MatrixXd::Zero(pa1,p1);
    MatrixXd padding2 = MatrixXd::Zero(p2,pa2); 

    MatrixXd temp(box.rows(), box.cols()+padding1.cols()+padding1.cols());
    temp << padding1, box, padding1;
    
    MatrixXd padded_box(temp.rows()+padding2.rows()+padding2.rows(), temp.cols());  
    padded_box << padding2, temp, padding2;

    return padded_box;
}

MatrixXd kernel2col(MatrixXd kernel) {
    // kernel: n x m
    int m = kernel.rows();
    int n = kernel.cols();

    kernel.transposeInPlace();

    MatrixXd kcollapsed(Map<VectorXd>(kernel.data(), kernel.cols()*kernel.rows()));

    return kcollapsed;
}

MatrixXd im2col(MatrixXd input, int k_size, int stride) {
    // input: A x B
    int m = input.rows();
    int n = input.cols();

    // kernel: C x D
    int kRows = sqrt(k_size);
    int kCols = sqrt(k_size);

    // output: xB*yB by C x D
    // yB = (A - C / stride) + 1
    // yB = (B - D / stride) + 1
    int yB = ((m - kRows) / stride) + 1;
    int xB = ((n - kCols) / stride) + 1;
    
    MatrixXd result(xB*yB, kRows*kCols);
    
    for(int i = 0; i < yB; i++)
    {
        for (int j = 0; j < xB; j++)
        {
            int rowIdx = i + j*yB;

            for(unsigned int yy =0; yy < kRows; ++yy)
                for(unsigned int xx=0; xx < kCols; ++xx)
                {
                    int colIdx = xx*kRows + yy; 

                    result(rowIdx, colIdx) = input(i+yy, j+xx);
                }
        }
    }
    return result;
}

std::pair<MatrixXd, double> convolve(MatrixXd image, int im_size, int im_height, int im_width, int im_depth, int k_size, int stride, VectorXd b, int p1, int p2, MatrixXd w, int output_size) {  
    // im2col for each slice, then concatinate slices.
    MatrixXd im(output_size, k_size*im_depth);
    for (int d=0; d < im_depth ; d++) {
        // Take slice out of im_depth.
        MatrixXd slice = image.block(d,0,1,im_size);
        // Resize slice to be square.
        Map<MatrixXd> box(slice.data(), im_height, im_width);
        // Pad box with 0s.
        MatrixXd padded_box = add_padding(box, im_height, im_width, p1, p2);       
        // im2col on particular slice.
        MatrixXd col_slice = im2col(padded_box, k_size, stride);
        // Concatinate col_slice to output 'im'.
        im.block(0,k_size*d, output_size, k_size) = col_slice; 
    } 

    // GEMM Multiplication Operation w/ Eigen.
//    clock_t start = clock();    
//    MatrixXd c = im*w.transpose();
//    clock_t end = clock();
//    double time = (double) (end-start) / CLOCKS_PER_SEC;           
  
    // GEMM Multiplication Operation w/ Gemmlowp.
    MatrixXd c;
    double time;
    std::tie(c, time) = glp(im.rows(), im.cols(), w.transpose().cols(), im, w.transpose());

    // Add biases.
    c.rowwise() += b.transpose();

    // Reshape back to image./    
    MatrixXd convolved = col2im(c);

    return make_pair(convolved, time);
}
