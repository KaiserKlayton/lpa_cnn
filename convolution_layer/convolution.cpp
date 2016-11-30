#include "convolution.h"

MatrixXd col2im(MatrixXd c) {
    MatrixXd ct = c.transpose();
//    cout << ct.rows() << endl;s
//    cout << ct.cols() << endl;

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
//    cout << padded_box.cols() << endl;
//    cout << padded_box.rows() << endl;

    return padded_box;
}

MatrixXd kernel2col(MatrixXd kernel) {
    // kernel: n x m
    int m = kernel.rows();
    int n = kernel.cols();

    kernel.transposeInPlace();

    MatrixXd kcollapsed(Map<VectorXd>(kernel.data(), kernel.cols()*kernel.rows()));
//    cout << kcollapsed << endl;

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
 //   cout << kRows << endl;
 //   cout << kCols << endl; 
    
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
//    cout << result << endl;
    return result;
}
