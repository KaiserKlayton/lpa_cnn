#include "convolution.h"
#include "lowp.h"

MatrixXd col2im(MatrixXd &c) {
    MatrixXd ct = c.transpose();

    return ct;
}

MatrixXd add_padding(const MatrixXd &box, const int im_height, const int im_width, const int p1, const int p2) {
    const int pa1 = im_width;
    const int pa2 = im_height+(2*p1);
    MatrixXd padding1 = MatrixXd::Zero(pa1,p1);
    MatrixXd padding2 = MatrixXd::Zero(p2,pa2);

    MatrixXd temp(box.rows(), box.cols()+padding1.cols()+padding1.cols());
    temp << padding1, box, padding1;

    MatrixXd padded_box(temp.rows()+padding2.rows()+padding2.rows(), temp.cols());
    padded_box << padding2, temp, padding2;

    return padded_box;
}

MatrixXd im2col(const MatrixXd &input, const int k_size, const int stride) {
    // input: A x B
    const int m = input.rows();
    const int n = input.cols();

    // kernel: C x D
    const int kRows = sqrt(k_size);
    const int kCols = sqrt(k_size);

    // output: xB*yB by C x D
    // yB = (A - C / stride) + 1
    // yB = (B - D / stride) + 1
    const int yB = ((m - kRows) / stride) + 1;
    const int xB = ((n - kCols) / stride) + 1;

    MatrixXd result(xB*yB, kRows*kCols);

    int indxc = 0;
    for(int i = 0; i < yB; i++)
    {
        int indxr = 0;
        for (int j = 0; j < xB; j++)
        {
            int rowIdx = i + j*yB;

            // Get block.
            for(unsigned int yy =0; yy < kRows; ++yy)
            {
                for(unsigned int xx=0; xx < kCols; ++xx)
                {
                    int colIdx = xx*kRows + yy;

                    result(rowIdx, colIdx) = input(indxc+yy, indxr+xx);
                }
            }
            indxr += stride;
        }
        indxc += stride;
    }
    return result;
}

std::tuple<MatrixXd, float, float> convolve(const MatrixXd &image, const int im_size, const int im_height, const int im_width, const int im_depth, const int k_size, const int stride, const VectorXd &b, const int p1, const int p2, const MatrixXd &w, const int output_size, std::string mode, const float w_min, const float w_max, const float result_min, const float result_max) {
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

    if (mode == "eigen") {
        // GEMM Multiplication Operation w/ Eigen.
        clock_t start = clock();
        MatrixXd c = im*w.transpose();
        clock_t end = clock();
        float gemm_time = (float) (end-start) / CLOCKS_PER_SEC;
        float offline_time = 0;
        std::cout << im_depth << "." << im.rows() << "." << im.cols() << "." << w.rows() << "." << w.cols() << std::endl;
        std::cout << gemm_time << "." << offline_time << std::endl;


        // Add biases.
        c.rowwise() += b.transpose();

        // Reshape back to image./
        MatrixXd convolved = col2im(c);

        return make_tuple(convolved, gemm_time, offline_time);
    }

    if (mode == "gemmlowp") {
        // GEMM Multiplication Operation w/ Gemmlowp.
        MatrixXd c;
        float gemm_time;
        float offline_time;
        std::tie(c, gemm_time, offline_time) = glp(im.rows(), im.cols(), w.transpose().cols(), im, w.transpose(), w_min, w_max, result_min, result_max);

        // Add biases.
        c.rowwise() += b.transpose();

        // Reshape back to image./
        MatrixXd convolved = col2im(c);

        return make_tuple(convolved, gemm_time, offline_time);
    }
}
