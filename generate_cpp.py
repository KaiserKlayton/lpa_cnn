#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

## Generates .cpp files from template for model inference execution.
## REQUIRED: Caffe .prototxt in: 'models/<model_name>/..'
##           Template as 'inference/template.cpp'
##           

import os
import re
import sys

from shutil import copyfile

from helper.extract_architecture import extract_architecture

def main():
    # Get models.
    models = os.listdir("models/")
    
    # For every model.
    for m in models:
        
        # Create file from template.
        infile = open("inference/helper/template.cpp", "r+").readlines()
        
        # Get model architecture.
        prototxt_dir = os.path.join("models", m)
        for f in os.listdir(prototxt_dir):
            if f.endswith('.prototxt'):
                prototxt_file_path = os.path.join(prototxt_dir, f)
                
        a = extract_architecture(prototxt_file_path)
        
        pos = 9
        tick = 1
        pool_tick = 1
        fc_tick = 1
        first = True
        to_write = []
        lines = []
        for i in a.keys()[1:]:
            i = str(i)    
            if "conv" in i:
                if first == True:
                    lines.extend([
                        "const int im_height_1 = %d;" % a['shape']['h'],
                        "const int im_width_1 = %d;" % a['shape']['w'],
                        "const int im_depth_1 = %d;" % a['shape']['d'],
                        "const int im_size_1 = im_height_1*im_width_1;",
                        ""
                    ])    
                else:
                    lines.extend([
                        "const int im_height_%d = ((output_height_%d - f_%d) / s_%d) + 1;" % (tick, tick-1, tick-1, tick-1),
                        "const int im_width_%d = ((output_width_%d - f_%d) / s_%d) + 1;" % (tick, tick-1, tick-1, tick-1),
                        "const int im_depth_%d = k_num_%d;" % (tick, tick-1),
                        "const int im_size_%d = im_height_%d * im_width_%d;" % (tick, tick, tick),
                        ""
                    ])
                                              
                lines.extend([
                    "const int k_num_%d = %d;" % (tick, a[i]['num_output']),
                    "const int k_size_%d = %d;" % (tick, a[i]['kernel_size']),
                    "const int stride_%d = %d;" % (tick, a[i]['stride']),
                    "const int k_depth_%d = im_depth_%d;" % (tick, tick),
                    "",
                    "const int p1_%d = %d;" % (tick, a[i]['pad']),
                    "const int p2_%d = %d;" % (tick, a[i]['pad']),
                    "",
                    "const int output_height_%d = (((im_height_%d+(2*p1_%d)) - sqrt(k_size_%d))/stride_%d) + 1;" % (tick, tick, tick, tick, tick),
                    "const int output_width_%d = (((im_width_%d+(2*p2_%d)) - sqrt(k_size_%d))/stride_%d) + 1;" % (tick, tick, tick, tick, tick),
                    "const int output_size_%d = output_height_%d * output_width_%d;" % (tick, tick, tick),
                    "",
                    "MatrixXd %s_weights = load_csv_arma('weights/%s/%s_weights.csv');" % (i, m, i),
                    "Map<MatrixXd> %s_w(%s_weights.data(), k_num_%d, k_size_%d * k_depth_%d);" % (i, i, tick, tick, tick),
                    "",
                    "MatrixXd %s_biases = load_csv_arma('weights/%s/%s_biases.csv');" % (i, m, i),
                    "VectorXd %s_b(Map<VectorXd>(%s_biases.data(), %s_biases.cols()*%s_biases.rows()));" % (i, i, i, i),
                    ""
                ])
                
                tick += 1
                
            elif "pool" in i:
                lines.extend([
                    "const int f_%i = %d;" % (pool_tick, a[i]['kernel_size']),
                    "const int s_%i = %d;" % (pool_tick, a[i]['stride']),
                    ""
                ])
                
                pool_tick += 1
                
            elif "fc" in i or "ip" in i:
                lines.extend([
                    "MatrixXd %s_weights = load_csv_arma('weights/%s/%s_weights.csv');" % (i, m, i),
                    "MatrixXd %s_biases = load_csv_arma('weights/%s/%s_biases.csv');" % (i, m, i),
                    "VectorXd %s_b(Map<VectorXd>(%s_biases.data(), %s_biases.cols()*%s_biases.rows()));" % (i, i, i, i),
                    ""
                ])
                
                fc_tick += 1
        
            first = False
                                 
        # Get input data path.
        input_files = os.listdir("inputs/" + m + "/production/")
        if (len(input_files) > 1):
            for i in range(len(input_files)):
                if input_files[i].endswith('.csv'):
                    input_file_path = "inputs/" + m + "/production/" + input_files[i]
                else:
                    continue
        else:
            input_file_path = "inputs/" + model + "/production/" + input_files[0]
            
        lines.extend([
            "const int im_num = %i;" % a['shape']['n'],
            "MatrixXd train = load_csv_arma(%s);" % input_file_path,
            ""
        ])
        
        for l in lines:
            to_write.append((pos, "\t"+l+"\n"))
            pos += 1           
                                           
        pos = pos + 7
        tick = 1
        pool_tick = 1
        relu_tick = 1
        fc_tick = 1
        lines_two = []
        for i in a.keys():
            current_index = a.keys().index(i)
            last_output = a.keys()[current_index-1]
            if last_output == "shape":
                last_output = "image"  
            i = str(i)
            if "shape" in i:
                lines_two.extend([
                    "MatrixXd img = train.block<1,%i>(i,0);" % (a[i]['h'] * a[i]['w'] * a[i]['d']),
                    "MatrixXd image = Map<Matrix<double, %i, %i, RowMajor>>(img.data());" % (a[i]['d'], a[i]['h'] * a[i]['w']),
                    ""
                ])
            elif "conv" in i:                    
                lines_two.extend([
                    "MatrixXd %s;" % i,
                    "double gemm_time_%i;" % tick,
                    "double offline_time_%i;" % tick,
                    "std::tie(convolved_%i, gemm_time_%i, offline_time_%i) = convolve(%s, im_size_%i, im_height_%i, im_width_%i, im_depth_%i, k_size_%i, stride_%i, %s_b, p1_%i, p2_%i, %s_w, output_size_%i);" % (tick, tick, tick, last_output, tick, tick, tick, tick, tick, tick, i, tick, tick, i, tick),
                    ""
                ])
                
                tick += 1
            elif "pool" in i:
                lines_two.extend([
                    "MatrixXd %s = pool(%s, f_%i, s_%i, output_width_%i, output_height_%i);" % (i, last_output, tick, tick, tick, tick),
                    ""
                ])
                    
                pool_tick += 1
            elif "relu" in i:
                lines_two.extend([
                    "MatrixXd %s = relu(%s);" % (i, last_output),
                    ""
                ])
                
                relu_tick += 1
            elif "fc" in i or "ip" in i:
                lines_two.extend([
                    "MatrixXd %s = fully_connect(%s, %s.rows(), %s_weights, %s_b);" % (i, last_output, last_output, i, i),
                    ""
                ])
                
                fc_tick += 1

        for l in lines_two:
            to_write.append((pos, "\t\t"+l+"\n"))
            pos += 1
            
        # TODO: Make dynamic.
        lines_three = []    
        lines_three.extend([
            "run_time_total += (run_time - offline_time_1 - offline_time_2 - offline_time_3);",
            "gemm_time_total += gemm_time_1 + gemm_time_2 + gemm_time_3;",
            ""
        ])      
        
        for j in a.keys()[1:]:
            if "conv" in j or "pool" in j or "fc" in j or "ip" in j:
                lines_three.extend([
                'std::string name = "features/%s/conv1_" + std::to_string(i) + ".csv";' % m,
                'write_to_csv(name, %s);' % j,
             ])
  
        for l in lines_three:
            to_write.append((pos, "\t\t"+l+"\n"))
            pos += 1

        # Add the content.
        for (index, string) in to_write:
            infile.insert(index, string)

        # Write to file.
        outfile_path = os.path.join("inference", "main_"+ m +".cpp")
        outfile = open(outfile_path, "w")
        [outfile.write(l) for l in infile]
        outfile.close()
        sys.exit()
        
if __name__ == "__main__":
    main()
