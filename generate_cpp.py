#!/usr/bin/env python

__author__ = "C. Clayton Violand"
__copyright__ = "Copyright 2017"

## Generates .cpp files and Makefiles from template for model inference execution.
## REQUIRED: Caffe .prototxt in: 'models/<model_name>/..'
##           Template as 'inference/helper/template.cpp'
##           

import os
import re
import sys
import glob

from shutil import copyfile

from helper.extract_architecture import extract_architecture

def main():
    # Get models.
    models = glob.glob('models/*[!.md]')
    models = [os.path.basename(m) for m in models]
    
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
        
        # Line Position
        pos = 17
        # Triggers
        first = True
        first_conv = True
        # Ticks
        tick = 1
        pool_tick = 1
        fc_tick = 1  
        counter = 1
        # Previous info
        last_output = ""
        last_type = ""
        second_last_output = ""
        second_last_type = ""
        
        to_write = []
        lines = []
        for i in a.keys()[1:]:
            i = str(i)  
            i_type = a[i]['type']
            current_index = a.keys().index(i)            
            if i_type == "convolution":
                if "branch1" in last_output:   
                    if not "pad" in a[i].keys():
                        a[i]['pad'] = 0
                    if not "stride" in a[i].keys():
                        a[i]['stride'] = 1                          
                    lines.extend([
                        "const int im_height_%d = im_height_%d;" % (counter, tick-1),
                        "const int im_width_%d = im_width_%d;" % (counter, tick-1),
                        "const int im_depth_%d = im_depth_%d;" % (counter, tick-1),
                        "const int im_size_%d = im_size_%d;" % (counter, tick-1),
                        ""
                    ])   

                    lines.extend([
                        "const int k_num_%d = %d;" % (counter, a[i]['num_output']),
                        "const int k_size_%d = %d;" % (counter, a[i]['kernel_size'] * a[i]['kernel_size']),
                        "const int stride_%d = %d;" % (counter, a[i]['stride']),
                        "const int k_depth_%d = im_depth_%d;" % (counter, tick),
                        "",
                        "const int p1_%d = %d;" % (counter, a[i]['pad']),
                        "const int p2_%d = %d;" % (counter, a[i]['pad']),
                        "",
                        "const int output_height_%d = (((im_height_%d+(2*p1_%d)) - sqrt(k_size_%d))/stride_%d) + 1;" % (counter, counter, counter, counter, counter),
                        "const int output_width_%d = (((im_width_%d+(2*p2_%d)) - sqrt(k_size_%d))/stride_%d) + 1;" % (counter, counter, counter, counter, counter),
                        "const int output_size_%d = output_height_%d * output_width_%d;" % (counter, counter, counter),
                        ""
                    ])
                    
                    lines.extend([
                        'MatrixXd %s_weights = load_csv_arma<MatrixXd>("../weights/%s/%s_weights.csv");' % (i, m, i),
                        "MatrixXd %s_w = %s_weights;" % (i, i),
                        "",
                        'MatrixXd %s_biases = load_csv_arma<MatrixXd>("../weights/%s/%s_biases.csv");' % (i, m, i),
                        "VectorXd %s_b(Map<VectorXd>(%s_biases.data(), %s_biases.cols()*%s_biases.rows()));" % (i, i, i, i),
                        ""
                    ]) 
                    
                else:                                   
                    if not "pad" in a[i].keys():
                        a[i]['pad'] = 0
                    if not "stride" in a[i].keys():
                        a[i]['stride'] = 1            
                    if first == True:
                        lines.extend([
                            "const int im_height_1 = %d;" % a['shape']['h'],
                            "const int im_width_1 = %d;" % a['shape']['w'],
                            "const int im_depth_1 = %d;" % a['shape']['d'],
                            "const int im_size_1 = im_height_1*im_width_1;",
                            ""
                        ])                        
                    elif second_last_type == "convolution" and last_type != "pooling" or second_last_type == "eltwise" and last_type != "pooling" or last_type == "convolution" or last_type == "eltwise":
                        lines.extend([
                            "const int im_height_%d = output_height_%d;" % (counter, tick-1),
                            "const int im_width_%d = output_width_%d;" % (counter, tick-1),
                            "const int im_depth_%d = k_num_%d;" % (counter, tick-1),
                            "const int im_size_%d = im_height_%d * im_width_%d;" % (counter, counter, counter),
                            ""
                        ])
                    else:  
                        lines.extend([                     
                            "const int im_height_%d = ((output_height_%d - f_%d + 2 * pp1_%d) / s_%d) + 1;" % (counter, tick-1, pool_tick-1, pool_tick-1, pool_tick-1),
                            "const int im_width_%d = ((output_width_%d - f_%d + 2 * pp2_%d) / s_%d) + 1;" % (counter, tick-1, pool_tick-1, pool_tick-1, pool_tick-1),
                            "const int im_depth_%d = k_num_%d;" % (counter, tick-1),
                            "const int im_size_%d = im_height_%d * im_width_%d;" % (counter, counter, counter),
                            ""
                        ])      

                    lines.extend([
                        "const int k_num_%d = %d;" % (counter, a[i]['num_output']),
                        "const int k_size_%d = %d;" % (counter, a[i]['kernel_size'] * a[i]['kernel_size']),
                        "const int stride_%d = %d;" % (counter, a[i]['stride']),
                        "const int k_depth_%d = im_depth_%d;" % (counter, tick),
                        "",
                        "const int p1_%d = %d;" % (counter, a[i]['pad']),
                        "const int p2_%d = %d;" % (counter, a[i]['pad']),
                        "",
                        "const int output_height_%d = (((im_height_%d+(2*p1_%d)) - sqrt(k_size_%d))/stride_%d) + 1;" % (counter, counter, counter, counter, counter),
                        "const int output_width_%d = (((im_width_%d+(2*p2_%d)) - sqrt(k_size_%d))/stride_%d) + 1;" % (counter, counter, counter, counter, counter),
                        "const int output_size_%d = output_height_%d * output_width_%d;" % (counter, counter, counter),
                        ""
                    ])
                    
                    if first_conv == True:
                        lines.extend([
                            'MatrixXd %s_weights = load_csv_arma<MatrixXd>("../weights/%s/%s_weights.csv");' % (i, m, i),
                            "Map<MatrixXd> %s_w(%s_weights.data(), k_num_%d, k_size_%d * k_depth_%d);" % (i, i, tick, tick, tick),
                            "",
                            'MatrixXd %s_biases = load_csv_arma<MatrixXd>("../weights/%s/%s_biases.csv");' % (i, m, i),
                            "VectorXd %s_b(Map<VectorXd>(%s_biases.data(), %s_biases.cols()*%s_biases.rows()));" % (i, i, i, i),
                            ""
                        ])
                    else:
                        lines.extend([
                            'MatrixXd %s_weights = load_csv_arma<MatrixXd>("../weights/%s/%s_weights.csv");' % (i, m, i),
                            "MatrixXd %s_w = %s_weights;" % (i, i),
                            "",
                            'MatrixXd %s_biases = load_csv_arma<MatrixXd>("../weights/%s/%s_biases.csv");' % (i, m, i),
                            "VectorXd %s_b(Map<VectorXd>(%s_biases.data(), %s_biases.cols()*%s_biases.rows()));" % (i, i, i, i),
                            ""
                        ])
                    
                first_conv = False                                    
                counter += 1                
                tick += 1
                
            elif i_type == "pooling":
                if not "pad" in a[i].keys():
                    a[i]['pad'] = 0
                    
                if not "stride" in a[i].keys():
                    a[i]['stride'] = 1
                              
                lines.extend([
                    "const int f_%i = %d;" % (pool_tick, a[i]['kernel_size']),
                    "const int s_%i = %d;" % (pool_tick, a[i]['stride']),
                    ""
                ])
                
                if a[i]['stride'] == 3:
                    lines.extend([
                        "const int pp1_%i = 1;" % pool_tick,
                        "const int pp2_%i = 1;" % pool_tick,
                        "" 
                    ]) 
                else:
                    lines.extend([
                        "const int pp1_%i = %d;" % (pool_tick, a[i]['pad']),
                        "const int pp2_%i = %d;" % (pool_tick, a[i]['pad']),
                        "" 
                    ]) 
                    
                pool_tick += 1
                
            elif i_type == "innerproduct":
                lines.extend([
                    'MatrixXd %s_weights = load_csv_arma<MatrixXd>("../weights/%s/%s_weights.csv");' % (i, m, i),
                    "",
                    'MatrixXd %s_biases = load_csv_arma<MatrixXd>("../weights/%s/%s_biases.csv");' % (i, m, i),
                    "VectorXd %s_b(Map<VectorXd>(%s_biases.data(), %s_biases.cols()*%s_biases.rows()));" % (i, i, i, i),
                    ""
                ])
                
                fc_tick += 1
                                                        
            first = False
            second_last_output = last_output
            last_output = i
            second_last_type = last_type
            last_type = a[i]['type']
                       
        # Get input data path.
        input_files = os.listdir("inputs/" + m + "/production/")
        if (len(input_files) > 1):
            for i in range(len(input_files)):
                if input_files[i].endswith('.csv'):
                    input_file_path = "../inputs/" + m + "/production/" + input_files[i]
                else:
                    continue
        else:
            input_file_path = "../inputs/" + m + "/production/" + input_files[0]
            
        lines.extend([
            #"const int im_num = %i;" % a['shape']['n'],
            "const int im_num = 1000;",
            "",
            'ifstream infile;',
            "infile.open(\"%s\");" % input_file_path,
            "",
        ])
        
        for l in lines:
            to_write.append((pos, "\t"+l+"\n"))
            pos += 1           
        
        pos += 4
        lines_input = []
        lines_input.extend([
            "MatrixXd line = load_csv<MatrixXd>(infile);",
            ""
        ])   
        
        for l in lines_input:
            to_write.append((pos, "\t\t"+l+"\n"))
            pos += 1   
                                      
        pos = pos + 12
        tick = 1
        pool_tick = 1
        relu_tick = 1
        fc_tick = 1
        gemm_tick = 1
        eltwise_tick = 1
        gemm_string = ""
        offline_string = ""
        last_output = ""
        last_type = ""
        second_last_output = ""
        second_last_type = ""
        lines_two = []
        for i in a.keys():
            current_index = a.keys().index(i)
            if i == "shape":
                a['shape']['type'] = "input"
            if last_output == "shape":
                last_output = "image"  
            i = str(i)
            i_type = a[i]['type']                                
            if i_type == "convolution":    
                if "branch1" in last_output:
                    lines_two.extend([
                        "MatrixXd %s;" % i,
                        "double gemm_time_%i;" % tick,
                        "double offline_time_%i;" % tick,
                        "std::tie(%s, gemm_time_%i, offline_time_%i) = convolve(%s, im_size_%i, im_height_%i, im_width_%i, im_depth_%i, k_size_%i, stride_%i, %s_b, p1_%i, p2_%i, %s_w, output_size_%i);" % (i, tick, tick, second_last_output, tick, tick, tick, tick, tick, tick, i, tick, tick, i, tick),
                        ""
                    ])
                    
                else:                
                    lines_two.extend([
                        "MatrixXd %s;" % i,
                        "double gemm_time_%i;" % tick,
                        "double offline_time_%i;" % tick,
                        "std::tie(%s, gemm_time_%i, offline_time_%i) = convolve(%s, im_size_%i, im_height_%i, im_width_%i, im_depth_%i, k_size_%i, stride_%i, %s_b, p1_%i, p2_%i, %s_w, output_size_%i);" % (i, tick, tick, last_output, tick, tick, tick, tick, tick, tick, i, tick, tick, i, tick),
                        ""
                    ])
                 
                offline_string += " - offline_time_%s" % gemm_tick
                gemm_string += " + gemm_time_%s" % gemm_tick
                gemm_tick += 1
                
                tick += 1
                
                if "branch1" in i:
                    eltwise_input = i
                
            elif i_type == "pooling":
                lines_two.extend([
                    "MatrixXd %s = pool(%s, f_%i, s_%i, output_width_%i, output_height_%i, pp1_%i, pp2_%i);" % (i, last_output, pool_tick, pool_tick, tick-1, tick-1, pool_tick, pool_tick),
                    ""
                ])
                    
                pool_tick += 1
                
            elif i_type == "relu":
                lines_two.extend([
                    "MatrixXd %s = relu(%s);" % (i, last_output),
                    ""
                ])
                
                relu_tick += 1
                
                if last_type == "eltwise":
                    eltwise_input = i
                
            elif i_type == "innerproduct":
                lines_two.extend([
                    "MatrixXd %s = fully_connect(%s, %s.rows(), %s_weights, %s_b);" % (i, last_output, last_output, i, i),
                    ""
                ])
                
                fc_tick += 1
            
            elif i_type == "eltwise":
                lines_two.extend([
                    "MatrixXd %s = eltwise(%s, %s);" % (i, last_output, eltwise_input),
                    ""
                ])
                
                eltwise_tick += 1  
                
            second_last_output = last_output
            last_output = i
            second_last_type = last_type
            last_type = a[i]['type']
                
        for l in lines_two:
            to_write.append((pos, "\t\t"+l+"\n"))
            pos += 1
            
        lines_three = []    
        lines_three.extend([
            "run_time_total += (run_time%s);" % offline_string,
            "gemm_time_total += 0.0%s;" % gemm_string,
            ""
        ])      
        
        name_tick = 1
        last_item = a.keys()[-1]
        for j in a.keys()[1:]:
            j_type = a[j]['type']
            
            if j == last_item:          
                lines_three.extend([
                'std::string name_%i = "../features/%s/%s_" + std::to_string(i) + ".csv";' % (name_tick, m, j),
                'write_to_csv(name_%i, %s);' % (name_tick, j),
             ])
                name_tick += 1
  
        pos = pos + 3
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
        
        # Makefile.
        makefile_template = open("inference/helper/makefile_template", "r+").readlines()
        makefile_template.insert(3, "SOURCES= main_%s.cpp helper/writer.cpp ../layers/convolution_layer/convolution.cpp ../layers/convolution_layer/lowp.cpp ../layers/pooling_layer/pooling.cpp ../layers/fully_connected_layer/fully_connected.cpp ../layers/relu_layer/relu.cpp ../layers/eltwise_layer/eltwise.cpp" % m)
        
        makefile_path = os.path.join("inference", "Makefile." + m)
        makefile = open(makefile_path, "w")
        [makefile.write(l) for l in makefile_template]
        makefile.close()
        
if __name__ == "__main__":
    main()
