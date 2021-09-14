#include <stdio.h>

#define M 1/32768.0

void conv_pre(float mel_spec[][80], float conv_pre_out[][128], signed short conv_weight[128][80][7], signed short conv_bias[128], int frame_nums){
    int i;
    int j;
    int k;
    float mel_spec_padding[frame_nums+6][80];
     for(i = 0; i < 245; i++){
        for(j = 0; j < 80; j++){
            mel_spec_padding[i+3][j] = mel_spec[i][j];
        }
    }


    for(i = 0; i < 245; i++){
        for(j = 0; j < 128; j++){
            for(k = 0; k < 80; k++){
                conv_pre_out[i][j] += (mel_spec_padding[i][k] * conv_weight[j][k][0]
                                    + mel_spec_padding[i+1][k] * conv_weight[j][k][1]
                                    + mel_spec_padding[i+2][k] * conv_weight[j][k][2]
                                    + mel_spec_padding[i+3][k] * conv_weight[j][k][3]
                                    + mel_spec_padding[i+4][k] * conv_weight[j][k][4]
                                    + mel_spec_padding[i+5][k] * conv_weight[j][k][5]
                                    + mel_spec_padding[i+6][k] * conv_weight[j][k][6]) * M;
        }
        conv_pre_out[i][j] += conv_bias[j] * M;
        }
    }
}
