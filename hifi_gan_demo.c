#include <stdio.h>
#include <time.h>
#include "kiss_fft/fft_compute.h"
#include "mel_basis.h"
#include "hifi_gan_weights.h"
#include "hifi_gan.h"

#define M 32768.0


int main(){

    int i, j, k;
    clock_t start, finish;
    double  duration;
    FILE *wavfile;
    long size;
    short win_size = 1024;
    short hop_size = 256;
    short frame_nums;
    short frame;
    short win_data[win_size];
    short hop_data[hop_size];
    float mel_basis[513][80];
    float hann_window[1024];
    float PI = 3.141592653589793;
    short data_in[1024];

    for(i = 0; i < 1024; i++){
        hann_window[i] = 0.5 * (1 - cos(2 * PI * i / 1024));
    }


    mel_filter_bank(mel_basis);

    /* confirm right mel_basis
      for(i = 0; i<80; i++){
        printf("%f,", mel_basis[2][i]);
    }*/

    kiss_fft_cfg cfg = kiss_fft_alloc(win_size, 0, 0 ,0 );


    kiss_fft_cpx buf_in[win_size];
    kiss_fft_cpx buf_out[win_size];


    start = clock();

    wavfile = fopen("p234_001_22k.wav", "rb");

    fseek(wavfile, 0, SEEK_END);

    /*data size*/
    size = (ftell(wavfile) - 78)/2;

    /*skip wav header*/
    fseek(wavfile, 78, SEEK_SET);
    frame_nums = (size - win_size)/hop_size + 1;
    //printf("%d, %d, ", frame_nums, size);

    float pow_spec[frame_nums][513];
    float mel_spec[frame_nums][80];

    int first = 1;

    int frame_id = 0;

    while(1){
        if (first){
            fread(win_data, sizeof(short), win_size, wavfile);
            for(i = 0; i < 1024; i++){
                //printf("%d, ", win_data[i]);
            }

            for(i = 0; i < win_size; i++){
                data_in[i] = win_data[i];
            }

        }
        else{
            for(i = 0; i < 768; i++){
                data_in[i] = data_in[i+256];
            }
            fread(hop_data, sizeof(short), hop_size, wavfile);
            if (feof(wavfile)) break;

            for(i = 0; i < 256; i++){
                data_in[i+768] = hop_data[i];
            }
        }

        for(i = 0; i < win_size; i++){
                buf_in[i].r = hann_window[i] * data_in[i] / 32768.0;
                buf_in[i].i = 0;
            }

        kiss_fft(cfg , buf_in , buf_out);

        for(i = 0; i < 513; i++){
            //printf("%f, ", buf_out[i].r);
        }


        for(i = 0; i < 513; i++){
            pow_spec[frame_id][i] = sqrt(pow(buf_out[i].r, 2) + pow(buf_out[i].i, 2) + 1e-9);
        }


        first = 0;
        frame_id += 1;
    }
                                                    
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 80; j++){
            for( k = 0; k < 513;k++){
                mel_spec[i][j] += pow_spec[i][k] * mel_basis[k][j];
            }
        }
    }

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 80; j++){
            if(mel_spec[i][j] < 1e-5){
                mel_spec[i][j] = 1e-5;
            }
            mel_spec[i][j] = log(mel_spec[i][j]);
        }
    }


    float conv_pre_out[frame_nums][128];
    conv_pre(mel_spec, conv_pre_out, conv_pre_weight, conv_pre_bias, frame_nums);


    for(i = 0; i < 245; i++){
        printf("%f, ", conv_pre_out[i][10]);
    }


    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%f seconds\n", duration);

    kiss_fft_free(cfg);

    //for
    //printf("size of %d", frame_nums);

    return 0;
}

                                                    
